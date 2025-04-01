import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Aggregation: Q = V(s) + A(s, a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class EnhancedDQN:
    """
    自定义的 DQN 算法实现：
      - 使用 Dueling DQN 架构
      - 引入 Prioritized Experience Replay
      - 使用 Double DQN 的目标值计算
    该类实现了 .learn() 和 .save() 方法，接口与现有训练框架保持一致。
    """
    def __init__(self, policy, env, learning_rate=1e-4, buffer_size=10000, gamma=0.99,
                 batch_size=64, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500,
                 beta_start=0.4, beta_frames=1000):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy_net = DuelingDQNNetwork(state_dim, action_dim)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.num_timesteps = 0

    def learn(self, total_timesteps, callback=None, progress_bar=False, reset_num_timesteps=True):
        if reset_num_timesteps:
            self.num_timesteps = 0

        state, _ = self.env.reset()
        episode_reward = 0
        episode = 0

        while self.num_timesteps < total_timesteps:
            epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                      np.exp(-1.0 * self.num_timesteps / self.epsilon_decay)
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            self.num_timesteps += 1

            if done:
                state, _ = self.env.reset()
                print(f"Episode {episode} reward: {episode_reward}")
                episode_reward = 0
                episode += 1

            if len(self.replay_buffer.buffer) > self.batch_size:
                beta = min(1.0, self.beta_start + self.num_timesteps * (1.0 - self.beta_start) / self.beta_frames)
                b_state, b_action, b_reward, b_next_state, b_done, indices, weights = \
                    self.replay_buffer.sample(self.batch_size, beta)
                b_state = torch.FloatTensor(b_state)
                b_action = torch.LongTensor(b_action).unsqueeze(1)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1)
                b_next_state = torch.FloatTensor(b_next_state)
                b_done = torch.FloatTensor(b_done).unsqueeze(1)
                weights = torch.FloatTensor(weights).unsqueeze(1)

                # Double DQN: 策略网络选动作，目标网络计算 Q 值
                current_q_values = self.policy_net(b_state).gather(1, b_action)
                with torch.no_grad():
                    next_actions = self.policy_net(b_next_state).argmax(1, keepdim=True)
                    next_q_values = self.target_net(b_next_state).gather(1, next_actions)
                    target_q_values = b_reward + self.gamma * (1 - b_done) * next_q_values

                loss = (current_q_values - target_q_values).pow(2) * weights
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新优先级
                td_errors = (current_q_values - target_q_values).detach().cpu().numpy().squeeze()
                new_priorities = np.abs(td_errors) + 1e-6
                self.replay_buffer.update_priorities(indices, new_priorities)

            # 定期更新目标网络
            if self.num_timesteps % 1000 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return self

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), save_path)
