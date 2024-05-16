import sys
import gymnasium as gym
import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


# 정책 신경망과 가치 신경망 생성
class ContinuousA2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(ContinuousA2C, self).__init__()
        self.actor_fc1 = nn.Linear(state_size, 24)
        self.actor_mu = nn.Linear(24, action_size)
        self.actor_sigma = nn.Linear(24, action_size)

        self.critic_fc1 = nn.Linear(state_size, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, 1)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        mu = self.actor_mu(actor_x)
        sigma = F.softplus(self.actor_sigma(actor_x))
        sigma = sigma + 1e-5

        critic_x = F.tanh(self.critic_fc1(x))
        critic_x = F.tanh(self.critic_fc2(critic_x))
        value = self.critic_out(critic_x)

        return mu, sigma, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class ContinuousA2CAgent:
    def __init__(self, state_size, action_size, max_action):
        # 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        # 정책신경망과 가치신경망 생성
        self.model = ContinuousA2C(state_size, action_size)
        self.model.load_state_dict(torch.load("./save_model/model_best.pth"))

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = torch.distributions.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    gym.envs.register(
        id="CartPoleContinuous-v0",
        entry_point="env:ContinuousCartPoleEnv",
        max_episode_steps=500,
        reward_threshold=475.0,
    )

    env = gym.make("CartPoleContinuous-v0")
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = ContinuousA2CAgent(state_size, action_size, max_action)

    scores, episodes = [], []

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while not done:
            env.render()

            action = agent.get_action(state)
            action = action.detach().numpy()[0]
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))
