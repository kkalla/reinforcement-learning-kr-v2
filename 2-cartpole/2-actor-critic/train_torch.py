import sys
import gymnasium as gym
import pylab
import numpy as np
import torch


# 정책 신경망과 가치 신경망 생성
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class A2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.actor_fc = nn.Linear(state_size, 24)
        self.actor_out = nn.Linear(24, action_size)
        self.critic_fc1 = nn.Linear(state_size, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, 1)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc(x))
        policy = F.softmax(self.actor_out(actor_x), dim=1)

        critic_x = F.tanh(self.critic_fc1(x))
        critic_x = F.tanh(self.critic_fc2(critic_x))
        value = self.critic_out(critic_x)

        return policy, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태, 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.state_size, self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy, _ = self.model(torch.Tensor(state))
        policy = np.array(policy[0].detach())
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.parameters()
        state = torch.Tensor(state)
        action = torch.Tensor([action]).type(torch.int64)
        reward = torch.Tensor([reward])
        next_state = torch.Tensor(next_state)
        done = torch.Tensor([done])

        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]

        # 정책 신경망 오류 함수 구하기
        one_hot_action = F.one_hot(action, num_classes=self.action_size)
        action_prob = torch.sum(one_hot_action * policy, dim=1)
        cross_entropy = -torch.log(action_prob + 1e-5)
        advantage = (target - value[0]).detach()
        actor_loss = torch.mean(cross_entropy * advantage)

        # 가치 신경망 오류 함수 구하기
        critic_loss = 0.5 * torch.square(target.detach() - value[0])
        critic_loss = torch.mean(critic_loss)

        # 하나의 오류 함수로 만들기
        loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().numpy()


if __name__ == "__main__":
    # CartPole-v1 environment, maximum number of timesteps is 500
    # render_mode="human"
    env = gym.make("CartPole-v1")
    # Get the state and action size from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create the A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        loss_list = []
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Reward of 0.1 at each timestep, -1 reward if the episode ends prematurely or reaches 500 timesteps
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # Train the agent at each timestep
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                # Print the training results for each episode
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print(
                    "episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(
                        e, score_avg, np.mean(loss_list)
                    )
                )

                # Save the training results as a graph for each episode
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, "b")
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                # Terminate when the moving average score is above 400
                if score_avg > 400:
                    torch.save(agent.model.state_dict(), "./save_model/model.pth")
                    sys.exit()
