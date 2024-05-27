import gymnasium as gym
import time
import random
import numpy as np
import tensorflow as tf

from skimage.color import rgb2gray
from skimage.transform import resize

from train_torch import ActorCritic
import torch


# 브레이크아웃에서의 테스트를 위한 A3C 에이전트 클래스
import torch.nn as nn
import torch.nn.functional as F


class A3CTestAgent:
    def __init__(self, action_size, state_size, model_path):
        self.action_size = action_size

        self.model = ActorCritic(action_size, state_size)
        self.model.load_state_dict(torch.load(model_path))

    def get_action(self, history):
        history = torch.FloatTensor(history / 255.0)
        policy = self.model(history)[0][0]
        policy = F.softmax(policy, dim=0)
        action_index = torch.multinomial(policy, num_samples=1).item()
        return action_index, policy


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode="constant") * 255
    )
    return processed_observe


if __name__ == "__main__":
    # 테스트를 위한 환경, 모델 생성
    env = gym.make("BreakoutDeterministic-v4", render_mode="human")
    state_size = (84, 84, 4)
    action_size = 3
    model_path = "./save_model/model.pth"
    render = True

    agent = A3CTestAgent(action_size, state_size, model_path)
    action_dict = {0: 1, 1: 2, 2: 3, 3: 3}

    num_episode = 10
    for e in range(num_episode):
        done = False
        dead = False

        score, start_life = 0, 5
        observe = env.reset()

        # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
        for _ in range(random.randint(1, 30)):
            observe, _, _, _, _ = env.step(1)

        # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
        state = pre_processing(observe)
        history = torch.FloatTensor(np.stack([state, state, state, state], axis=0))
        history = history.unsqueeze(0)
        print(f"history shape: {history.shape}")

        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            # 정책 확률에 따라 행동을 선택
            action, policy = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            real_action = action_dict[action]
            # 죽었을 때 시작하기 위해 발사 행동을 함
            if dead:
                action, real_action, dead = 0, 1, False

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, truncated, info = env.step(real_action)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
            next_history = torch.cat((next_state, history[:, :3, :, :]), dim=1)

            if start_life > info["lives"]:
                dead, start_life = True, info["lives"]

            score += reward

            if dead:
                print("dead")
                history = np.stack(
                    (next_state, next_state, next_state, next_state), axis=1
                )
                history = torch.FloatTensor(np.reshape([history], (1, 4, 84, 84)))
            else:
                history = next_history

            if done:
                # 각 에피소드 당 학습 정보를 기록
                print("episode: {:3d} | score : {:4.1f}".format(e, score))
