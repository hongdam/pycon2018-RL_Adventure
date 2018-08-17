from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.layers import NoisyLinear

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import imageio


# model은 학습모형에 맞는 모델을 class로 설계해주시면 됩니
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.features = nn.Sequential(
            # ((84 - 8 - 2*0) / 4) + 1 = 20
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # batch_size x 32 x 20 x 20
            nn.ReLU(),

            # ((20 - 4 - 2*0) / 2) + 1 = 9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # batch_size x 64 x 9 x 9
            nn.ReLU(),

            # ((9 - 3 - 2*0) / 2) + 1 = 4
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # batch_size x 64 x 4 x 4
            nn.ReLU()
        )

        self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)


    def forward(self, x):
        batch_size = x.size(0)

        x = x / 255.
        x = self.features(x)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


USE_CUDA = torch.cuda.is_available()  # False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_atoms = 51
Vmin = -10
Vmax = 10


# 모델을 초기화하고 load하는 부분
current_model = RainbowDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
current_model.load_state_dict(torch.load('./save_model/999398-BreakoutNoFrameskip-v4_RainbowDQN.pkl'))

if USE_CUDA:
    current_model = current_model.cuda()


EPISODE = 1
batch_size = 32
gamma = 0.99

losses = []
scores = []

global_steps = 1
rendering = True

for episode in range(1, EPISODE + 1):
    frames = []
    done = False
    score = 0
    state = env.reset()

    # for gif
    obs = env.render(mode='rgb_array')
    frames.append(obs)

    while True:
        if rendering:
            env.render()

        frames.append(env.render(mode='rgb_array'))
        action = current_model.act(state)

        next_state, reward, done, _ = env.step(action)

        state = next_state
        score += reward
        global_steps += 1

        if done:
            break

string = '{}_{}_{}.gif'
imageio.mimsave('./save_gif/' + string.format(env_id, "RainbowDQN", episode), frames, duration=0.0286)

env.close()
