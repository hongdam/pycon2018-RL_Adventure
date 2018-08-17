from common.layers import NoisyLinear

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import imageio
import gym


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

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

    def action(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


USE_CUDA = torch.cuda.is_available()  # False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)

env_id = "CartPole-v1"
env = gym.make(env_id)

num_atoms = 51
Vmin = -10
Vmax = 10

current_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
current_model.load_state_dict(torch.load('./save_model/27586-CartPole-v1_RainbowDQN.pkl'))

if USE_CUDA:
    current_model = current_model.cuda()

EPISODE = 10
rendering = True
frames = []

for episode in range(1, EPISODE + 1):
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
        action = current_model.action(state)

        next_state, reward, done, _ = env.step(action)

        state = next_state
        score += reward
        print(len(frames))

        if done:
            break

string = '{}_{}_{}.gif'
imageio.mimsave('./save_gif/' + string.format(env_id, "RainbowDQN", episode), frames, duration=0.0286)

env.close()
