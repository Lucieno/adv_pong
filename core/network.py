"""
This file implement neural network for you.

Nothing you need to do in this file.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0),
                                          nn.init.calculate_gain('relu'))
        # The network structure is designed for 42X42 observation.
        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1))
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.maxpool2 = nn.MaxPool2d(3, stride=3)

        self.conv3_1 = init_(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        # self.conv3_2 = init_(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.maxpool3 = nn.MaxPool2d(3, stride=3, padding=1)

        self.conv4_1 = init_(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        # self.conv4_2 = init_(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.maxpool4 = nn.MaxPool2d(3, stride=3)

        flatten_size = 256
        # self.fc1 = nn.Linear(flatten_size, 512)

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(flatten_size, 1))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.actor_linear = init_(nn.Linear(flatten_size, num_actions))

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs / 255.0))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3_1(x))
        # x = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        x = self.maxpool4(x)
        x = x.view(x.size(0), -1)
        value = self.critic_linear(x)
        logits = self.actor_linear(x)
        return logits, value

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(
            torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.policy = nn.Linear(100, output_size)
        self.value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.policy(x)
        value = self.value(x)
        return action, value
