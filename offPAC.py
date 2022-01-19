import torch
import gym
import math
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim+a_dim, 256, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0)
        return a0
    
    def put(self, *transition): 
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        # critic
        a1 = self.actor(s1).detach()
        adv = r1 + self.gamma * self.critic(s1, a1).detach() - self.critic(s0, a0)
        c_loss = torch.mean(adv**2)
        self.critic_optim.zero_grad()
        c_loss.backward()
        self.critic_optim.step()
        
        # actor
        a0 = agent.act(s0)
        dist = Normal(loc=a0, scale=1)
        a0 = dist.sample()
        log_pi = dist.log_prob(a0)
        a_loss = -adv.detach() * log_pi
        self.actor_optim.zero_grad()
        a_loss.mean().backward()
        self.actor_optim.step()
                                           
                                                                          
env = gym.make('Pendulum-v0')
env.reset()
# env.render()
wandb.init(project="PG", name="SPG")
params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.001, 
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000, 
    'batch_size': 32,
    }

agent = Agent(**params)

for episode in range(300):
    s0 = env.reset()
    episode_reward = 0
    
    for step in range(500):
        # env.render()
        a0 = agent.act(s0)
        dist = Normal(loc=a0, scale=1)
        a0 = dist.sample()
        log_pi = dist.log_prob(a0)
        a0 = [np.clip(a0.item(), -1, 1)*2]
        s1, r1, done, _ = env.step(a0)
        agent.put(s0, a0, r1, s1)

        episode_reward += r1 
        s0 = s1

        agent.learn()
    wandb.log({"reward": episode_reward}, step=episode)
    if episode % 10 == 0:
        print("Episode: %d, Reward: %f" % (episode, episode_reward))
