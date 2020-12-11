#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


# In[4]:


class Actor(nn.Module):
    """Actor (policy) model."""

    def __init__(self, state_size, action_size, hidden_layer_sizes=[400, 300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_sizes: Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes
        # Hidden layers
        self.fc1 = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.bn2 = nn.BatchNorm1d(self.hidden_layer_sizes[1])
        # Output layer
        self.fc3 = nn.Linear(self.hidden_layer_sizes[1], action_size)
        
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim()==1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state)) 
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        return F.tanh(self.fc3(x))


# In[5]:


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_layer_sizes=[400, 300]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_sizes: Number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes
        # Hidden layers
        self.fcs1 = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0]+action_size, hidden_layer_sizes[1])
        # Output layer
        self.fc3 = nn.Linear(hidden_layer_sizes[1], 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim()==1:
            state = torch.unsqueeze(state, 0)  
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs) 
        x = torch.cat((xs, torch.squeeze(action, 0)), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

