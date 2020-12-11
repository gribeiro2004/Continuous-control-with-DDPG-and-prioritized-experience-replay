#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

#Import model
from Continuous_Control_model import Actor, Critic


from prioritized_memory import Memory


# In[2]:


#Hyperparameters
BUFFER_SIZE = int(1e6)      # replay buffer size
BATCH_SIZE = 256           # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-4                  # for soft update of target parameters
LR_ACTOR = 1e-4            # learning rate of the actor 
LR_CRITIC = 1e-4            # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2         # how often to update the networks


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.buffer_size = buffer_size
        self.memory = Memory(capacity=self.buffer_size)  # internal memory using SumTree
        self.batch_size = batch_size
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    
    def step(self, state, action, reward, next_state, done, batch_size=BATCH_SIZE, update_every=UPDATE_EVERY):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        
        self.add(state, action, reward, next_state, done)
            
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if self.memory.tree.n_entries >= batch_size:
                experiences, idxs, is_weights = self.sample()
                self.learn(experiences, idxs, is_weights)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            #action = [act + self.noise.sample() for act in action]
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, idxs, is_weights, batch_size=BATCH_SIZE, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        #Loss calculation
        critic_loss = (torch.from_numpy(is_weights).float().to(device) * F.mse_loss(Q_expected, Q_targets)).mean()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        #Introducing gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) 
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)        
        
        
        
        #.......................update priorities in prioritized replay buffer.......#
        #Calculate errors used in prioritized replay buffer
        errors = (Q_expected-Q_targets).squeeze().cpu().data.numpy()
        
        # update priority
        for i in range(batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
    def add(self, state, action, reward, next_state, done, gamma=GAMMA):
        """Add a new experience to memory."""
        
        next_state_torch = torch.from_numpy(next_state).float().to(device)
        reward_torch = torch.unsqueeze(torch.from_numpy(np.array(reward)).float().to(device), 1)
        done_torch = torch.unsqueeze(torch.from_numpy(np.array(done).astype(np.uint8)).float().to(device), 1)
        state_torch = torch.from_numpy(state).float().to(device)
        action_torch = torch.from_numpy(action).float().to(device)
        
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()
        with torch.no_grad():
            action_next = self.actor_target(next_state_torch)
            Q_target_next = self.critic_target(next_state_torch, action_next)
            Q_target = reward_torch + (gamma * Q_target_next * (1 -done_torch))
            Q_expected = self.critic_local(state_torch, action_torch)
        self.actor_local.train()
        self.critic_target.train()
        self.critic_local.train()
        
        #Error used in prioritized replay buffer
        error = (Q_expected - Q_target).squeeze().cpu().data.numpy()
        
        #Adding experiences to prioritized replay buffer
        for i in np.arange(len(reward)):
            self.memory.add(error[i], (state[i], action[i], reward[i], next_state[i], done[i]))
        
    
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences, idxs, is_weights = self.memory.sample(self.batch_size)

        states = np.vstack([e[0] for e in experiences])
        states = torch.from_numpy(states).float().to(device)
        
        actions = np.vstack([e[1] for e in experiences])
        actions = torch.from_numpy(actions).float().to(device)
        
        rewards = np.vstack([e[2] for e in experiences])
        rewards = torch.from_numpy(rewards).float().to(device)
        
        next_states = np.vstack([e[3] for e in experiences])
        next_states = torch.from_numpy(next_states).float().to(device)
        
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        dones = torch.from_numpy(dones).float().to(device)
        
        return (states, actions, rewards, next_states, dones), idxs, is_weights
    
    
    
    
    
# In[5]:


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
