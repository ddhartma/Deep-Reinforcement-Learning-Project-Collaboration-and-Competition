import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_DECAY = 0.999     # NOISE Decay (not needed)
UPDATE_EVERY = 20       # Update experience tuple every ... time steps
NUM_UPDATES = 10        # Number of updates --> call of learn function
NOISE_SIGMA = 0.05      # Diffusion parameter for Ornstein-Uhlenbeck noise, weight parameter for adding random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment
    """

    def __init__(self, state_size, action_size, num_agents, random_seed, checkpoint_actor=None, checkpoint_critic=None):
        """Initialize an Agent object.

            INPUTS:
            ------------
                state_size - (int) dimension of each state
                action_size - (int) dimension of each action
                random_seed - (int) random seed
                checkpoint_actor -  (string) path to actor model parameters
                checkpoint_critic - (string) path to critic model parameters

            OUTPUTS:
            ------------
                No direct
        """
        print('Training on ', device)
        print()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Load model parameters
        if checkpoint_actor != None:
            self.actor_local.load_state_dict(torch.load(checkpoint_actor))
            self.actor_target.load_state_dict(torch.load(checkpoint_actor))
        if checkpoint_critic != None:
            self.critic_local.load_state_dict(torch.load(checkpoint_critic))
            self.critic_target.load_state_dict(torch.load(checkpoint_critic))

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.noise_decay = NOISE_DECAY

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0

        print('------- ACTOR ------')
        print('self.actor_local',self.actor_local)
        print()
        print('self.actor_target',self.actor_target)
        print()
        print('self.actor_optimizer',self.actor_optimizer)
        print()
        print('------- CRITIC ------')
        print('self.critic_local',self.critic_local)
        print()
        print('self.critic_target',self.critic_target)
        print()
        print('self.critic_optimizer',self.critic_optimizer)
        print()


    def step(self, state, action, reward, next_state, done, timestep):
        """ Save experience in replay memory, and use random sample from buffer to learn.

            INPUTS:
            ------------
                state - (numpy array) with shape (24,) state vector for the actual agent
                action - (numpy array) with shape (2,) actual action values for the agent
                reward - (float) actual reward
                next_state - (numpy array) with shape (24,) next state vector for the actual agent
                done - (bool) if True epsiode is finished for the agent
                timestep - (int) actual time step

            OUTPUTS:
            ------------
                No direct
        """

        #print('STEP FUNCTION ---------------------')
        #print('state')
        #print(state)
        #print('state shape', state.shape)
        #print('type', type(state))
        #print()
        #print('action ', action)
        #print('action shape', action.shape)
        #print('type', type(action))
        #print()
        #print('reward ', reward)
        #print('type', type(reward))
        #print()
        #print('next_state')
        #print(next_state)
        #print('next_state shape', next_state.shape)
        #print('type', type(next_state))
        #print()
        #print('done')
        #print(done)
        #print('type', type(done))
        #print()

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        #print('STEP FUNCTION ---------------------')
        #print('self.memory')
        #print(self.memory)
        #print('type', type(self.memory))
        #print()



        # Learn, if enough samples are available in memory
        # Learn every <UPDATE_EVERY> time steps.
        # Repeat learning process <NUM_UPDATES> times
        if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

                #print('STEP FUNCTION ---------------------')
                #print('experiences')
                #print(experiences)
                #print('experiences len', len(experiences))
                #print('type', type(experiences))
                #print()

    def act(self, state, add_noise=True):
        """ Returns actions for given state as per current policy.

            INPUTS:
            ------------
                state - (numpy array) with shape (1,24), a vector with 24 float values, contains the agent's position, rotation, velocity, angular-velocity of the arm. Given this information, the agent has to learn how to best select actions.
                add_noise - (bool) if True add noise to chosen action based on Ornstein-Uhlenbeck

            OUTPUTS:
            ------------
                clipped_action - (numpy array) with shape (1,2) each value of action is clipped withion (-1, 1)
        """
        #print('ACT FUNCTION ---------------------')
        #print('state')
        #print(state)
        #print('state shape', state.shape)
        #print('type', type(state))
        #print()


        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() #* self.noise_decay
            #action *=  self.epsilon
            self.noise_decay *= self.noise_decay
        else:
            clipped_action =  action

        clipped_action = np.clip(action, -1, 1)

        #print('clipped_action')
        #print(clipped_action)
        #print('state shape', clipped_action.shape)
        #print('type', type(clipped_action))
        #print()

        return clipped_action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """ Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

            INPUTS:
            ------------
                experiences - (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples
                gamma - (float) discount factor

            OUTPUTS:
            ------------
                No direct
        """
        #print('LEARN FUNCTION ---------------------')
        #print('experiences')
        #print(experiences)
        #print('experiences len', len(experiences))
        #print('type', type(experiences))
        #print()

        states, actions, rewards, next_states, dones = experiences

        #print('states')
        #print(states)
        #print('states shape', states.shape)
        #print('type', type(states))
        #print()
        #print('actions ', actions)
        #print('actions shape', actions.shape)
        #print('type', type(actions))
        #print()
        #print('rewards ', rewards)
        #print('rewards shape', rewards.shape)
        #print('type', type(rewards))
        #print()
        #print('next_states')
        #print(next_states)
        #print('next_states shape', next_states.shape)
        #print('type', type(next_states))
        #print()
        #print('dones')
        #print(dones)
        #print('type', type(dones))
        #print()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        #print()
        #print('actions_next')
        #print(actions_next)
        #print('actions_next shape', actions_next.shape)
        #print('type', type(actions_next))
        #print()
        #print('Q_targets_next')
        #print(Q_targets_next)
        #print('Q_targets_next shape', Q_targets_next.shape)
        #print('type', type(Q_targets_next))
        #print()
        #print('Q_targets')
        #print(Q_targets)
        #print('Q_targets shape', Q_targets.shape)
        #print('type', type(Q_targets))
        #print()
        #print('Q_expected')
        #print(Q_expected)
        #print('Q_expected shape', Q_expected.shape)
        #print('type', type(Q_expected))
        #print()
        #print('critic_loss')
        #print(critic_loss)
        #print('critic_loss shape', critic_loss.shape)
        #print('type', type(critic_loss))
        #print()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        #print()
        #print('actions_pred')
        #print(actions_next)
        #print('actions_pred shape', actions_pred.shape)
        #print('type', type(actions_pred))
        #print()
        #print('actor_loss')
        #print(actor_loss)
        #print('actor_loss shape', actor_loss.shape)
        #print('type', type(actor_loss))

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ---------------------------- update noise ---------------------------- #

        self.reset()

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            INPUTS:
            ------------
                local_model - (PyTorch model) weights will be copied from
                target_model - (PyTorch model) weights will be copied to
                tau - (float) interpolation parameter

            OUTPUTS:
            ------------
                No direct
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """ Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=NOISE_SIGMA):
        """ Initialize parameters and noise process.

            INPUTS:
            ------------
                size - (int) here its value is 4 (because of 4 action values)
                seed - (int) random seed
                mu - (float) default=0.
                theta - (float) default=0.15,
                sigma - (float) default=0.2

            OUTPUTS:
            ------------
                No direct
        """
        #print('OUNoise FUNCTION ---------------------')
        #print('size')
        #print(size)
        #print('type', type(size))
        #print()
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """ Reset the internal state (= noise) to mean (mu).

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                No direct
        """

        self.state = copy.copy(self.mu)

    def sample(self):
        """ Update internal state and return it as a noise sample.

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                self.state - (numpy array) with shape (4,), vector of float values, same shape as action numpy array in Agent.act() function.
                             Noise sample which will be added to 'action' in act function
        """

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        #print('OUNoise SAMPLE FUNCTION ---------------------')
        #print('self.state')
        #print(self.state)
        #print('self.state shape', self.state.shape)
        #print('type', type(self.state))
        #print()

        return self.state

class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ Initialize a ReplayBuffer object.

            INPUTS:
            ------------
                buffer_size - (int) maximum size of buffer
                batch_size - (int)size of each training batch

            OUTPUTS:
            ------------
                No direct
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """ Add a new experience to memory.

            INPUTS:
            ------------
                state - (numpy array) with shape (24,) state vector for the actual agent
                action - (numpy array) with shape (4,) actual action values for the agent
                reward - (float) actual reward
                next_state - (numpy array) with shape (24,) next state vector for the actual agent
                done - (bool) if True epsiode is finished for the agent

            OUTPUTS:
            ------------
                No direct
        """
        #print('REPLAY ADD FUNCTION ---------------------')
        #print('state')
        #print(state)
        #print('state shape', state.shape)
        #print('type', type(state))
        #print()
        #print('action ', action)
        #print('action shape', action.shape)
        #print('type', type(action))
        #print()
        #print('reward ', reward)
        #print('type', type(reward))
        #print()
        #print('next_state')
        #print(next_state)
        #print('next_state shape', next_state.shape)
        #print('type', type(next_state))
        #print()
        #print('done')
        #print(done)
        #print('type', type(done))
        #print()

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences from memory.

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                states - (torch tensor)
                actions - (torch tensor)
                rewards - (torch tensor)
                next_states - (torch tensor)
                dones - (torch tensor)
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        #print('REPLAY SAMPLE FUNCTION ---------------------')
        #print('experiences')
        #print(experiences)
        #print('experiences len', len(experiences))
        #print('type', type(experiences))
        #print()
        #print('states')
        #print(states)
        #print('states shape', states.shape)
        #print('type', type(states))
        #print()
        #print('actions ', actions)
        #print('actions shape', actions.shape)
        #print('type', type(actions))
        #print()
        #print('rewards ', rewards)
        #print('rewards shape', rewards.shape)
        #print('type', type(rewards))
        #print()
        #print('next_states')
        #print(next_states)
        #print('next_states shape', next_states.shape)
        #print('type', type(next_states))
        #print()
        #print('dones')
        #print(dones)
        #print('type', type(dones))
        #print()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Return the current size of internal memory.

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                current_size - (int) length of self.memory
        """

        current_size = len(self.memory)
        return current_size
