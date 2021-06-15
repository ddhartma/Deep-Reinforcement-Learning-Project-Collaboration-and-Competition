import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """ Actor (Policy) Model
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """ Initialize parameters and build model

            INPUTS:
            ------------
                state_size - (int) Dimension of each state
                action_size - (int) Dimension of each action
                seed - (int) Random seed
                fc1_units - (int) Number of nodes in first hidden layer
                fc2_units - (int) Number of nodes in second hidden layer

            OUTPUTS:
            ------------
                No direct
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        #self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset the parameters of the network (random uniform distribution)

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                No direct
        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Build an actor (policy) network that maps states -> actions.

            INPUTS:
            ------------
                state - (torch tensor) --> tensor([[33x floats], x size of minibatch])

            OUTPUTS:
            ------------
                actions - (torch tensor) --> tensor([[4x floats], x size of minibatch])
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))
        return actions


class Critic(nn.Module):
    """ Critic (Value) Model
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """ Initialize parameters and build model

            INPUTS:
            ------------
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fcs1_units (int): Number of nodes in the first hidden layer
                fc2_units (int): Number of nodes in the second hidden layer

            OUTPUTS:
            ------------
                No direct
        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()


    def reset_parameters(self):
        """ Reset the parameters of the network (random uniform distribution)

            INPUTS:
            ------------
                None

            OUTPUTS:
            ------------
                No direct
        """

        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values.

            INPUTS:
            ------------
                state - (torch tensor) --> tensor([[33x floats], x size of minibatch])
                action - (torch tensor) --> tensor([[4x floats], x size of minibatch])

            OUTPUTS:
            ------------
                Q_values - (torch tensor) --> tensor([[1x float], x size of minibatch])
        """

        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        Q_values = self.fc3(x)
        return Q_values
