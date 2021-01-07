import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None] # What does this mean?

        # TODO return the action that the policy prescribes √?
        # observation = ptu.from_numpy(observation.astype(np.float32))
        # action = self(observation)
        # return ptu.to_numpy(action)
        # if self.discrete:
        #     return ptu.to_numpy(self.logits_na(torch.from_numpy(observation).float()))
        # else:
        #     return ptu.to_numpy(self.mean_net(torch.from_numpy(observation).float()))
        return ptu.to_numpy(self.forward(ptu.from_numpy(observation)).rsample())
        # observation = ptu.from_numpy(observation.astype(np.float32))
        # action = self(observation).rsample()
        # return ptu.to_numpy(action)
        # mean, _ = self.forward(torch.from_numpy(observation).float())
        # return ptu.to_numpy(mean)
        raise NotImplementedError

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            return torch.distributions.categorical.Categorical(logits=self.logits_na(observation))
            # return self.logits_na(observation)
        else:
            return torch.distributions.normal.Normal(self.mean_net(observation), torch.exp(self.logstd)[None])
            # return self.mean_net(observation), self.logstd
        raise NotImplementedError


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        # obs = torch.from_numpy(MLPPolicy.get_action(self, observations))
        # obs = MLPPolicy.forward(self, torch.from_numpy(observations).float()).sample()
        # distribution = self.mean_net(observations) + Normal(torch.tensor([0.0]), torch.tensor([1.0])) * self.logstd
        # if self.discrete:
        #     distribution = MLPPolicy.forward(self, torch.from_numpy(observations).float())
        # else:
        #     mean, std = MLPPolicy.forward(self, torch.from_numpy(observations).float())
        #     print("1: ", mean.shape)
        #     print("2: ", self.ac_dim)
        #     print("3: ", std.shape)
        #     sampled_action = mean + torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample(mean.shape) * torch.exp(std)

        # obs = distribution.sample()
        # sampled_action.requires_grad = True
        actions = ptu.from_numpy(actions)
        # print("acs: ", ptu.to_numpy(acs)[0])
        observations = ptu.from_numpy(observations)
        # sampled_action = MLPPolicy.forward(self, ptu.from_numpy(observations)).rsample()
        action_distribution = self(observations)
        # print("sampled_action: ", ptu.to_numpy(sampled_action)[0])
        # acs.requires_grad = True
        # loss = self.loss(sampled_action, acs)
        loss = self.loss(action_distribution.rsample(), actions)
        # loss = -action_distribution.log_prob(actions).mean() # why is the loss defined this way instead of self.loss(sampled_action, acs)?
        self.optimizer.zero_grad()
        loss.backward()
        # added
        # if self.discrete:
        #     for param in self.logits_na.parameters():
        #         print(param.grad) 
        # else:
        #     for param in self.mean_net.parameters():
        #         print(param.grad) 
        self.optimizer.step()
        
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
