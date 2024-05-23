import torch
import torch.nn as nn

from mlp import MLP


class ActorCriticMLP(nn.Module):

    def __init__(
            self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation="elu",
            init_noise_std=1.0,
            fixed_std=False,
            **kwargs,
    ):
        print("----------------------------------")
        print("ActorCriticMLP")

        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super(ActorCriticMLP, self).__init__()

        # Policy
        actor_num_input = num_actor_obs
        actor_num_output = num_actions
        actor_activation = activation

        self.actor = MLP(
            actor_num_input,
            actor_num_output,
            actor_hidden_dims,
            actor_activation,
            norm="none",
        )

        print(f"Actor MLP: {self.actor}")

        # Value function
        critic_num_input = num_critic_obs
        critic_num_output = 1
        critic_activation = activation

        self.critic = MLP(
            critic_num_input,
            critic_num_output,
            critic_hidden_dims,
            critic_activation,
            norm="none",
        )

        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        actions = self.actor(observations.detach())
        return actions

    def evaluate(self, critic_observations, **kwargs):
        values = self.critic(critic_observations)
        return values
