import numpy
import torch
import time
import argparse

from actor_critic_mlp import ActorCriticMLP


class ActorCriticMLPPolicy:
    def __init__(self,
                 num_actor_obs=39,
                 num_critic_obs=168,
                 num_actions=10,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation="elu", ):
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.activation = activation
        self.init_noise_std = 1.0

        # create model
        self.rl_network = ActorCriticMLP(
            self.num_actor_obs,
            self.num_critic_obs,
            self.num_actions,
            self.actor_hidden_dims,
            self.critic_hidden_dims,
            self.activation,
            self.init_noise_std,
        )

    def load_model(self, model_path):
        self.model_path = model_path
        self.model_jit_path = model_path[:-3] + "_jit.pt"
        print("model_path: \n", self.model_path)
        print("model_jit_path: \n", self.model_jit_path)

        loaded_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        print("loaded_dict : \n", loaded_dict)

        self.rl_network.load_state_dict(loaded_dict["model_state_dict"])
        print("self.rl_network : \n", self.rl_network)

    def transtojit(self):
        self.rl_network.to("cpu")
        obs_numpy = numpy.zeros(self.num_actor_obs)
        obs_torch = torch.tensor(
            obs_numpy, dtype=torch.float, device="cpu", requires_grad=False
        )
        traced_script_module = torch.jit.trace(self.rl_network, obs_torch)

        # save jit model
        traced_script_module.save(self.model_jit_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python scripts for generating the jit models.')
    parser.add_argument('model_path', type=str, help='Path to the model file')

    args = parser.parse_args()

    # create model
    model_policy = ActorCriticMLPPolicy(num_actor_obs=(49 + 1 * 3) * 5,
                                        num_critic_obs=(178 + 1 * 3) * 5,
                                        num_actions=(12 + 1))

    # load model
    model_policy.load_model(model_path=args.model_path)

    # transform to jit
    model_policy.transtojit()

    # load jit model
    model_jit = torch.jit.load(model_policy.model_jit_path)
    print("model_jit = \n", model_jit)

    obs_numpy = numpy.zeros(model_policy.num_actor_obs)
    obs_torch = torch.tensor(
        obs_numpy, dtype=torch.float, device="cpu", requires_grad=False
    )
    print(obs_torch.shape)

    output_policy = model_policy.rl_network.forward(obs_torch)
    print("model_policy forward output = ", output_policy)

    time_start = time.time()
    output_jit = model_jit.forward(obs_torch)
    time_end = time.time()
    print("model_jit forward time = ", time_end - time_start)
    print("model_jit forward output = ", output_jit)
