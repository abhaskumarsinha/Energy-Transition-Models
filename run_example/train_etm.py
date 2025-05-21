import argparse
import random
import sys

import numpy as np
import torch

# ─── Dummy gym / d4rl stubs ────────────────────────────────────────────────────

class Box:
    """Minimal stand-in for gym.spaces.Box"""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

class Env:
    """Dummy base class for gym.core.Env"""
    pass

class DummyEnv(Env):
    """
    Dummy environment returned by gym.make().
    Includes get_dataset() to replace d4rl environments.
    """
    def __init__(self, name: str):
        self.obs_dim = 17
        self.act_dim = 6
        self._max_episode_steps = 10

        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32
        )

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def get_dataset(self, **kwargs):
        """
        Dummy offline dataset: returns a dict with correct keys and shapes.
        """
        n_samples = 1000  # or any small number for testing

        return {
            "observations": np.random.randn(n_samples, self.obs_dim).astype(np.float32),
            "actions": np.random.randn(n_samples, self.act_dim).astype(np.float32),
            "next_observations": np.random.randn(n_samples, self.obs_dim).astype(np.float32),
            "rewards": np.random.randn(n_samples).astype(np.float32),
            "terminals": np.random.randint(0, 2, size=(n_samples,), dtype=np.bool_)
        }
    


class gym:
    """Dummy replacement for the gym module"""
    spaces = sys.modules[__name__]  # points to this module's Box
    core = sys.modules[__name__]    # points to this module's Env

    @staticmethod
    def make(task: str) -> DummyEnv:
        # ignore task name, return dummy
        return DummyEnv(task)


# ─── Your usual imports (offlinerlkit, etc.) ─────────────────────────────────

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, MLPETM
from offlinerlkit.dynamics import EnergyDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="etm")
    parser.add_argument("--task", type=str, default="walker2d-medium-v2")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--etm_lr", type=float, default=1e-3)
    parser.add_argument("--etm_hidden_dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--etm_activation", type=str, default="relu")
    parser.add_argument("--etm_with_reward", type=bool, default=True)
    parser.add_argument("--etm_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--etm_num_negative_samples", type=int, default=16)

    parser.add_argument("--etm_loss_type", type=str, default="info_nce")
    parser.add_argument("--etm_add_grad_penalty", type=bool, default=True)
    parser.add_argument("--etm_grad_penalty_margin", type=float, default=5.0)
    parser.add_argument("--etm_langevin_iter", type=int, default=50)

    parser.add_argument("--etm_max_epochs", type=int, default=300)
    parser.add_argument("--etm_batch_size", type=int, default=1024)

    parser.add_argument("--load_etm_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # make the dummy env
    env = gym.make(args.task)

    # load your offline dataset (unaffected by the dummy env)
    dataset = env.get_dataset()

    args.obs_shape = env.observation_space.shape
    args.action_dim = int(np.prod(env.action_space.shape))
    args.max_action = float(1.0)

    print(args.device)
    # create dynamics
    energy_model = MLPETM(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.etm_hidden_dims,
        activation=args.etm_activation,
        with_reward=args.etm_with_reward,
        device=args.device
    )
    etm_optim = torch.optim.Adam(energy_model.parameters(), lr=args.etm_lr, weight_decay=1e-4)
    scaler = StandardScaler()
    delta_scaler = StandardScaler(box_normalization=True)

    termination_fn = get_termination_fn(task=args.task)
    etm_dynamics = EnergyDynamics(
        energy_model, etm_optim, scaler, delta_scaler, termination_fn,
        softmax_temperature=args.etm_softmax_temperature,
        num_negative_samples=args.etm_num_negative_samples,
        etm_loss_type=args.etm_loss_type,
        add_grad_penalty=args.etm_add_grad_penalty,
        grad_penalty_margin=args.etm_grad_penalty_margin,
        langevin_iterations=args.etm_langevin_iter,
    )
    if args.load_etm_path:
        etm_dynamics.load(args.load_etm_path)

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    real_buffer.size = real_buffer._size

    # set up logger
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=[])
    output_config = {
        "consoleout_backup": "stdout",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    print("Replay buffer loaded:")
    print("observations:", dataset["observations"].shape)
    print("actions:", dataset["actions"].shape)
    print("next_observations:", dataset["next_observations"].shape)
    print("rewards:", dataset["rewards"].shape)
    print("terminals:", dataset["terminals"].shape)

    print("Replay buffer size after loading:", real_buffer._size)

    # train
    etm_dynamics.learn_energy(
        real_buffer.sample_all(),
        logger,
        eval_holdout=True,
        max_epochs=args.etm_max_epochs,
        batch_size=args.etm_batch_size,
    )


if __name__ == "__main__":
    train()
