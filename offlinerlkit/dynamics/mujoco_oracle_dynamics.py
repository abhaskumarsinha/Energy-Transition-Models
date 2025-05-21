import numpy as np
from typing import Tuple, Dict

# ─── Dummy stubs for MuJoCo environment and model ─────────────────────────────

class DummyModel:
    """
    Dummy replacement for mujoco.Env.model
    Holds dimensions for qpos (nq) and qvel (nv).
    """
    def __init__(self, nq: int = 7, nv: int = 7):
        self.nq = nq
        self.nv = nv


class MujocoEnv:
    """
    Dummy stand‐in for gym.envs.mujoco.mujoco_env.MujocoEnv.
    You can override nq and nv as needed when instantiating.
    """
    def __init__(self, nq: int = 7, nv: int = 7):
        # model dimensions
        self.model = DummyModel(nq=nq, nv=nv)
        # track steps
        self._elapsed_steps = 0
        # store last state
        self._qpos = np.zeros(self.model.nq)
        self._qvel = np.zeros(self.model.nv)

    def reset(self) -> np.ndarray:
        """Dummy reset: zero out state and return a flat obs vector."""
        self._elapsed_steps = 0
        self._qpos = np.zeros(self.model.nq)
        self._qvel = np.zeros(self.model.nv)
        # drop first xpos element to mimic original length
        return np.concatenate([self._qpos, self._qvel])[1:]

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Store the provided qpos/qvel as the internal state."""
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self._qpos = qpos.copy()
        self._qvel = qvel.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Dummy step:
          - increments _elapsed_steps
          - returns the same state as next_obs
          - reward is dot(state, action)
          - episode ends after 100 steps
        """
        self._elapsed_steps += 1
        # next_obs is just current qpos/qvel concatenation (minus the first element)
        full_obs = np.concatenate([self._qpos, self._qvel])
        next_obs = full_obs[1:]
        # dummy reward
        reward = float(np.dot(full_obs, np.pad(action, (0, full_obs.size - action.size))))
        # terminal if too many steps
        done = self._elapsed_steps >= 100
        info = {"elapsed_steps": self._elapsed_steps}
        return next_obs, reward, done, info


# ─── Oracle Dynamics class unchanged, but now uses our dummy MujocoEnv ───────

class MujocoOracleDynamics:
    def __init__(self, env: MujocoEnv) -> None:
        self.env = env

    def _set_state_from_obs(self, obs: np.ndarray) -> None:
        # if obs is missing the first xpos element, pad it
        expected = self.env.model.nq + self.env.model.nv
        if obs.size == expected - 1:
            obs = np.concatenate([np.zeros(1), obs])
        qpos = obs[: self.env.model.nq]
        qvel = obs[self.env.model.nq :]
        self.env._elapsed_steps = 0
        self.env.set_state(qpos, qvel)

    def step(
        self, obs: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        if obs.ndim > 1 or action.ndim > 1:
            raise ValueError("Only 1D obs and action supported.")
        _ = self.env.reset()  # reset to zero state
        self._set_state_from_obs(obs)
        return self.env.step(action)


# ─── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = MujocoEnv(nq=5, nv=5)
    oracle = MujocoOracleDynamics(env)

    # sample dummy observation and action
    obs = np.arange(9).astype(float)  # size = nq+nv-1 = 9
    action = np.array([1.0, 2.0, 3.0])

    next_obs, reward, done, info = oracle.step(obs, action)
    print("next_obs:", next_obs)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
