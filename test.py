import numpy as np
import torch

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)

# from src.envs import make_minigrid_env
# from src.envs import VecPyTorchMinigrid

from src.envs import make_crafter_env
from src.envs import VecPyTorchCrafter

from stable_baselines3.common.vec_env import VecMonitor as SB3VecMonitor
from stable_baselines3.common.vec_env import VecNormalize as SB3VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv as SB3DummyVecEnv

from src.envs import VecPyTorchProcgen


def evaluate(args, actor_critic, device):
    actor_critic.eval()

    # if 'MiniGrid' in args.env_name:
    #     venv = SB3DummyVecEnv([make_minigrid_env(args.test_env_name) for _ in range(args.num_processes)])
    #     venv = SB3VecNormalize(SB3VecMonitor(venv), norm_reward=True, norm_obs=False, clip_reward=1.)
    #     eval_envs = VecPyTorchMinigrid(venv, device)
    
    if 'crafter' in args.env_name:
        venv = SB3DummyVecEnv([make_crafter_env(log_file) for _ in range(args.num_processes)])
        venv = SB3VecNormalize(SB3VecMonitor(venv), norm_reward=True, norm_obs=False, clip_reward=1.)
        envs = VecPyTorchCrafter(venv, device)
    else:
        # Sample Levels From the Full Distribution 
        venv = ProcgenEnv(num_envs=1, env_name=args.env_name, \
            num_levels=0, start_level=0, distribution_mode=args.distribution_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            if args.separation == 'none' or args.separation == 'early' or args.separation == 'late':
                _, action, _ = actor_critic.act(obs)
            else:
                _, _, action, _ = actor_critic.act(obs)

        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
        .format(len(eval_episode_rewards), \
        np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    return eval_episode_rewards

