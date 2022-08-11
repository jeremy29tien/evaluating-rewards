import dataclasses
import glob
import importlib
import os
from typing import Mapping, Tuple, TypeVar

import gym
from gpu_utils import determine_default_torch_device
from imitation.data import types
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluating_rewards import datasets
from evaluating_rewards.distances import tabular
from evaluating_rewards.rewards import base
from evaluating_rewards.analysis import util


from evaluating_rewards.distances.epic_sample import _tile_first_dim

import multiprocessing, ray
from ray.rllib.agents import ppo, sac


K = TypeVar("K")
DEVICE = None


def gt_reward_feeding(action, next_obs):
    ### GT REWARD WEIGHTS ###
    distance_weight = 1.0
    action_weight = 0.01
    food_reward_weight = 1.0
    task_success_threshold = 0.75
    velocity_weight = 0.25
    force_nontarget_weight = 0.01
    high_forces_weight = 0.05
    food_hit_weight = 1.0
    food_velocities_weight = 1.0
    dressing_force_weight = 0.01
    high_pressures_weight = 0.01

    # TODO: correct this hard-coded feature configuration
    distance = next_obs[0:3]
    spoon_force_on_human = next_obs[3]
    foods_in_mouth = next_obs[4]
    foods_on_floor = next_obs[5]
    foods_hit_human = next_obs[6]
    sum_food_mouth_velocities = next_obs[7]
    prev_spoon_pos_real = next_obs[8:11]
    robot_force_on_human = next_obs[11]

    total_force_on_human = spoon_force_on_human + robot_force_on_human

    reward_food = 20 * foods_in_mouth - 5 * foods_on_floor

    ### Get human preferences ###
    # end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
    # Slow end effector velocities
    # reward_velocity = -end_effector_velocity
    # < 10 N force at target
    reward_high_target_forces = 0 if spoon_force_on_human < 10 else -spoon_force_on_human
    # --- Scratching, Wiping ---
    # Any force away from target is low
    reward_force_nontarget = -(total_force_on_human - spoon_force_on_human)
    # --- Scooping, Feeding, Drinking ---
    reward_force_nontarget = -total_force_on_human
    # Penalty when robot spills food on the person
    reward_food_hit_human = -foods_hit_human
    # Human prefers food entering mouth at low velocities
    reward_food_velocities = -sum_food_mouth_velocities
    preferences_score = force_nontarget_weight * reward_force_nontarget + high_forces_weight * reward_high_target_forces + food_hit_weight * reward_food_hit_human + food_velocities_weight * reward_food_velocities
    ### End get human preferences ###

    reward_distance_mouth_target = -np.linalg.norm(distance)  # Penalize robot for distance between the spoon and human mouth.
    reward_action = -np.linalg.norm(action)  # Penalize actions

    reward = distance_weight * reward_distance_mouth_target + action_weight * reward_action + food_reward_weight * reward_food + preferences_score

    return reward


def gt_reward_scratchitch():
    pass

class Net(nn.Module):
    def __init__(self, env, hidden_dims=(128,64), augmented=False, fully_observable=False, pure_fully_observable=False, new_fully_observable=False, new_pure_fully_observable=False, num_rawfeatures=25, state_action=False, norm=False):
        super().__init__()

        if new_pure_fully_observable:
            if env == "feeding":
                raise Exception("NOT IMPLEMENTED.")
            elif env == "scratch_itch":
                input_dim = 20
        if new_fully_observable:
            if env == "feeding":
                raise Exception("NOT IMPLEMENTED.")
            elif env == "scratch_itch":
                input_dim = 43
        elif pure_fully_observable:
            if env == "feeding":
                input_dim = 19
            elif env == "scratch_itch":
                input_dim = 19
        elif fully_observable:
            if env == "feeding":
                input_dim = 40
            elif env == "scratch_itch":
                input_dim = 42
        elif augmented and state_action:
            # Feeding only
            input_dim = 35
        elif augmented:
            if env == "feeding":
                input_dim = num_rawfeatures + 3
            elif env == "scratch_itch":
                input_dim = num_rawfeatures + 2
        elif state_action:
            # Feeding only
            input_dim = 32
        else:
            if env == "feeding":
                input_dim = 25
            elif env == "scratch_itch":
                input_dim = 30

        self.normalize = norm
        if self.normalize:
            print("Normalizing input features...")
            self.layer_norm = nn.LayerNorm(input_dim)
        self.num_layers = len(hidden_dims) + 1

        self.fcs = nn.ModuleList([None for _ in range(self.num_layers)])
        if len(hidden_dims) == 0:
            self.fcs[0] = nn.Linear(input_dim, 1, bias=False)
        else:
            self.fcs[0] = nn.Linear(input_dim, hidden_dims[0])
            for l in range(len(hidden_dims)-1):
                self.fcs[l+1] = nn.Linear(hidden_dims[l], hidden_dims[l+1])
            self.fcs[len(hidden_dims)] = nn.Linear(hidden_dims[-1], 1, bias=False)

        print(self.fcs)

    def forward(self, x):
        '''calculate return of a state'''
        # Normalize features
        if self.normalize:
            x = self.layer_norm(x)

        for l in range(self.num_layers - 1):
            x = F.leaky_relu(self.fcs[l](x))
        r = self.fcs[-1](x)

        return r


def evaluate_models(
    models: Mapping[K, Net],
    batch: types.Transitions
) -> Mapping[K, np.ndarray]:
    # Call the cum_return() function in Net, but do it for just one observation/observation-action pair
    # Do this for each obs-action-nextobs set in the batch
    # Do all this for each model

    observations = batch.obs
    print("observations:", observations.shape)
    # TODO: Change this depending on the type of feature space we are using
    obs_1 = observations[:, 0:4]
    print("obs_1:", obs_1.shape)
    obs_2 = observations[:, 4:]
    print("obs_2:", obs_2.shape)
    actions = batch.acts
    print("actions:", actions.shape)
    next_observations = batch.next_obs
    input = np.concatenate((obs_1, actions, obs_2), axis=-1)
    input = torch.from_numpy(input).float().to(DEVICE)

    output = {k: torch.flatten(m.forward(input)).cpu().detach().numpy() for k, m in models.items() if k != "ground_truth"}

    gt_rewards = []
    for action, next_obs in zip(actions, next_observations):
        gt_reward = gt_reward_feeding(action, next_obs)
        gt_rewards.append(gt_reward)
    output["ground_truth"] = np.array(gt_rewards)
    return output


def sample_mean_rews(
    models: Mapping[K, Net],
    mean_from_obs: np.ndarray,
    act_samples: np.ndarray,
    next_obs_samples: np.ndarray,
    batch_size: int = 2 ** 28,
) -> Mapping[K, np.ndarray]:
    """
    Estimates the mean reward from observations `mean_from_obs` using given samples.

    Evaluates in batches of at most `batch_size` bytes to avoid running out of memory. Note that
    the observations and actions, being vectors, often take up much more memory in RAM than the
    results, a scalar value.

    Args:
        models: A mapping from keys to reward models.
        mean_from_obs: Observations to compute the mean starting from.
        act_samples: Actions to compute the mean with respect to.
        next_obs_samples: Next observations to compute the mean with respect to.
        batch_size: The maximum number of points to compute the reward with respect to in a single
            batch.

    Returns:
        A mapping from keys to NumPy array of shape `(len(mean_from_obs),)`, containing the
        mean reward of the model over triples:
            `(obs, act, next_obs) for act, next_obs in zip(act_samples, next_obs_samples)`
    """
    assert act_samples.shape[0] == next_obs_samples.shape[0]
    assert mean_from_obs.shape[1:] == next_obs_samples.shape[1:]

    # Compute indexes to not exceed batch size
    sample_mem_usage = act_samples.nbytes + mean_from_obs.nbytes
    obs_per_batch = batch_size // sample_mem_usage
    if obs_per_batch <= 0:
        msg = f"`batch_size` too small to compute a batch: {batch_size} < {sample_mem_usage}."
        raise ValueError(msg)
    idxs = np.arange(0, len(mean_from_obs), obs_per_batch)
    idxs = np.concatenate((idxs, [len(mean_from_obs)]))  # include end point

    # Compute mean rewards
    mean_rews = {k: [] for k in models.keys()}
    reps = min(obs_per_batch, len(mean_from_obs))
    act_tiled = _tile_first_dim(act_samples, reps)
    next_obs_tiled = _tile_first_dim(next_obs_samples, reps)

    for start, end in zip(idxs[:-1], idxs[1:]):
        obs = mean_from_obs[start:end]
        obs_repeated = np.repeat(obs, len(act_samples), axis=0)
        batch = types.Transitions(
            obs=obs_repeated,
            acts=act_tiled[: len(obs_repeated), :],
            next_obs=next_obs_tiled[: len(obs_repeated), :],
            dones=np.zeros(len(obs_repeated), dtype=np.bool),
            infos=None,
        )
        # base.evaluate_models returns a dictionary of model returns on the batch
        rews = evaluate_models(models, batch)
        print("length of obs", len(obs))
        rews = {k: v.reshape(len(obs), -1) for k, v in rews.items()}
        # print("rews:", rews)
        for k, m in mean_rews.items():
            means = np.mean(rews[k], axis=1)
            m.extend(means)

    mean_rews = {k: np.array(v) for k, v in mean_rews.items()}
    for v in mean_rews.values():
        assert v.shape == (len(mean_from_obs),)
    return mean_rews


def sample_canon_shaping(
    models: Mapping[K, Net],
    batch: types.Transitions,
    act_samples: datasets.SampleDist,
    next_obs_samples: datasets.SampleDist,
    discount: float = 1.0,
    p: int = 1,
) -> Mapping[K, np.ndarray]:
    r"""
    Canonicalize `batch` for `models` using a sample-based estimate of mean reward.

    Specifically, the algorithm works by sampling `n_mean_samples` from `act_dist` and `obs_dist`
    to form a dataset of pairs $D = \{(a,s')\}$. We then consider a transition dynamics where,
    for any state $s$, the probability of transitioning to $s'$ after taking action $a$ is given by
    its measure in $D$. The policy takes actions $a$ independent of the state given by the measure
    of $(a,\cdot)$ in $D$.

    This gives value function:
        \[V(s) = \expectation_{(a,s') \sim D}\left[R(s,a,s') + \gamma V(s')\right]\].
    The resulting shaping works out to be:
        \[F(s,a,s') = \gamma \expectation_{(a',s'') \sim D}\left[R(s',a',s'')\right]
                    - \expectation_{(a,s') \sim D}\left[R(s,a,s')\right]
                    - \gamma \expectation_{(s, \cdot) \sim D, (a,s') \sim D}\left[R(s,a,s')\right]
        \].

    If `batch` was a mesh of $S \times A \times S$ and $D$ is a mesh on $A \times S$,
    where $S$ and $A$ are i.i.d. sampled from some observation and action distributions, then this
    is the same as discretizing the reward model by $S$ and $A$ and then using
    `tabular.fully_connected_random_canonical_reward`. The action and next-observation in $D$ are
    sampled i.i.d., but since we are not computing an entire mesh, the sampling process introduces a
    faux dependency. Additionally, `batch` may have an arbitrary distribution.

    Empirically, however, the two methods produce very similar results. The main advantage of this
    method is its computational efficiency, for similar reasons to why random search is often
    preferred over grid search when some unknown subset of parameters are relatively unimportant.

    Args:
        models: A mapping from keys to reward models.
        batch: A batch to evaluate the models with respect to.
        act_samples: Samples of actions.
        next_obs_samples: Samples of observations. Same length as `act_samples`.
        discount: The discount parameter to use for potential shaping.
        p: Controls power in the L^p norm used for normalization.

    Returns:
        A mapping from keys to NumPy arrays containing rewards from the model evaluated on batch
        and then canonicalized to be invariant to potential shaping and scale.
    """
    # Sample-based estimate of mean reward
    n_mean_samples = len(act_samples)
    if len(next_obs_samples) != n_mean_samples:
        raise ValueError(f"Different sample length: {len(next_obs_samples)} != {n_mean_samples}")

    # EPIC only defined on infinite-horizon MDPs, so pretend episodes never end.
    # SOMEDAY(adam): add explicit support for finite-horizon?
    batch = dataclasses.replace(batch, dones=np.zeros_like(batch.dones))
    # base.evaluate_models returns a dictionary of model returns on the batch
    raw_rew = evaluate_models(models, batch)
    # print("raw_rew:", raw_rew)

    all_obs = np.concatenate((next_obs_samples, batch.obs, batch.next_obs), axis=0)
    unique_obs, unique_inv = np.unique(all_obs, return_inverse=True, axis=0)
    mean_rews = sample_mean_rews(models, unique_obs, act_samples, next_obs_samples)
    # print("mean_rews:", mean_rews)
    mean_rews = {k: v[unique_inv] for k, v in mean_rews.items()}

    dataset_mean_rews = {k: v[0:n_mean_samples] for k, v in mean_rews.items()}
    total_mean = {k: np.mean(v) for k, v in dataset_mean_rews.items()}

    batch_mean_rews = {k: v[n_mean_samples:].reshape(2, -1) for k, v in mean_rews.items()}

    # Use mean rewards to canonicalize reward up to shaping
    deshaped_rew = {}
    for k in models.keys():
        raw = raw_rew[k]
        mean = batch_mean_rews[k]
        total = total_mean[k]
        mean_obs = mean[0, :]
        mean_next_obs = mean[1, :]
        # Note this is the only part of the computation that depends on discount, so it'd be
        # cheap to evaluate for many values of `discount` if needed.
        deshaped = raw + discount * mean_next_obs - mean_obs - discount * total
        deshaped *= tabular.canonical_scale_normalizer(deshaped, p)
        deshaped_rew[k] = deshaped

    return deshaped_rew

def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False
    config['num_workers'] = num_processes
    print('num_workers:', num_processes)
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    # config['callbacks'] = CustomCallbacks
    # if algo == 'sac':
    #     config['num_workers'] = 1
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)  # 'assistive_gym:'+env_name
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)  # 'assistive_gym:'+env_name
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
            print("##################")
            print("Loading directly from a specific policy path:", policy_path)
            print("##################")
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                print("##################")
                print("Inferring policy to load based on env_name:", checkpoint_path)
                print("##################")

                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001, reward_net_path=None, indvar=None):
    if not coop and reward_net_path is not None and indvar is not None:
        env = gym.make('assistive_gym:'+env_name, reward_net_path=reward_net_path, indvar=indvar)
    elif not coop and reward_net_path is not None:
        env = gym.make('assistive_gym:' + env_name, reward_net_path=reward_net_path)
    elif not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def get_transitions(env_name, policy_path, seed, num_demos, noise_level, augmented, fully_observable, pure_fully_observable, new_fully_observable, new_pure_fully_observable, state_action):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    np.random.seed(seed)

    # Set up the environment
    env = make_env(env_name, seed=seed)

    # Load pretrained policy from file
    algo = 'ppo'

    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop=False, seed=seed)

    obses = []
    acts = []
    next_obses = []
    rewards = []
    traj_len = 0
    for demo in range(num_demos):
        traj = []
        total_reward = 0
        observation = env.reset()
        # print("Initial observation:", observation)
        info = None
        done = False
        while not done:
            # Take random action with probability noise_level
            if np.random.rand() < noise_level:
                action = env.action_space.sample()
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(observation)

            # Collect the data
            # print("Observation:", observation)
            # print("Action:", action)

            # FeedingSawyer
            # augmented (privileged) features: spoon-mouth distance, amount of food particles in mouth, amount of food particles on the floor
            # fully-observable: add previous end effector position, robot force on human, food information
            if env_name == "FeedingSawyer-v1":
                distance = np.linalg.norm(observation[7:10])
                if info is None:
                    foods_in_mouth = 0
                    foods_on_floor = 0
                    foods_hit_human = 0
                    sum_food_mouth_velocities = 0
                    prev_spoon_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                else:
                    foods_in_mouth = info['foods_in_mouth']
                    foods_on_floor = info['foods_on_ground']
                    foods_hit_human = info['foods_hit_human']
                    sum_food_mouth_velocities = info['sum_food_mouth_velocities']
                    prev_spoon_pos_real = info['prev_spoon_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                privileged_features = np.array([distance, foods_in_mouth, foods_on_floor])
                fo_features = np.concatenate(([foods_in_mouth, foods_on_floor, foods_hit_human,
                                               sum_food_mouth_velocities], prev_spoon_pos_real, [robot_force_on_human]))
                pure_obs = np.concatenate((observation[7:10], observation[24:25]))

            # ScratchItchJaco privileged features: end effector - target distance, total force at target
            if env_name == "ScratchItchJaco-v1":
                distance = np.linalg.norm(observation[7:10])
                if info is None:
                    tool_force_at_target = 0.0
                    prev_tool_pos_real = np.zeros(3)
                    robot_force_on_human = 0
                    prev_tool_force = 0
                    scratched = 0
                else:
                    tool_force_at_target = info['tool_force_at_target']
                    prev_tool_pos_real = info['prev_tool_pos_real']
                    robot_force_on_human = info['robot_force_on_human']
                    prev_tool_force = info['prev_tool_force']
                    scratched = info['scratched']
                privileged_features = np.array([distance, tool_force_at_target])
                fo_features = np.concatenate((prev_tool_pos_real, [robot_force_on_human, prev_tool_force]))
                new_fo_features = np.concatenate(
                    (prev_tool_pos_real, [robot_force_on_human, prev_tool_force, scratched]))
                pure_obs = np.concatenate((observation[0:3], observation[7:10], observation[29:30]))

            if new_pure_fully_observable:
                data = np.concatenate((pure_obs, action, new_fo_features))
                obses.append(np.concatenate((pure_obs, new_fo_features)))
            elif new_fully_observable:
                data = np.concatenate((observation, action, new_fo_features))
                obses.append(np.concatenate((observation, new_fo_features)))
            elif pure_fully_observable:
                data = np.concatenate((pure_obs, action, fo_features))
                obses.append(np.concatenate((pure_obs, fo_features)))
            elif fully_observable:
                data = np.concatenate((observation, action, fo_features))
                obses.append(np.concatenate((observation, fo_features)))
            elif augmented and state_action:
                data = np.concatenate((observation, action, privileged_features))
                obses.append(np.concatenate((observation, privileged_features)))
            elif augmented:
                data = np.concatenate((observation, privileged_features))
                obses.append(np.concatenate((observation, privileged_features)))
            elif state_action:
                data = np.concatenate((observation, action))
                obses.append(observation)
            else:
                data = observation
                obses.append(observation)

            acts.append(action)
            if info is not None:
                next_obses.append(obses[-1])

            # Step the simulation forward using the action from our trained policy
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                next_obses.append(obses[-1])  # Duplicate the last state at the end for next_obses

            traj.append(data)
            total_reward += reward
            # print("reward:", reward)
        # demos.append(traj)
        traj_len = len(traj)

        # print(total_reward)
        # total_rewards.append(total_reward)

    total_timesteps = num_demos * traj_len
    dones = np.zeros(total_timesteps, dtype=np.bool)

    return types.Transitions(
        obs=np.array(obses),
        acts=np.array(acts),
        next_obs=np.array(next_obses),
        dones=dones,
        infos=None,
    )


if __name__ == '__main__':
    env_name = "FeedingSawyer-v1"
    policy_path = "/home/jeremy/assistive-gym/trained_models/seed1/ppo/FeedingSawyer-v1/checkpoint_521/checkpoint-521"
    seed = 0
    num_demos = 5
    noise_level = 0.0
    augmented = False
    fully_observable = False
    pure_fully_observable = True
    new_fully_observable = False
    new_pure_fully_observable = False
    state_action = False
    n_samples = 512  # number of samples to take final expectation over
    n_mean_samples = 1000  # number of samples to use to canonicalize potential
    hidden_dims = (128, 64)
    model_kinds = [
        "learned_proper",
        "learned_hacking",
        "ground_truth"
    ]
    model_paths = [
        "/home/jeremy/assistive-gym/trex/models/feeding/vanilla/pure_fully_observable/324demos_allpairs_hdim128-64_100epochs_10patience_0001lr_000001weightdecay_seed0.params",
        "/home/jeremy/assistive-gym/trex/models/feeding/vanilla/pure_fully_observable/324demos_allpairs_hdim128-64_100epochs_10patience_0001lr_000001weightdecay_seed2.params",
        "placeholder"
    ]

    sess = tf.Session()
    with sess.as_default():
        tf.set_random_seed(seed)

    DEVICE = torch.device(determine_default_torch_device(not torch.cuda.is_available()))

    models = {}
    for i, kind in enumerate(model_kinds):
        if kind != "ground_truth":
            reward_net = Net("feeding", hidden_dims=hidden_dims, augmented=augmented,
                                  new_pure_fully_observable=new_pure_fully_observable,
                                  new_fully_observable=new_fully_observable,
                                  pure_fully_observable=pure_fully_observable,
                                  fully_observable=fully_observable,
                                  state_action=state_action)
            reward_net.load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))
            reward_net.to(DEVICE)
            models[kind] = reward_net
        else:
            models[kind] = None

    # Visitation distribution (obs,act,next_obs)
    batch = get_transitions(env_name, policy_path, seed, num_demos, noise_level, augmented, fully_observable, pure_fully_observable, new_fully_observable, new_pure_fully_observable, state_action)
    # print("batch:", batch)

    # Visitation distribution (obs,act,next_obs) is IID sampled from spaces
    # with datasets.transitions_factory_iid_from_sample_dist_factory(
    #         obs_dist_factory=datasets.sample_dist_from_space,
    #         act_dist_factory=datasets.sample_dist_from_space,
    #         obs_kwargs={"space": venv.observation_space},
    #         act_kwargs={"space": venv.action_space},
    #         seed=seed,
    # ) as iid_generator:
    #     batch = iid_generator(n_samples)

    env = make_env(env_name, seed=seed)
    with datasets.sample_dist_from_space(env.observation_space, seed=seed + 1) as obs_dist:
        next_obs_samples = obs_dist(n_mean_samples)
        next_obs_samples = np.concatenate((next_obs_samples[:, 7:10], next_obs_samples[:, 24:25]), axis=-1)  # TODO: need to change this hard-coded feature configuration
        next_obs_samples = np.concatenate((next_obs_samples, np.zeros((n_mean_samples, 8))), axis=-1)  # TODO: need to change this hard-coded 8
    with datasets.sample_dist_from_space(env.action_space, seed=seed + 2) as act_dist:
        act_samples = act_dist(n_mean_samples)

    # Finally, let's compute the EPIC distance between these models.
    # First, we'll canonicalize the rewards.
    with sess.as_default():
        deshaped_rew = sample_canon_shaping(
            models=models,
            batch=batch,
            next_obs_samples=next_obs_samples,
            act_samples=act_samples,
            # You can also specify the discount rate and the type of norm,
            # but defaults are fine for most use cases.
        )
    print("deshaped_rew['learned_proper']:", deshaped_rew['learned_proper'].shape)
    print("deshaped_rew['learned_hacking']:", deshaped_rew['learned_hacking'].shape)
    print("deshaped_rew['ground_truth']:", deshaped_rew['ground_truth'].shape)
    # Now, let's compute the Pearson distance between these canonicalized rewards.
    # The canonicalized rewards are quantized to `n_samples` granularity, so we can
    # then compute the Pearson distance on this (finite approximation) exactly.
    epic_distance = util.cross_distance(deshaped_rew, deshaped_rew, tabular.pearson_distance, parallelism=1)
    print("pearson_distance(deshaped_rew['ground_truth'], deshaped_rew['learned_hacking'])", tabular.pearson_distance(deshaped_rew['ground_truth'], deshaped_rew['learned_hacking']))
    print("pearson_distance(deshaped_rew['learned_hacking'], deshaped_rew['ground_truth'])", tabular.pearson_distance(deshaped_rew['learned_hacking'], deshaped_rew['ground_truth']))

    print("EPIC DISTANCE:", epic_distance)
    epic_df = pd.Series(epic_distance).unstack()
    # epic_df.index = epic_df.index.str.replace(r'evaluating_rewards/PointMass(.*)-v0', r'\1')
    # epic_df.columns = epic_df.columns.str.replace(r'evaluating_rewards/PointMass(.*)-v0', r'\1')
    print(epic_df)
