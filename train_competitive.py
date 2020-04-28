"""
This file implements the train scripts for both A2C and PPO

You need to implement all TODOs in this script.

Note that you may find this file is completely compatible for both A2C and PPO.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import argparse
from collections import deque

import gym
import numpy as np
import torch
from competitive_pong import make_envs

import torch.optim as optim
from torch import nn

from core.a2c_trainer import A2CTrainer, a2c_config
from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import verify_log_dir, pretty_print, Timer, evaluate, \
    mirror_evaluate, summary, save_progress, FrameStackTensor, step_envs, mirror_step_envs

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="",
    type=str,
    help="(Required) The algorithm you want to run. Must in [PPO, A2C]."
)
parser.add_argument(
    "--log-dir",
    default="/tmp/ierg6130_hw4/",
    type=str,
    help="The path of directory that you want to store the data to. "
         "Default: /tmp/ierg6130_hw4/ppo/"
)
parser.add_argument(
    "--load-dir",
    default="",
    type=str,
    help="The path of directory where the trained model resides"
)
parser.add_argument(
    "--load-suffix",
    default="",
    type=str,
    help="The suffix of the the trained model file"
)
parser.add_argument(
    "--agent",
    default="",
    type=str,
    help="The name of the agent with whom you want to compete"
)
parser.add_argument(
    "--num-envs",
    default=15,
    type=int,
    help="The number of parallel environments. Default: 15"
)
parser.add_argument(
    "--learning-rate", "-LR",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--seed",
    default=100,
    type=int,
    help="The random seed. Default: 100"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--env-id",
    default="CompetitivePong-v0",
    type=str,
    help="The environment id, should be in ['CompetitivePong-v0', "
         "'CartPole-v0', 'CompetitivePongTournament-v0']. "
         "Default: CompetitivePong-v0"
)
args = parser.parse_args()


def train(args):
    # Verify algorithm and config
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    elif algo == "A2C":
        config = a2c_config
    else:
        raise ValueError("args.algo must in [PPO, A2C]")
    config.num_envs = args.num_envs
    # assert args.env_id in ["CompetitivePong-v0", "CartPole-v0", "CompetitivePongTournament-v0", "CompetitivePongDouble-v0"]

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    log_dir = verify_log_dir(args.log_dir, algo)

    # Create vectorized environments
    num_envs = args.num_envs
    env_id = args.env_id
    envs = make_envs(
        env_id=env_id,
        seed=seed,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=True,
        resized_dim=config.resized_dim
    )
    eval_envs = make_envs(
        env_id=env_id,
        seed=seed,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=False,
        resized_dim=config.resized_dim
    )
    test = env_id == "CartPole-v0"
    tournament = env_id == "CompetitivePongTournament-v0"
    frame_stack = 4 if not test else 1
    if tournament:
        assert algo == "PPO", "Using PPO in tournament is a good idea, " \
                              "because of its efficiency compared to A2C."

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainer(envs, config, frame_stack, _test=test)
    else:
        trainer = A2CTrainer(envs, config, frame_stack, _test=test)

    # print('args.load_dir', args.load_dir)
    # print('args.load_suffix', args.load_suffix)
    if args.load_dir and args.load_suffix:
        trainer.load_w(args.load_dir, args.load_suffix)

    # Create a placeholder tensor to help stack frames in 2nd dimension
    # That is turn the observation from shape [num_envs, 1, 84, 84] to
    # [num_envs, 4, 84, 84].
    if args.env_id == "CompetitivePongDouble-v0":
        frame_stack_tensor = FrameStackTensor(num_envs, envs.observation_space[0].shape, frame_stack, config.device)
        mirror_frame_stack_tensor = FrameStackTensor(num_envs, envs.observation_space[1].shape, frame_stack, config.device)
    else:
        frame_stack_tensor = FrameStackTensor(num_envs, envs.observation_space.shape, frame_stack, config.device)

    # Setup some stats helpers
    # episode_rewards = np.zeros([num_envs, 2 if args.env_id == "CompetitivePongDouble-v0" else 1], dtype=np.float)
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    episode_length_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    if args.env_id == "CompetitivePongDouble-v0":
        mirror = PPOTrainer(envs, config, frame_stack, _test=test)
        if args.load_dir and args.load_suffix:
            mirror.load_w(args.load_dir, args.load_suffix)

        obs = envs.reset()
        frame_stack_tensor.update(obs[0])
        trainer.rollouts.observations[0].copy_(frame_stack_tensor.get())

        mirror_frame_stack_tensor.update(obs[1])
        mirror.rollouts.observations[0].copy_(mirror_frame_stack_tensor.get())
    else:
        obs = envs.reset()
        frame_stack_tensor.update(obs)
        trainer.rollouts.observations[0].copy_(frame_stack_tensor.get())

    if args.agent and args.env_id != "CompetitivePongDouble-v0":
        envs.reset_opponent(agent_name=args.agent)

    best_index = 1

    while True:  # Break when total_steps exceeds maximum value
        # ===== Sample Data =====
        with sample_timer:
            for index in range(config.num_steps):
                if args.env_id == "CompetitivePongDouble-v0":
                    # print(torch.sum(trainer.rollouts.observations - torch.flip(mirror.rollouts.observations, [4])))
                    # print(torch.sum(trainer.rollouts.observations - mirror.rollouts.observations))
                    # print(torch.sum(trainer.rollouts.observations - trainer.rollouts.observations))
                    with torch.no_grad():
                        values, actions, action_log_prob = trainer.compute_action(trainer.rollouts.observations[index])
                        mirror_values, mirror_actions, mirror_action_log_prob = mirror.compute_action(mirror.rollouts.observations[index])
                        # mirror_values, mirror_actions, mirror_action_log_prob = trainer.compute_action(trainer.rollouts.observations[index])

                    cpu_actions = actions.view(-1).cpu().numpy()
                    mirror_cpu_actions = mirror_actions.view(-1).cpu().numpy()

                    # print('action', list(zip(cpu_actions, mirror_cpu_actions)))
                    # Step the environment
                    # (Check step_envs function, you need to implement it)
                    obs, reward, done, info, masks, total_episodes, \
                    total_steps, episode_rewards = mirror_step_envs(
                        cpu_actions, mirror_cpu_actions, envs, episode_rewards, frame_stack_tensor, mirror_frame_stack_tensor,
                        reward_recorder, episode_length_recorder, total_steps,
                        total_episodes, config.device, test)

                    # print('reward', reward)
                    rewards = torch.from_numpy(reward[:,0].astype(np.float32)).view(-1, 1).to(config.device)
                    mirror_rewards = torch.from_numpy(reward[:,1].astype(np.float32)).view(-1, 1).to(config.device)

                    # Store samples
                    trainer.rollouts.insert( frame_stack_tensor.get(), actions.view(-1, 1), action_log_prob, values, rewards, masks)
                    mirror.rollouts.insert( mirror_frame_stack_tensor.get(), mirror_actions.view(-1, 1), mirror_action_log_prob, mirror_values, mirror_rewards, masks)
                else:
                    # Get action
                    # [TODO] Get the action
                    # Hint:
                    #   1. Remember to disable gradient computing
                    #   2. trainer.rollouts is a storage containing all data
                    #   3. What observation is needed for trainer.compute_action?
                    # print('trainer.rollouts.observations[index]', trainer.rollouts.observations[index].shape)
                    with torch.no_grad():
                        values, actions, action_log_prob = \
                                trainer.compute_action(trainer.rollouts.observations[index])
                    # values = trainer.compute_values()
                    # actions = trainer.compute_action()
                    # action_log_prob = None

                    cpu_actions = actions.view(-1).cpu().numpy()

                    # Step the environment
                    # (Check step_envs function, you need to implement it)
                    obs, reward, done, info, masks, total_episodes, \
                    total_steps, episode_rewards = step_envs(
                        cpu_actions, envs, episode_rewards, frame_stack_tensor,
                        reward_recorder, episode_length_recorder, total_steps,
                        total_episodes, config.device, test)

                    rewards = torch.from_numpy(
                        reward.astype(np.float32)).view(-1, 1).to(config.device)

                    # Store samples
                    trainer.rollouts.insert(
                        frame_stack_tensor.get(), actions.view(-1, 1),
                        action_log_prob, values, rewards, masks)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(
                    trainer.rollouts.observations[-1])
            trainer.rollouts.compute_returns(next_value, config.GAMMA)

        # ===== Update Policy =====
        def update_uni(this_trainer, this_rollout):
            num_sgd_steps = 10
            mini_batch_size = 500
            uni_step = 10
            uni_perturb_pth = "./data/uni_perturb.pt"

            try:
                uni_perturb = torch.load(uni_perturb_pth)
            except:
                uni_perturb = torch.zeros(this_trainer.num_feats)

            uni_perturb = uni_perturb.cuda()
            perturb_delta = nn.Parameter(torch.zeros(this_trainer.num_feats).cuda())
            perturb_delta.requires_grad = True

            optimizer = optim.Adam(nn.ParameterList([perturb_delta]), lr=5e-3, eps=1e-5)
            sm = nn.Softmax(dim=1)

            advantages = this_rollout.returns[:-1] - this_rollout.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
            for e in range(num_sgd_steps):
                data_generator = this_rollout.feed_forward_generator(
                    advantages, mini_batch_size)

                loss_epoch = []
                for sample in data_generator:
                    observations_batch, actions_batch, return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ = sample
                    perturb_delta.value = torch.zeros_like(perturb_delta)
                    for uni_stepping in range(uni_step):
                        logits, _ = this_trainer.model(observations_batch + uni_perturb + perturb_delta)
                        target_prob = torch.zeros_like(logits)
                        target_prob[0] = 1
                        # loss = torch.pow(target_prob - torch.exp(logits), 2).mean() + torch.pow(perturb_delta, 2).mean()
                        loss = torch.pow(target_prob - sm(logits), 2).mean() + torch.pow(perturb_delta, 2).mean()

                        # print('loss:', loss)
                        loss_epoch.append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
                        optimizer.step()
                    uni_perturb += perturb_delta 

            norm = torch.pow(uni_perturb, 2).mean()
            delta_norm = torch.pow(perturb_delta, 2).mean()
            print('norm, delta_norm, np.mean(loss_epoch):', norm.item(), delta_norm.item(), np.mean(loss_epoch))
            torch.save(uni_perturb.cpu(), uni_perturb_pth)

        with update_timer:
            # policy_loss, value_loss, dist_entropy, total_loss = trainer.update(trainer.rollouts)
            update_uni(trainer, trainer.rollouts)
            trainer.rollouts.after_update()
            if args.env_id == "CompetitivePongDouble-v0":
                mirror.rollouts.after_update()

        iteration += 1
        continue

        # ===== Reset opponent if in tournament mode =====
        # if tournament and iteration % config.num_steps == 0:
        if tournament:
            # Randomly choose one agent in each iteration
            envs.reset_opponent()

        # ===== Evaluate Current Policy =====
        if iteration % config.eval_freq == 0:
            eval_timer = Timer()
            if args.env_id == "CompetitivePongDouble-v0":
                evaluate_rewards, evaluate_lengths = mirror_evaluate( trainer, mirror, eval_envs, frame_stack, 20)
            else:
                evaluate_rewards, evaluate_lengths = evaluate( trainer, eval_envs, frame_stack, 20)
            evaluate_stat = summary(evaluate_rewards, "episode_reward")
            if evaluate_lengths:
                evaluate_stat.update(
                    summary(evaluate_lengths, "episode_length"))
            evaluate_stat.update(dict(
                win_rate=float(
                    sum(np.array(evaluate_rewards)[0,:] >= 0) / len(
                        evaluate_rewards)),
                evaluate_time=eval_timer.now,
                evaluate_iteration=iteration
            ))

        # ===== Log information =====
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder,
                                                "episode_reward"),
                training_episode_length=summary(episode_length_recorder,
                                                "episode_length"),
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg +
                                 update_timer.avg
                ),
                iteration=iteration
            )
            if tournament:
                stats["opponent"] = envs.current_agent_name

            progress.append(stats)
            pretty_print({
                "===== {} Training Iteration {} =====".format(
                    algo, iteration): stats
            })

            training_episode_reward=summary(reward_recorder, "episode_reward")
            # print(training_episode_reward)

            if args.env_id == "CompetitivePongDouble-v0" and training_episode_reward["episode_reward_mean"] > 5.0:
                trainer.save_w(log_dir, "best{}".format(best_index))
                mirror.load_w(log_dir, "best{}".format(best_index))
                best_index += 1


        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        # [TODO] Stop training when total_steps is greater than args.max_steps
        # if total_steps > args.max_steps:
        #     break

        iteration += 1

    trainer.save_w(log_dir, "final")
    envs.close()


if __name__ == '__main__':
    train(args)
