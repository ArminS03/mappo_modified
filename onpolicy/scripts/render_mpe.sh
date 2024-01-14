#!/bin/sh
env="MPE"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 xvfb-run -s \"-screen 0 1400x900x24\" python render/render_mpe.py --save_gifs True \
     --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
     --num_landmarks ${num_landmarks} --seed ${seed} --share_policy False \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --model_dir "/content/mappo_modified/onpolicy/scripts/results/MPE/simple_speaker_listener/rmappo/check/run1/models"
done
