python run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 1000 \
    --naive_run \
    --prompt_sample standard_zero_shot \
    --n_generate_sample 1 \
    --backend deepseek-r1 \
    --temperature 0.7 \
    --max_tokens 40000 \
    ${@} 