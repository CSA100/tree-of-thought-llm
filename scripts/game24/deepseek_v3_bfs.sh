python run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 902 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --backend deepseek-v3 \
    --temperature 0.7 \
    --max_tokens 40000 \
    ${@} 