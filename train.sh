CUDA_LAUNCH_BLOCKING=1 python SFTT_DPO.py --train --batch_size 4 --epoch_size 3 --base_model "yanolja/KoSOLAR-10.7B-v0.1-deprecated" --wandb_key "c6c2bd0b3bad8601939e0320069e068cbf921b8f" --dataset_path "Ja-ck/Orca-DPO-Pairs-KO" --max_len 4096 --gradient_accumulation_steps 4 --project_desc "lora_finetuning" --name "lora_dpo" 

#CUDA_LAUNCH_BLOCKING=1 python main.py --train --batch_size 2 --epoch_size 5 --base_model "Upstage/SOLAR-10.7B-Instruct-v1.0" --wandb_key "c6c2bd0b3bad8601939e0320069e068cbf921b8f" --dataset_path "kyujinpy/KOR-OpenOrca-Platypus-v3" --max_len 4096 --gradient_accumulation_steps 8 --project_desc "lora_finetuning" --name "lora_ft" 
#CUDA_LAUNCH_BLOCKING=1 python SFTT.py --train --batch_size 1 --epoch_size 5 --base_model "Upstage/SOLAR-10.7B-Instruct-v1.0" --wandb_key "c6c2bd0b3bad8601939e0320069e068cbf921b8f" --dataset_path "kyujinpy/KOR-OpenOrca-Platypus-v3" --max_len 4096 --gradient_accumulation_steps 1 --project_desc "lora_finetuning" --name "lora_sftt" 

#CUDA_VISIBLE_DEVICES=0,1 python main.py --train --batch_size 1 --base_model "maywell/Synatra-7B-v0.3-RP" --wandb_key "d221e8d631a38e44cb1c4280769081527d5597f0" --dataset_path "kyujinpy/KOpen-platypus" --max_len 4096 --gradient_accumulation_steps 8 --project_desc "full finetuning" --name "full" 
