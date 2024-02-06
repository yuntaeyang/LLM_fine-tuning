import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import TextStreamer, GenerationConfig

base_model_name='yanolja/KoSOLAR-10.7B-v0.1-deprecated' #
peft_model_name = '/home/yuntaeyang_0629/taeyang_2024/CDSI/keyword/sllm_ft/checkpoints/KoSOLAR-10.7B-v0.1-deprecated_lora_ft/checkpoint-13000' #
repo_name="KoSOLAR-10.7B-keword-v1.0" #

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda:0",torch_dtype=torch.float16)

peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

peft_model=peft_model.merge_and_unload()

peft_model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)

peft_model.push_to_hub(repo_name,use_temp_dir=False)
tokenizer.push_to_hub(repo_name,use_temp_dir=False)