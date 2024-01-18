from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  BitsAndBytesConfig,
  TrainingArguments,
  pipeline, 
  logging, 
  TextStreamer,
  )

from accelerate import Accelerator
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import os, wandb, platform, warnings,sys
from datasets import load_dataset
from trl import DPOTrainer, SFTTrainer,DataCollatorForCompletionOnlyLM
import argparse
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
import ipdb
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,required=True)
    parser.add_argument("--dataset_path",type=str,required=True)
    parser.add_argument("--wandb_key",type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.02)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch_size', type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
    parser.add_argument("--gradient_checkpointing",type=bool,default=False,help="reduce required memory size but slower training")
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--project_desc",type=str, default = "Fine tuning llm")
    parser.add_argument("--name",type=str,default=None, help="file name to add")
    parser.add_argument("--full_ft",action="store_true",help="full finetuning otherwise lora")
    return parser.parse_args()

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def return_prompt_and_responses(samples):
  if samples['input'] == "":
    samples['text'] = "###instruction: " + samples['instruction'] + "###output:" + samples['output']
  else:
    samples['text'] = "###input: " + samples['input'] + "###instruction: " + samples['instruction'] + "###output:" + samples['output']
  return samples

def return_prompt_and_responses_dpo(samples):
  return {
        "prompt": samples['question'],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

def load_and_prepare_dataset(dataset_path, types):
    dataset=load_dataset(dataset_path, split="train")
    original_columns = dataset.column_names
    if types == "lora_sftt":
      dataset = dataset.map(
        return_prompt_and_responses
        )
      dataset = dataset.map( 
      batched=True,
      remove_columns=original_columns
      )
    else:
      dataset = dataset.map(
        return_prompt_and_responses_dpo,
        batched=True,
        remove_columns=original_columns
        )

    dataset=dataset.train_test_split(test_size=0.1)
    return dataset

def load_tokenizer(base_model_path,additional_special_tokens:list[str]=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if additional_special_tokens is not None:
      tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
      )

    tokenizer.padding_side="right"
    return tokenizer

def load_model(base_model_path, gradient_checkpointing=False,quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )
    return model

def load_model_dpo(base_model_path, gradient_checkpointing=False,quantization_config=None):

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )
    model.config.use_cache = False
    #model = prepare_model_for_kbit_training(model)

    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )

    return model, model_ref

def main(args):
    seed_everything(args.seed)
    ## load tokenizer and model
    if args.name == "lora_sftt":
      model=load_model(args.base_model,gradient_checkpointing=args.gradient_checkpointing,quantization_config=None)
      tokenizer=load_tokenizer(args.base_model)
      model.resize_token_embeddings(len(tokenizer))
      model.config.use_cache = False
    else:
      bnb_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16,
      )
      model, model_ref=load_model_dpo(args.base_model,gradient_checkpointing=args.gradient_checkpointing,quantization_config=None)
      tokenizer=load_tokenizer(args.base_model)
      model.resize_token_embeddings(len(tokenizer))
      model_ref.resize_token_embeddings(len(tokenizer))
      model.config.use_cache = False
      #model_ref.config.use_cache = False
  

    ## dataset
    if args.name == "lora_sftt":
      dataset = load_and_prepare_dataset(args.dataset_path, args.name)
    else:
      dataset = load_and_prepare_dataset(args.dataset_path, args.name)
    train_dataset=dataset["train"]
    eval_dataset=dataset["test"]

    response_template="###output:"
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template,add_special_tokens=False),tokenizer=tokenizer)
    
    ## wandb
    wandb.login(key = args.wandb_key)
    run = wandb.init(project=args.project_desc, job_type="training", anonymous="allow")

    if not args.full_ft:
      ## peft (lora)
      peft_config = LoraConfig(
              r=16,
              lora_alpha=16,
              lora_dropout=args.dropout_rate,
              bias="none",
              task_type="CAUSAL_LM",
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"]
          )
      
      model=get_peft_model(model,peft_config)

    if args.name is not None:
      output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}_{args.name}"
    else:
      output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}"

    if args.name == "lora_sftt":
      training_arguments = TrainingArguments(
        output_dir= output_dir,
        num_train_epochs= args.epoch_size,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        learning_rate= args.learning_rate,
        weight_decay= args.weight_decay,
        optim = "paged_adamw_32bit",
        evaluation_strategy="steps",
        save_steps= 500,
        logging_steps= 100,
        eval_steps=500,
        save_total_limit=5,
        # save_strategy="epoch"
        fp16= False,
        # bf16= True,
        # max_grad_norm= 0.3,
        # max_steps= -1,
        warmup_ratio= 0.1,
        # group_by_length= True,
        lr_scheduler_type= "linear",
        report_to="wandb"
        )

      trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config if not args.full_ft else None,
        max_seq_length= args.max_len,
        dataset_text_field="text",
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_arguments,
        # neftune_noise_alpha=5,
        # packing= True,
        )
    else:
      training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=1, 
        save_steps= 100,
        learning_rate=2e-5,
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
        )
      trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
        )
    
    if args.train:
      if args.ckpt_path is not None:
        trainer.train(args.ckpt_path)
      else:
        trainer.train()
      wandb.finish()

if __name__=="__main__":
  args=parse_args()
  main(args)