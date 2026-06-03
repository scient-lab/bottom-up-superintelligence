'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.

Adapted from https://github.com/simplescaling/s1/blob/main/train/sft.py
'''

import os
import torch
import torch.distributed as dist
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_from_disk
import transformers
import trl
from huggingface_hub import HfApi
from socket import gethostname
from peft import LoraConfig, get_peft_model, TaskType
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/QwQ-32B")
    block_size: int = field(default=32768)
    wandb_project: str = field(default="sft_kg")
    wandb_dir: str = field(default="/wandb_logs")
    train_dataset_path: str = field(default="/curriculum_training_data/tokenized_curriculum_dataset_hop_3_decontaminated/")
    dagger: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # kg-pipeline fork-patch: removed `push_to_hub` and `hub_repo_id` fields
    # from TrainingConfig. They were never used by the driver (which doesn't
    # pass --push_to_hub / --hub_repo_id), and trl.SFTConfig already inherits
    # both from transformers.TrainingArguments. With newer transformers
    # (>=4.46) registering both --push_to_hub and --push-to-hub as aliases
    # for each dataclass field, having the same field on two dataclasses in
    # HfArgumentParser((TrainingConfig, trl.SFTConfig)) raises:
    #   argparse.ArgumentError: conflicting option strings: --push_to_hub,
    #   --push-to-hub
    # If you want to push to HF Hub, pass --push_to_hub=True --hub_model_id=...
    # directly — SFTConfig handles it.

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_DIR'] = self.wandb_dir

    
def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    # Load model with LoRA config if enabled
    if config.use_lora:
        # Prepare model for LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["wte","lm_head"]
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Print trainable vs total parameters

        if args.gradient_checkpointing:
            model.enable_input_require_grads()
       
    dataset = load_from_disk(config.train_dataset_path)
    # setting up trainer
    # kg-pipeline fork-patch: upstream references undefined `model_path` here
    # (NameError at runtime). Use config.model_name — same value the model
    # load above used, so tokenizer matches the model.
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    instruction_template = "<|im_start|>user\n"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    tokenizer.add_special_tokens({'pad_token': '<|fim_pad|>'})
    # Only compute loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
        instruction_template=instruction_template
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
        peft_config=lora_config if config.use_lora else None
    )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if getattr(config, "push_to_hub", False):
        # Push model and tokenizer to the Hugging Face Hub
        repo_id = getattr(config, "hub_repo_id", None)
        if repo_id is None:
            raise ValueError("hub_repo_id must be set in config to push to hub.")
        trainer.push_to_hub()
        tokenizer.push_to_hub(repo_id)
        
if __name__ == "__main__":
   
    train()

   
