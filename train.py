from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from transformers import TrainingArguments
import torch
from trl import SFTTrainer

# huggingface-cli login
# git config --global credential.helper store

model_id = "meta-llama/Llama-2-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, token="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", quantization_config=quantization_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token="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", trust_remote_code=True)
# Set it to a new token to correctly attend to EOS tokens.
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config)

train_dataset = load_dataset("albieta1/odoo", split="train[:1%]")

output_dir = "albieta1/odoo-nice-first-vibes-test"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    push_to_hub=True,
)

def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    packing=True,
    dataset_text_field="id",
    tokenizer=tokenizer,
    max_seq_length=1024,
    formatting_func=formatting_func,
)

trainer.train()