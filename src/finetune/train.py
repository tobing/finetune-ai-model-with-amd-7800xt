from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import torch

def main() -> None:
    print("Hello from finetune!")
    
    MODEL_ID = "Qwen/Qwen3-0.6B"
    OUTPUT_DIR = "Qwen3-0.6B-finetuned-chat"
    MAX_TRAINING_SAMPLES = 1024

    # Load dataset
    ds = load_dataset("csv", data_files="mydataset.csv")

    # Limit dataset size
    ds["train"] = ds["train"].select(
        range(min(MAX_TRAINING_SAMPLES, len(ds["train"])))
    )

    # Convert CSV rows to chat format
    def to_chat(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": example["question"],
                },
                {
                    "role": "assistant",
                    "content": example["answer"],
                },
            ]
        }

    ds_chat = ds.map(to_chat)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )

    # Convert chat messages to tokens
    def tokenize_function(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        return tokenizer(
            text,
            truncation=True,
            max_length=512,
        )

    tokenized_ds = ds_chat.map(
        tokenize_function,
        remove_columns=ds_chat["train"].column_names,
    )

    # Training configuration
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        num_train_epochs=50,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_ds["train"],
        args=args,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
