# Text-generation-with-GPT-2-
import os
import threading
import torch
import tkinter as tk
from tkinter import messagebox, scrolledtext
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# -------------------------
# Device Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./gpt2-finetuned"

# -------------------------
# Larger Training Dataset (Included)
# -------------------------
training_texts = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine learning allows computers to learn from data automatically.",
    "Deep learning uses neural networks inspired by the human brain.",
    "Entrepreneurship requires innovation, risk taking and leadership.",
    "Success comes from discipline, persistence and hard work.",
    "Technology is shaping the future of business and education.",
    "Data science combines statistics and computer science.",
    "Neural networks are powerful models for pattern recognition.",
    "Creative thinking leads to groundbreaking discoveries.",
    "AI applications include healthcare, robotics and finance."
]

# Convert to dataset
dataset = Dataset.from_dict({"text": training_texts})

# -------------------------
# Training Function
# -------------------------
def train_model_thread():
    train_button.config(state="disabled")
    status_label.config(text="Training started... Please wait.")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_total_limit=1,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    model.to(device)
    trainer.train()

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    status_label.config(text="Training completed successfully!")
    messagebox.showinfo("Success", "Model training completed!")

def train_model():
    threading.Thread(target=train_model_thread).start()

# -------------------------
# Generate Function
# -------------------------
def generate_text():
    try:
        prompt = prompt_entry.get()
        
        if not prompt.strip():
            messagebox.showwarning("Warning", "Please enter a prompt!")
            return
        
        # Use fine-tuned model if it exists, otherwise use base GPT-2
        model_to_use = MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "config.json")) else "gpt2"

        tokenizer = GPT2Tokenizer.from_pretrained(model_to_use)
        model = GPT2LMHeadModel.from_pretrained(model_to_use)
        model.to(device)

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs,
            max_length=120,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        output_box.delete(1.0, tk.END)
        output_box.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Error", f"Generation failed: {str(e)}")

# -------------------------
# GUI Layout
# -------------------------
root = tk.Tk()
root.title("GPT-2 Text Generation (Advanced GUI)")
root.geometry("750x550")

title = tk.Label(root, text="GPT-2 Text Generation", font=("Arial", 18))
title.pack(pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model, bg="lightblue")
train_button.pack(pady=5)

prompt_label = tk.Label(root, text="Enter Prompt:")
prompt_label.pack()

prompt_entry = tk.Entry(root, width=80)
prompt_entry.pack(pady=5)

generate_button = tk.Button(root, text="Generate Text", command=generate_text, bg="lightgreen")
generate_button.pack(pady=5)

output_box = scrolledtext.ScrolledText(root, width=90, height=18)
output_box.pack(pady=10)

status_label = tk.Label(root, text=f"Using device: {device}")
status_label.pack()

root.mainloop()
