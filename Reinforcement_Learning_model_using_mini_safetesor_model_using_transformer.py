"""
ppo_train_safetensors.py
Minimal PPO example using TRL (Hugging Face) with a small LM.
This script:
 - loads a small causal LM (distilgpt2 / gpt2)
 - wraps it with a value head for PPO
 - runs a tiny RL loop with a dummy reward function
 - saves the policy as .safetensors

Notes:
 - Replace `dummy_reward` with a real reward function or reward model for serious experiments.
 - For better VRAM usage, consider bitsandbytes + load_in_8bit (not shown).
"""

import os
import torch
from transformers import AutoTokenizer
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from datasets import Dataset
import random

# ---------- CONFIG ----------
MODEL_NAME = "distilgpt2"   # small model for toy experiments
OUTPUT_DIR = "./ppo_policy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2              # keep tiny for demonstration
EPOCHS = 2
MAX_GEN_TOKENS = 64
SEED = 42

# ---------- HELPERS ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dummy_reward(prompt: str, response: str) -> float:
    """
    Toy reward: +1 if response contains the word 'def' (i.e., generates code-like text),
    otherwise -0.1. Replace with a reward model for real work.
    """
    r = 1.0 if "def " in response or "def\n" in response else -0.1
    # add small length bonus to encourage non-empty responses
    r += min(len(response.split()), 10) * 0.01
    return float(r)

# ---------- MAIN ----------
def main():
    set_seed(SEED)

    # 1) tokenizer + model (with value head)
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AutoModelForCausalLMWithValueHead integrates value head used by PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME).to(DEVICE)

    # 2) PPO config and trainer
    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        ppo_epochs=1,
        learning_rate=1.41e-5,
        log_with=None,  # or "wandb"
    )

    # Create a reference model used by PPO to compute KL / keep policy close to reference
    reference_model = create_reference_model(model)

    ppo_trainer = PPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        **ppo_config.__dict__,
    )

    # 3) small prompt dataset (toy)
    prompts = [
        "Write a Python function to compute factorial using recursion:",
        "Write a Python function to check if a number is prime:",
        "Write a Python function to compute Fibonacci numbers:",
        "Explain quicksort in short steps:",
        "Write a Python function to reverse a string:"
    ]
    # duplicate to reach dataset size and shuffle
    prompts = prompts * 20
    random.shuffle(prompts)

    # 4) RL loop
    print("Starting PPO loop...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # iterate in batches
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            # encode prompts
            batch_encoding = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

            # generate responses from current policy via trainer (uses model.generate)
            # Use ppo_trainer.generate if available; fallback to model.generate
            with torch.no_grad():
                gen_ids = model.generate(
                    **batch_encoding,
                    max_length=batch_encoding["input_ids"].shape[1] + MAX_GEN_TOKENS,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # extract only generated part (strip the prompt tokens)
            responses = []
            for idx, g in enumerate(gen_ids):
                g = g.tolist()
                # find where prompt ends (simple approach)
                inp_len = (batch_encoding["input_ids"][idx] == tokenizer.eos_token_id).nonzero(as_tuple=False)
                # safer: just decode and remove prompt text
                full_text = tokenizer.decode(g, skip_special_tokens=True)
                # remove the prompt text prefix if present
                prompt_text = batch_prompts[idx]
                if full_text.startswith(prompt_text):
                    response_text = full_text[len(prompt_text):].strip()
                else:
                    # fallback: token-level slicing using lengths
                    response_text = tokenizer.decode(g[len(batch_encoding["input_ids"][idx]):], skip_special_tokens=True)
                responses.append(response_text)

            # compute rewards (list of floats)
            rewards = [dummy_reward(p, r) for p, r in zip(batch_prompts, responses)]

            # convert responses to tokens for PPO step
            # ppo_trainer.step expects raw prompts and generated tokens (or text), API may accept text directly
            try:
                # TRL commonly offers ppo_trainer.step(prompts, responses, rewards)
                ppo_trainer.step(batch_prompts, responses, rewards)
            except TypeError:
                # Some TRL versions expect tokenized inputs; fallback to lower-level API
                # Convert to list of token sequences
                resp_token_ids = [tokenizer.encode(r) for r in responses]
                ppo_trainer.step(batch_prompts, resp_token_ids, rewards)

            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  processed batch {i // BATCH_SIZE} - avg reward {sum(rewards)/len(rewards):.3f}")

    # 5) Save final policy as safetensors
    print("Saving final model (policy) to safetensors...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # save_pretrained supports safe_serialization=True to write .safetensors
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved to", OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
