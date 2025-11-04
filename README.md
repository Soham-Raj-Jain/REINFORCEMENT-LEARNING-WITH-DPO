# README - Colab 3: DPO Reinforcement Learning

### Overview

This notebook demonstrates Direct Preference Optimization (DPO), a reinforcement learning technique that trains models using preference pairs. Unlike traditional RLHF, DPO doesn't require a separate reward model, making it simpler and more stable.

### What You'll Learn

- Understanding preference-based training
- How DPO differs from traditional RLHF
- Formatting preference datasets
- Configuring beta parameter
- Aligning models to human preferences
- When to use DPO for model alignment

### Requirements

- Google Colab with T4 GPU (free tier)
- Approximately 15-20 minutes of training time
- Preference dataset with chosen/rejected pairs

### Dataset Format

DPO requires three columns per example:

```python
{
    "prompt": "How can I improve productivity?",
    "chosen": "Here are 5 evidence-based strategies: ...",
    "rejected": "Just work harder and never rest."
}
```

We use the Anthropic HH-RLHF dataset containing human preferences for helpfulness and harmlessness.

### Configuration

```python
beta = 0.1               # DPO temperature parameter
lora_r = 16              # Using LoRA for efficiency
learning_rate = 5e-5     # Lower than SFT
max_steps = 50
```

### Beta Parameter

Beta controls preference optimization strength:
- Low (0.01-0.1): Conservative, stays close to reference model
- Medium (0.1-0.3): Balanced, standard choice
- High (0.3-0.5): Aggressive preference learning

Start with 0.1 and adjust based on results.

### Training Process

1. Model processes both chosen and rejected responses
2. Calculates log probabilities for each
3. Optimizes to increase chosen probability
4. Decreases rejected probability
5. Beta controls optimization aggressiveness

### DPO vs Traditional RLHF

| Aspect | Traditional RLHF | DPO |
|--------|------------------|-----|
| Stages | Multi-stage | Single-stage |
| Reward Model | Required | Not needed |
| Complexity | High | Low |
| Stability | Can be unstable | More stable |
| Implementation | PPO training | Direct optimization |

### Output Files

```
./dpo_finetuned_smollm2/
├── adapter_config.json
├── adapter_model.bin
└── [Other config files]

./smollm2_dpo_merged/
└── [Merged model]
```

### Usage

#### Loading DPO Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "./smollm2_dpo_merged"
)

FastLanguageModel.for_inference(model)

response = model.generate(inputs, max_new_tokens=128)
```

#### Creating Preference Data

Collect human comparisons:
```python
# Generate multiple responses
responses = [generate(prompt) for _ in range(4)]

# Human ranks them: [4, 2, 1, 3] (scores)

# Create pairs
pairs = [
    {"prompt": prompt, "chosen": responses[2], "rejected": responses[3]},
    {"prompt": prompt, "chosen": responses[2], "rejected": responses[1]},
]
```

### When to Use DPO

Recommended For:
- Have preference comparison data
- Want to align model to human preferences
- Reduce harmful or unhelpful outputs
- Improve instruction following
- Make responses more helpful/honest
- Have completed supervised fine-tuning

Not Recommended For:
- Only have single-example data
- Need complex reward functions
- Teaching new factual knowledge
- Model already well-aligned

### Data Sources

Public Preference Datasets:
- Anthropic HH-RLHF: Helpfulness and harmlessness
- OpenAI WebGPT: Web search quality
- Stanford SHP: Reddit preferences
- UltraFeedback: GPT-4 rated responses

Creating Your Own:
1. Generate multiple responses per prompt
2. Have humans or GPT-4 rank them
3. Create chosen vs rejected pairs
4. Ensure diverse preference patterns

### Troubleshooting

Issue: Model not learning preferences
- Increase beta (try 0.2 or 0.3)
- Verify data format is correct
- Check that chosen responses are actually better
- Train for more steps

Issue: Model outputs become unnatural
- Lower beta (try 0.05)
- Reduce learning rate
- Check preference data quality
- Ensure not overfitting

Issue: No improvement over base model
- Verify DPO trainer is running correctly
- Check loss is decreasing
- Ensure sufficient preference pairs (500+)
- May need more training steps

### Best Practices

1. Use after supervised fine-tuning, not as first step
2. Ensure preference pairs are high quality
3. Balance helpful vs harmless preferences
4. Start with conservative beta (0.1)
5. Monitor both chosen and rejected loss
6. Evaluate on held-out preference pairs

### Extensions

#### Iterative DPO

```python
# Round 1: Initial DPO
train_dpo(model, preference_data_v1)

# Round 2: Collect new preferences from improved model
new_preferences = collect_preferences(improved_model)
train_dpo(model, preference_data_v2)
```

#### Multi-Objective DPO

Balance multiple objectives:
```python
preferences = {
    "helpfulness": helpful_pairs,
    "harmlessness": safe_pairs,
    "truthfulness": factual_pairs,
}

# Mix datasets appropriately
```

### Resources

- DPO Paper: "Direct Preference Optimization"
- Anthropic HH-RLHF: https://github.com/anthropics/hh-rlhf


