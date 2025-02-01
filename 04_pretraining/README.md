
# GPT Model Implementation with PyTorch

## Introduction
This repository implements a basic GPT model with key components, including tokenization, multi-head attention, transformer blocks, and training with a dataloader. It utilizes the PyTorch framework and the `tiktoken` tokenizer for efficient tokenization. The model can be trained on custom datasets and generate text based on the learned patterns.

## Features
- **Tokenizer**: Utilizes `tiktoken` for encoding and decoding text using a GPT-2 tokenizer.
- **Model Architecture**: The model uses a Transformer-based architecture with multi-head self-attention.
- **Sliding Window Dataset**: The dataset is split into overlapping chunks using a sliding window approach for efficient training.
- **Loss Calculation**: Implements a loss calculation method using cross-entropy for training evaluation.
- **Text Generation**: Includes a function to generate text based on a provided prompt.
- **Visualization**: Includes a function to plot training and validation loss with respect to epochs and tokens seen.

## Installation
To install the required dependencies, you can run:

```bash
pip install torch tiktoken matplotlib
```

## Components

### 1. `GPTDatasetV1` (Dataset Class)
- Tokenizes the input text using `tiktoken` and splits it into overlapping sequences for training.
- Each sequence is divided into input and target sequences for the model.

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        ...
    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...
```

### 2. `MultiHeadAttention` (Self-Attention Layer)
- Implements multi-head self-attention with a causal mask to ensure autoregressive behavior.
- The attention mechanism allows the model to focus on different parts of the input sequence for each token.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        ...
    def forward(self, x):
        ...
```

### 3. `TransformerBlock` (Transformer Layer)
- Combines self-attention and feedforward networks with residual connections and layer normalization.
- Multiple transformer blocks are stacked to build the model.

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        ...
    def forward(self, x):
        ...
```

### 4. `GPTModel` (Main GPT Model)
- Embeds tokens and positions and passes them through multiple transformer blocks.
- Outputs logits which can be used for generating text.

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        ...
    def forward(self, in_idx):
        ...
```

### 5. `Loss Calculation`
- Implements functions for calculating loss over a batch or dataloader.
- Uses cross-entropy loss to compare the predicted logits with the target tokens.

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    ...
def calc_loss_loader(data_loader, model, device, num_batches=None):
    ...
```

### 6. `Text Generation`
- A simple text generation function that takes a prompt and generates a sequence of text based on the learned model.
- The model uses a causal mask to generate text token by token.

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    ...
```

### 7. `Plot Losses`
- A helper function to visualize training and validation loss with respect to epochs and tokens seen.
- Displays the losses and saves the plot as a PDF.

```python
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    ...
```

## Usage

### 1. Tokenization and Dataset Creation
- Tokenize the input text using `tiktoken` and prepare the dataset for training using the `GPTDatasetV1` class.
- Example:
```python
tokenizer = tiktoken.get_encoding("gpt2")
txt = "Your input text here."
dataset = GPTDatasetV1(txt, tokenizer, max_length=256, stride=128)
```

### 2. Training the Model
- Create a model configuration and initialize the model.
- Example:
```python
cfg = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 256,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}
model = GPTModel(cfg)
```

### 3. Text Generation
- Generate text from a prompt using the trained model.
- Example:
```python
start_context = "Once upon a time"
generate_and_print_sample(model, tokenizer, device, start_context)
```

## Evaluation and Loss Tracking
Evaluate the model on training and validation datasets and track the losses over epochs and tokens seen.

```python
train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter=10)
plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)
```

## Conclusion
This implementation provides a foundation for training and evaluating a GPT model using PyTorch. You can modify it to train on your own dataset, tweak the model architecture, or experiment with different tokenization methods.
```
