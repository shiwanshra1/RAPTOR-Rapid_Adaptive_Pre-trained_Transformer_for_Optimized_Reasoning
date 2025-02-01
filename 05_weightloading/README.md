
# Loading Pretrained Weights into GPTModel

## Overview
In this segment, we focus on transferring the pretrained weights for a 124M parameter GPT-2 model into our `GPTModel` instance.

## Steps

1. **Initialize GPTModel**  
   - First, create a new instance of the `GPTModel` class.

2. **Adjust for Bias Vectors**  
   - The original GPT-2 model initializes its attention layers with bias vectors. To ensure compatibility, set `qkv_bias=True` in the `GPTModel` implementation, as this is necessary for loading the pretrained weights correctly.

3. **Set Context Length**  
   - The original GPT-2 model uses a context length of 1024 tokens, which should be replicated in our model for accurate weight transfer.

4. **Load Weights**  
   - Transfer the downloaded GPT-2 model weights into the `GPTModel` instance.

## Notes
- The `qkv_bias=True` setting allows for the correct loading of the weights, even though biases are not necessary for the model's functionality.  
- Using the same context length (1024 tokens) ensures the model operates in line with the original GPT-2 architecture.

