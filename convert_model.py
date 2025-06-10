import tensorflow as tf
import numpy as np
import json
import os

os.makedirs('model', exist_ok=True)

YOUR_MODEL_NAME = 'fashion_mnist'
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'model/{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'model/{YOUR_MODEL_NAME}.json'

# === Step 1: Load Keras model (.h5) ===
model = tf.keras.models.load_model(TF_MODEL_PATH)

# === Step 2: Extract and save weights to .npz ===
params = {}
print("🔍 Extracting weights from model...\n")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            print(f"  {param_name}: shape={w.shape}")
            params[param_name] = w
        print()

np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ Saved all weights to {MODEL_WEIGHTS_PATH}")

# === Step 3: Save architecture to .json ===
arch = []
for layer in model.layers:
    config = layer.get_config()
    info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": config,
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    }
    arch.append(info)

with open(MODEL_ARCH_PATH, "w") as f:
    json.dump(arch, f, indent=2)

print(f"✅ Architecture saved to {MODEL_ARCH_PATH}")

# === Step 4: Load back and run forward pass test ===
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# === Flatten and Dense ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

# === Forward function ===
def forward(x):
    for layer in architecture:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

# === Test with dummy input ===
dummy_input = np.random.rand(1, 28 * 28).astype(np.float32)
output = forward(dummy_input)
print("🧠 Output probabilities:", output)
print("✅ Predicted class:", np.argmax(output, axis=-1))
