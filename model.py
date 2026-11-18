"""
makemore in PyTorch: From Bigram Counts to an MLP Language Model

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_names
import os
import tempfile
import urllib.request

def load_names():
    """Cached download of makemore's names.txt; returns the list of names."""
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    cache_path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")

    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    with open(cache_path, "r", encoding="utf-8") as f:
        words = f.read().lower().splitlines()

    return words

# Step 2 - build_vocab
def build_vocab(words):
    """Return (stoi, itos): '.' -> 0, the sorted characters -> 1, 2, ..."""
    chars = sorted(set("".join(words)))
    stoi = {ch: i for i, ch in enumerate(chars, start=1)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

# Step 3 - bigram_counts
import torch

def bigram_counts(words, stoi):
    """(V, V) int32 tensor of bigram counts, with '.' around every word."""
    V = len(stoi)
    N = torch.zeros((V,V), dtype=torch.int32)

    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    return N

# Step 4 - bigram_probs (not yet solved)
# TODO: implement

# Step 5 - sample_bigram (not yet solved)
# TODO: implement

# Step 6 - bigram_nll (not yet solved)
# TODO: implement

# Step 7 - bigram_dataset (not yet solved)
# TODO: implement

# Step 8 - train_neural_bigram (not yet solved)
# TODO: implement

# Step 9 - build_context_dataset (not yet solved)
# TODO: implement

# Step 10 - split_words (not yet solved)
# TODO: implement

# Step 11 - MLP (not yet solved)
# TODO: implement

# Step 12 - evaluate_loss (not yet solved)
# TODO: implement

# Step 13 - train_mlp (not yet solved)
# TODO: implement

# Step 14 - sample_mlp (not yet solved)
# TODO: implement

