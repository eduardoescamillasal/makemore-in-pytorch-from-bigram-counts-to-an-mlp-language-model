"""
makemore in PyTorch: From Bigram Counts to an MLP Language Model scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""makemore in PyTorch: from bigram counts to an MLP language model.

Story: count bigrams over 32,000 names and sample from the table; learn the same table with
a one-layer network; then train an MLP that sees three characters, check it on names it never
trained on, and sample new names from it.
"""
import torch


def main():
    words = load_names()
    stoi, itos = build_vocab(words)

    # ---- 1. Counting ----
    N = bigram_counts(words, stoi)
    P = bigram_probs(N, smoothing=1)
    print("bigram loss on all names:", round(bigram_nll(P, words, stoi), 4))
    print("bigram samples:", sample_bigram(P, itos, n=8, seed=2147483647))

    # ---- 2. The same model as a network ----
    xs, ys = bigram_dataset(words, stoi)
    W, losses = train_neural_bigram(xs, ys, len(stoi), steps=100, lr=50.0, reg=0.01, seed=2147483647)
    print("neural bigram loss:", round(losses[0], 4), "->", round(losses[-1], 4))

    # ---- 3. An MLP over three characters ----
    train, dev, test = split_words(words, seed=42)
    Xtr, Ytr = build_context_dataset(train, stoi, block_size=3)
    Xdev, Ydev = build_context_dataset(dev, stoi, block_size=3)
    Xte, Yte = build_context_dataset(test, stoi, block_size=3)
    model = MLP(len(stoi), block_size=3, emb_dim=10, hidden=200, seed=2147483647)
    print("MLP parameters:", sum(p.nelement() for p in model.parameters()))
    train_mlp(model, Xtr, Ytr, steps=30000, batch_size=64, lr=0.1, seed=0)
    print("MLP train loss:", round(evaluate_loss(model, Xtr, Ytr), 4))
    print("MLP dev loss:  ", round(evaluate_loss(model, Xdev, Ydev), 4))
    print("MLP test loss: ", round(evaluate_loss(model, Xte, Yte), 4), "(checked once, at the end)")
    print("MLP samples:", sample_mlp(model, itos, n=8, seed=2147483647))


if __name__ == "__main__":
    main()

