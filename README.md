# makemore in PyTorch: From Bigram Counts to an MLP Language Model

Build Andrej Karpathy's makemore from scratch in PyTorch, the way the lectures do. Count which letter follows which across 32,000 names, sample new names and score the model with the negative log-likelihood. Learn the same bigram table with a one-layer neural network. Then train the MLP from Bengio et al. (2003): an embedding table, a tanh hidden layer and minibatch training on three characters of context, checked on a dev split it never trained on, and sample names that sound far more like names.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_names
- [x] **2.** build_vocab
- [x] **3.** bigram_counts
- [x] **4.** bigram_probs
- [x] **5.** sample_bigram
- [x] **6.** bigram_nll
- [x] **7.** bigram_dataset
- [x] **8.** train_neural_bigram
- [ ] **9.** build_context_dataset
- [ ] **10.** split_words
- [ ] **11.** MLP
- [ ] **12.** evaluate_loss
- [ ] **13.** train_mlp
- [ ] **14.** sample_mlp

---

Built on Deep-ML.
