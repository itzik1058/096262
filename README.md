# Competition

## Reproduction

To reproduce the results run
`main.py . path_to_queries`

To reproduce and evaluate the results
run `main(evaluate=True)` with arguments
`main.py path_to_qrels path_to_queries`
or alternatively use trec_eval with the .res files created from reproducing the results.

To run cross-validation with [Weights & Biases](https://wandb.ai/) run the `make_sweeps()` function once to make a sweep for every method, then copy the sweep id (from the output or the website) and run `sweep(sweep_id, method)` for every method (can be run simultaneously from many sources).

## Results

Cross-validation best results for every method:

| Method | Dirichlet | Jelinek-Mercer | Two-stage | tfidf | Okapi |
| --- | :---: | :---: | :---: | :---: | :---: |
| Parameters | `μ=990` | `collectionLambda=0.2534` `documentLambda=0.8633` | `μ=325` `lambda=0.4762` | `k1=0.4844` `b=0.4867` | `k1=0.3899` `b=0.76` `k3=9` |
| Relevance Feedback Parameters | `fbDocs=14` `fbTerms=12` `fbMu=71` `fbOrigWeight=0.4308` | `fbDocs=12` `fbTerms=11` `fbMu=32` `fbOrigWeight=0.3892` | `fbDocs=15` `fbTerms=13` `fbMu=103` `fbOrigWeight=0.4352` | `fbDocs=10` `fbTerms=10` `fbMu=77` `fbOrigWeight=0.8694` | `fbDocs=8` `fbTerms=10` `fbMu=63` `fbOrigWeight=0.7273` |
| MAP | 0.2481 | 0.239 | **0.2526** | 0.2521 | 0.2468 |

Cross-validation fusion results:

| Fusion Method | Borda | Copeland | Weighted Copeland |
| --- | :---: | :---: | :---: |
| MAP | 0.2584 | 0.2636 | **0.2646** |

Weights used for weighted copeland are
| Method | Dirichlet | Jelinek-Mercer | Two-stage | tfidf | Okapi |
| --- | :---: | :---: | :---: | :---: | :---: |
| Weight | 0.4201 | 0.2555 | 0.8505 | 0.6863 | 0.6547 |
