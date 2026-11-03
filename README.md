# BiCrossMFB-Kla

Benchmark data and evaluation resources for lysine lactylation site prediction.

- `data/train_*.fasta` and `data/test_*.fasta`: benchmark sequences from [DeepKla](https://github.com/linDing-group/DeepKla).
- `data/mouse_test_positive.fasta` and `data/mouse_test_negative.fasta`: mouse test sequences with 355 positive and 355 negative samples, respectively. Mouse positive-site annotations originate from UniProt and [Wu et al. (2025)](https://doi.org/10.3389/fcvm.2025.1644886).
- `predictions.csv`: saved DeepKla test labels and prediction probabilities.
- `reproduce.py`: a script to recalculate evaluation metrics.

This release supports metric recalculation from the saved DeepKla predictions.
