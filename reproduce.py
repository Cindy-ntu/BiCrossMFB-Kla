#!/usr/bin/env python3
"""Recalculate metrics from saved predictions; no model or dependencies needed."""

import csv
import json
import math
from itertools import groupby
from pathlib import Path


def calculate_metrics(labels, probabilities):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError('Both positive and negative samples are required.')

    predicted = [int(probability > 0.5) for probability in probabilities]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predicted))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predicted))
    fp, fn = negatives - tn, positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Group equal scores before integrating ROC and precision-recall curves.
    # The PR curve starts at recall=0, precision=1, matching sklearn's endpoint.
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0], reverse=True)
    cumulative_tp = cumulative_fp = 0
    previous_recall = previous_fpr = 0.0
    previous_precision = 1.0
    auc_roc = auc_prc = 0.0
    for _, group in groupby(ranked, key=lambda item: item[0]):
        for _, label in group:
            cumulative_tp += label
            cumulative_fp += 1 - label
        current_recall = cumulative_tp / positives
        current_fpr = cumulative_fp / negatives
        current_precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        auc_roc += (current_fpr - previous_fpr) * (current_recall + previous_recall) / 2
        auc_prc += (current_recall - previous_recall) * (current_precision + previous_precision) / 2
        previous_recall, previous_fpr = current_recall, current_fpr
        previous_precision = current_precision

    return {
        'accuracy': (tp + tn) / len(labels),
        'precision': precision,
        'recall': recall,
        'specificity': tn / negatives,
        'f1': 2 * tp / (2 * tp + fp + fn),
        'mcc': (tp * tn - fp * fn) / denominator if denominator else 0.0,
        'auc_roc': auc_roc,
        'auc_prc': auc_prc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def main():
    path = Path(__file__).resolve().parent / 'predictions.csv'
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('predictions.csv is empty.')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Sample IDs must be unique.')
    labels = [int(row['label']) for row in rows]
    probabilities = [float(row['probability']) for row in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError('Labels must be 0 or 1.')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError('Probabilities must be finite values between 0 and 1.')
    print(json.dumps({'samples': len(rows), **calculate_metrics(labels, probabilities)}, indent=2))


if __name__ == '__main__':
    main()
