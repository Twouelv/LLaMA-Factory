import numpy as np
from transformers import PreTrainedTokenizer
from typing import Dict

def compute_poker_metrics(decoded_preds: list[str], decoded_labels: list[str]) -> Dict[str, float]:
    
    preds_cleaned = [pred.split("assistant\n\n")[-1].strip() for pred in decoded_preds]
    labels_cleaned = [label.split("assistant\n\n")[-1].strip() for label in decoded_labels]

    correct_predictions = 0
    exact_match = 0

    if not labels_cleaned:
        return {"poker_custom_accuracy": 0.0, "poker_exact_match": 0.0}

    for pred, label in zip(preds_cleaned, labels_cleaned):
        if not label: continue
        if pred == label:
            exact_match += 1
        
        action_pred = pred[0] if pred else ''
        action_label = label[0] if label else ''

        if action_pred != action_label:
            continue

        if action_pred in ['b', 'r', 'a']:
            try:
                val_pred = float(pred[1:])
                val_label = float(label[1:])
                if abs(val_pred - val_label) > 0.1:
                    continue
            except (ValueError, IndexError):
                continue
        
        correct_predictions += 1
        
    return {
        "poker_custom_accuracy": correct_predictions / len(labels_cleaned),
        "poker_exact_match": exact_match / len(labels_cleaned)
    }