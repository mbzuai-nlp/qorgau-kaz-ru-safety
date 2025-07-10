# Notebooks for Qorgau Kaz-Ru Safety Evaluation

This directory contains Jupyter notebooks and supporting scripts used in the evaluation pipeline for the Qorgau Kaz-Ru Safety benchmark. 

## File Overview

| Filename                         | Description                                                        |
|----------------------------------|--------------------------------------------------------------------|
| `01_prepare_data_for_evaluation.ipynb` | Prepares raw model outputs for downstream evaluation tasks.         |
| `02_binary_evaluation.ipynb`     | Reproduces binary safety classification results (safe vs. unsafe) from the paper. |
| `03_risk_area.ipynb`             | Reproduces risk area analysis.     |
| `04_question_type.ipynb`         | Reproduces analysis of question types. |
| `05_fine_grained.ipynb`          | Reproduces fine-grained safety evaluation results.     |
| `06_human_eval.ipynb`            | Reproduces human - gpt-4o agreement. |
| `evaluate_binary_safety.py`      | Contains reusable helper functions used in safety evaluation.       |

## Usage

To reproduce the evaluation results reported in the paper, run notebooks `02_binary_evaluation.ipynb` through `06_human_eval.ipynb` in sequence. Notebook `01_prepare_data_for_evaluation.ipynb` is used to format the model outputs before evaluation. The helper script `evaluate_binary_safety.py` provides shared functions for binary safety metrics.
