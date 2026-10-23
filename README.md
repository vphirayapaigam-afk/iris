# Data Classification Using AI (Iris + KNN)

A supervised-learning project built for DecodeLabs AI Project 2. It trains a
K-Nearest Neighbors classifier to identify iris species from flower measurements.

## Pipeline (IPO)
| Stage | What happens |
|-------|--------------|
| **Input** | Load the Iris dataset (150 samples, 4 features, 3 balanced classes), explore it, check for missing values |
| **Process** | Shuffled, stratified 80/20 train-test split, `StandardScaler` fitted on training data only, K tuned with 5-fold cross-validation, KNN trained |
| **Output** | Accuracy, confusion matrix, precision / recall / macro F1 score, saved plots |

## Setup
```bash
pip install -r requirements.txt
```
Requires Python 3.9+.

## Run
```bash
python classifier.py            # auto-tunes K
python classifier.py --k 7      # use a fixed K
```
Plots are saved in the `outputs/` folder (`confusion_matrix.png`, `k_selection.png`).

## Test
```bash
python -m unittest -v
```

## Sample result (test set of 30 unseen flowers)
- Best K: 5
- Accuracy: 0.933
- Macro F1: 0.933
- Setosa is classified perfectly; the only mistakes are 2 virginica flowers predicted as versicolor, the two species that overlap most.

## Design notes
- **Scaling after splitting:** the scaler learns mean and standard deviation from the training set only, so no test information leaks into training.
- **K selection on training data:** cross-validation picks K without touching the test set, which stays locked for the final check.
- **Beyond accuracy:** the confusion matrix and F1 score show *where* the model errs, not just how often.

## Ideas to extend
- Compare KNN with Decision Tree, SVM, or Logistic Regression
- Test with your own flower measurements via `predict_flower()`
- Try a different dataset such as Wine or Breast Cancer (both in scikit-learn)
