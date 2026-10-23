"""Data Classification Using AI - DecodeLabs AI Project 2.

Pipeline (IPO model):
    Input   -> load the Iris dataset, explore it, scale the features
    Process -> shuffled 80/20 train-test split, tune K, train KNN
    Output  -> accuracy, confusion matrix, precision / recall / F1 score
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% train / 20% test


# ---------------------------------------------------------------- INPUT
def load_data():
    """Load Iris and return (DataFrame, X, y, class_names)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, iris.data.to_numpy(), iris.target.to_numpy(), list(iris.target_names)


def explore(df: pd.DataFrame) -> None:
    """Print a quick overview so we understand the dataset."""
    print("=== DATASET OVERVIEW ===")
    print(f"Samples: {len(df)} | Features: {df.shape[1] - 2} | Classes: {df['species'].nunique()}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nClass balance:")
    print(df["species"].value_counts().to_string())
    print("\nFeature statistics:")
    print(df.drop(columns=["target", "species"]).describe().round(2).to_string())
    print(f"\nMissing values: {int(df.isna().sum().sum())}")


# -------------------------------------------------------------- PROCESS
def split_and_scale(X, y):
    """Shuffle + stratified split, then scale using TRAIN statistics only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # learn mean/std from train only
    X_test = scaler.transform(X_test)  # apply the same transform (no data leakage)
    return X_train, X_test, y_train, y_test, scaler


def find_best_k(X_train, y_train, k_values=range(1, 26)):
    """Pick K by 5-fold cross-validation on the TRAINING set (test set stays locked)."""
    from sklearn.model_selection import cross_val_score

    errors = []
    for k in k_values:
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
        errors.append(1 - scores.mean())
    best_k = list(k_values)[int(np.argmin(errors))]
    return best_k, list(k_values), errors


def train_model(X_train, y_train, k: int) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=k)  # INSTANTIATE
    model.fit(X_train, y_train)  # FIT
    return model


# --------------------------------------------------------------- OUTPUT
def evaluate(model, X_test, y_test, class_names, out_dir: Path) -> dict:
    predictions = model.predict(X_test)  # PREDICT
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("\n=== EVALUATION (unseen test set) ===")
    print(f"Accuracy      : {acc:.3f}")
    print(f"Macro F1 score: {f1:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=class_names, digits=3))

    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion Matrix - KNN on Iris")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}


def plot_k_curve(k_values, errors, best_k, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, errors, marker="o")
    plt.axvline(best_k, color="orange", linestyle="--", label=f"best K = {best_k}")
    plt.xlabel("K value")
    plt.ylabel("Cross-validation error rate")
    plt.title("Choosing K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "k_selection.png", dpi=150)
    plt.close()


def predict_flower(model, scaler, class_names, measurements) -> str:
    """Classify one new flower: [sepal_len, sepal_wid, petal_len, petal_wid] in cm."""
    scaled = scaler.transform(np.array(measurements, dtype=float).reshape(1, -1))
    return class_names[int(model.predict(scaled)[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Iris classification with KNN")
    parser.add_argument("--k", type=int, default=None, help="use a fixed K (default: auto-tune)")
    parser.add_argument("--out", default="outputs", help="folder for saved plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y, class_names = load_data()
    explore(df)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    if args.k is None:
        best_k, k_values, errors = find_best_k(X_train, y_train)
        plot_k_curve(k_values, errors, best_k, out_dir)
        print(f"Best K (5-fold CV): {best_k}")
    else:
        best_k = args.k
        print(f"Using K = {best_k}")

    model = train_model(X_train, y_train, best_k)
    evaluate(model, X_test, y_test, class_names, out_dir)

    sample = [5.9, 3.0, 5.1, 1.8]
    print(f"\nNew flower {sample} -> {predict_flower(model, scaler, class_names, sample)}")
    print(f"Plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
