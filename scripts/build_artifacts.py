"""Build the model + preprocessor artifacts for the inference API.

Runs the same training procedure as Week 2 (three candidate classifiers,
GridSearchCV 5-fold, F1 scoring) and writes the winning end-to-end pipeline
to ``artifacts/model.joblib`` plus the fitted preprocessing pipeline to
``artifacts/preprocessor.joblib``.

The pickled pipeline references the ``preprocessing`` module (``src/`` is
added to ``sys.path``), so ``app/main.py`` must import ``preprocessing`` the
same way before unpickling.

Usage:
    python scripts/build_artifacts.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import TARGET_COL, build_preprocessor, load_raw  # noqa: E402

SEED = 42

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            "clf__max_iter": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [None, 5, 10],
        },
    ),
}


def main() -> None:
    df = load_raw(ROOT / "data" / "raw" / "titanic.csv")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET_COL]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df[TARGET_COL]
    )
    X_train = train_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    y_train = train_df[TARGET_COL].astype(int).reset_index(drop=True)
    X_test = test_df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    y_test = test_df[TARGET_COL].astype(int).reset_index(drop=True)

    best_name, best_f1, best_pipe = None, -1.0, None
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        f1 = float(f1_score(y_test, search.predict(X_test)))
        print(f"{name}: cv_f1={search.best_score_:.4f} test_f1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_f1, best_pipe = name, f1, search.best_estimator_
    print(f"Winner: {best_name} (test F1={best_f1:.4f})")

    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, art / "model.joblib")
    joblib.dump(build_preprocessor().fit(X_train), art / "preprocessor.joblib")
    (art / "model_card.json").write_text(
        json.dumps({"model": best_name, "test_f1": best_f1,
                    "seed": SEED, "target": TARGET_COL}, indent=2)
    )
    print(f"Wrote {art / 'model.joblib'} and {art / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
