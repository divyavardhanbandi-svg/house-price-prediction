"""
House Price Prediction
----------------------
End-to-end regression project:
  1. Load data (California Housing; falls back to synthetic data if offline)
  2. Explore and clean
  3. Build preprocessing + model pipelines
  4. Compare models with cross-validation
  5. Tune the best model
  6. Evaluate on a held-out test set
  7. Save the model and make a sample prediction

Run:  python house_price_prediction.py
Install:  pip install pandas numpy scikit-learn matplotlib joblib
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TARGET = "MedHouseVal"  # median house value in units of $100,000


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    try:
        df = fetch_california_housing(as_frame=True).frame
        print("Loaded California Housing dataset:", df.shape)
    except Exception as exc:  # offline fallback
        print(f"Could not download dataset ({type(exc).__name__}); using synthetic data.")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 5000
        df = pd.DataFrame(
            {
                "MedInc": rng.gamma(4, 1, n),
                "HouseAge": rng.integers(1, 52, n),
                "AveRooms": rng.normal(5.5, 1.5, n).clip(1, None),
                "AveBedrms": rng.normal(1.1, 0.2, n).clip(0.5, None),
                "Population": rng.integers(100, 5000, n),
                "AveOccup": rng.normal(3, 0.8, n).clip(1, None),
                "Latitude": rng.uniform(32.5, 42, n),
                "Longitude": rng.uniform(-124, -114, n),
            }
        )
        df[TARGET] = (
            0.45 * df["MedInc"]
            + 0.1 * df["AveRooms"]
            - 0.005 * df["HouseAge"]
            + rng.normal(0, 0.3, n)
        ).clip(0.15, 5.0)
    return df


# --------------------------------------------------------------------------
# 2. Explore and clean
# --------------------------------------------------------------------------
def explore(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Data overview ---")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # Cap the target at the 99th percentile to reduce the effect of outliers
    cap = df[TARGET].quantile(0.99)
    df = df[df[TARGET] <= cap].copy()

    # Feature engineering
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[TARGET].hist(bins=40, ax=axes[0])
    axes[0].set_title("House value distribution")
    corr = df.corr()[TARGET].drop(TARGET).sort_values()
    corr.plot.barh(ax=axes[1])
    axes[1].set_title("Correlation with house value")
    plt.tight_layout()
    plt.savefig("eda.png", dpi=120)
    plt.close()
    return df


# --------------------------------------------------------------------------
# 3-4. Models and cross-validation
# --------------------------------------------------------------------------
def build_models() -> dict:
    def pipe(model):
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "LinearRegression": pipe(LinearRegression()),
        "Ridge": pipe(Ridge(alpha=1.0)),
        "RandomForest": pipe(
            RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
        ),
        "GradientBoosting": pipe(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


def compare_models(models, X_train, y_train) -> str:
    print("\n--- 5-fold cross-validation (R²) ---")
    scores = {}
    for name, model in models.items():
        r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        scores[name] = r2.mean()
        print(f"{name:18s} R² = {r2.mean():.4f} (+/- {r2.std():.4f})")
    best = max(scores, key=scores.get)
    print(f"Best model: {best}")
    return best


# --------------------------------------------------------------------------
# 5. Hyperparameter tuning
# --------------------------------------------------------------------------
def tune(best_name, models, X_train, y_train):
    grids = {
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 3],
        },
        "GradientBoosting": {
            "model__n_estimators": [150, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4],
        },
        "Ridge": {"model__alpha": [0.1, 1.0, 10.0]},
        "LinearRegression": {},
    }
    grid = grids[best_name]
    if not grid:
        return models[best_name].fit(X_train, y_train)

    print(f"\nTuning {best_name} ...")
    search = GridSearchCV(models[best_name], grid, cv=3, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_


# --------------------------------------------------------------------------
# 6. Evaluate
# --------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print("\n--- Test set performance ---")
    print(f"MAE  : {mae:.4f}  (~${mae * 100_000:,.0f})")
    print(f"RMSE : {rmse:.4f}  (~${rmse * 100_000:,.0f})")
    print(f"R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_test, pred, alpha=0.3, s=8)
    lims = [y_test.min(), y_test.max()]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs predicted")
    axes[1].hist(y_test - pred, bins=40)
    axes[1].set_title("Residuals")
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=120)
    plt.close()

    # Feature importance for tree models
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=X_test.columns).sort_values()
        imp.plot.barh(figsize=(7, 5), title="Feature importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=120)
        plt.close()


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------
def main():
    df = explore(load_data())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = build_models()
    best_name = compare_models(models, X_train, y_train)
    final_model = tune(best_name, models, X_train, y_train)
    evaluate(final_model, X_test, y_test)

    joblib.dump(final_model, "house_price_model.joblib")
    print("\nSaved model -> house_price_model.joblib")

    # Sample prediction using one row from the test set
    sample = X_test.iloc[[0]]
    price = final_model.predict(sample)[0] * 100_000
    print(f"Sample prediction: ${price:,.0f} (actual: ${y_test.iloc[0] * 100_000:,.0f})")


if __name__ == "__main__":
    main()
