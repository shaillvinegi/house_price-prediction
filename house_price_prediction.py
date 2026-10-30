import os
import re
import html
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "housing.csv"
OUTPUT_DIR = "outputs"
TARGET = "median_house_value"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def clean_column_name(name):
    """Clean unwanted characters from column names."""
    name = html.unescape(str(name)).strip()
    name = re.sub(r"^\*+", "", name)
    name = re.sub(r"^&#x[0-9a-fA-F]+;", "", name)
    return name.strip()


def save_plot(filename):
    """Save a graph inside the outputs folder."""
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 2. LOAD AND INSPECT DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find '{DATA_FILE}'. Put housing.csv in the same "
        "folder as house_price_prediction.py."
    )

df = pd.read_csv(DATA_FILE)

df.columns = [clean_column_name(col) for col in df.columns]

# Remove unnecessary spreadsheet index columns.
drop_cols = [
    col for col in df.columns
    if col.lower().startswith("unnamed")
    or col.lower() in {"index", "row_number"}
]

if drop_cols:
    df.drop(columns=drop_cols, inplace=True)

# Remove duplicate column names.
df = df.loc[:, ~pd.Index(df.columns).duplicated()]

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns: {list(df.columns)}"
    )

print("\n" + "=" * 60)
print("HOUSE PRICE PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\nFirst five rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print(f"\nDuplicate rows: {df.duplicated().sum():,}")


# ============================================================
# 3. MISSING-VALUE ANALYSIS
# ============================================================

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": df.isnull().mean() * 100
})

print("\nMissing-value summary:")
print(missing_summary.to_string())

missing_summary.to_csv(
    os.path.join(OUTPUT_DIR, "eda_missing_values.csv"),
    index_label="Column"
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove duplicate records.
df = df.drop_duplicates().copy()

# Convert columns to numeric when most values are numeric.
# Categorical columns such as ocean_proximity remain as text.
for col in df.columns:
    if df[col].dtype == "object":
        converted = pd.to_numeric(df[col], errors="coerce")
        non_empty = df[col].notna()

        if (
            non_empty.any()
            and converted[non_empty].notna().mean() >= 0.90
        ):
            df[col] = converted

# Ensure the target column contains valid non-negative prices.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

df = df[
    df[TARGET].notna() & (df[TARGET] >= 0)
].copy()

if df.empty:
    raise ValueError(
        "No usable rows remain after cleaning the target column."
    )

print(f"\nRows after cleaning: {len(df):,}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_data = df[numeric_cols].copy()


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

q1 = numeric_data.quantile(0.25)
median = numeric_data.median()
q3 = numeric_data.quantile(0.75)
iqr = q3 - q1

mode_values = numeric_data.mode(dropna=True)

if not mode_values.empty:
    mode = mode_values.iloc[0]
else:
    mode = pd.Series(index=numeric_cols, dtype=float)

stats = pd.DataFrame({
    "Count": numeric_data.count(),
    "Mean": numeric_data.mean(),
    "Median": median,
    "Mode": mode,
    "Variance": numeric_data.var(),
    "Standard Deviation": numeric_data.std(),
    "Minimum": numeric_data.min(),
    "Q1": q1,
    "Q2 (Median)": median,
    "Q3": q3,
    "Maximum": numeric_data.max(),
    "Range": numeric_data.max() - numeric_data.min(),
    "IQR": iqr
})

print(stats.round(3).to_string())

stats.to_csv(
    os.path.join(OUTPUT_DIR, "eda_descriptive_statistics.csv"),
    index_label="Feature"
)


# ============================================================
# 6. OUTLIER ANALYSIS USING IQR
# ============================================================

# Detect and report outliers without automatically deleting them.
outlier_rows = []

for col in numeric_cols:

    values = numeric_data[col].dropna()

    if values.empty:
        continue

    first_quartile = values.quantile(0.25)
    third_quartile = values.quantile(0.75)

    column_iqr = third_quartile - first_quartile

    lower_limit = first_quartile - 1.5 * column_iqr
    upper_limit = third_quartile + 1.5 * column_iqr

    outlier_count = int(
        ((values < lower_limit) | (values > upper_limit)).sum()
    )

    outlier_rows.append({
        "Feature": col,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": outlier_count,
        "Outlier Percentage": (
            outlier_count / len(values)
        ) * 100
    })

outlier_df = pd.DataFrame(outlier_rows)

print("\nPotential outliers detected using the IQR method:")
print(outlier_df.round(3).to_string(index=False))

outlier_df.to_csv(
    os.path.join(OUTPUT_DIR, "eda_outlier_analysis.csv"),
    index=False
)


# ============================================================
# 7. GRAPH 1 - HOUSE PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x=TARGET,
    bins=30,
    kde=True
)

plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value (USD)")
plt.ylabel("Number of Records")

save_plot("01_house_price_distribution.png")


# ============================================================
# 8. GRAPH 2 - MEDIAN INCOME VS HOUSE PRICE
# ============================================================

income_col = next(
    (
        col for col in df.columns
        if col.lower() == "median_income"
    ),
    None
)

if income_col is not None:

    plot_data = df[[income_col, TARGET]].copy()

    plot_data[income_col] = pd.to_numeric(
        plot_data[income_col],
        errors="coerce"
    )

    plot_data = plot_data.dropna()

    # Sample large datasets to make the graph easier to read.
    if len(plot_data) > 8000:
        plot_data = plot_data.sample(
            n=8000,
            random_state=42
        )

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=plot_data,
        x=income_col,
        y=TARGET,
        alpha=0.35,
        s=18
    )

    plt.title("Median Income vs. Median House Value")
    plt.xlabel("Median Income (dataset units)")
    plt.ylabel("Median House Value (USD)")

    save_plot("02_income_vs_house_price.png")

else:
    print(
        "\nSkipping income graph: "
        "median_income column was not found."
    )


# ============================================================
# 9. GRAPH 3 - CORRELATION HEATMAP
# ============================================================

if len(numeric_cols) >= 2:

    # Show the target first and limit the graph to 10 columns.
    selected_cols = [TARGET] + [
        col for col in numeric_cols
        if col != TARGET
    ]

    selected_cols = selected_cols[:10]

    correlation_matrix = numeric_data[selected_cols].corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 8}
    )

    plt.title("Correlation Between Numerical Features")

    save_plot("03_correlation_heatmap.png")


# ============================================================
# 10. TRAIN THE LINEAR REGRESSION MODEL
# ============================================================

print("\n" + "=" * 60)
print("HOUSE PRICE PREDICTION MODEL")
print("=" * 60)

# Separate input features and the target.
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify numerical and categorical input columns.
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# Handle missing numerical values and scale numerical features.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Handle missing categorical values and encode categories.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)

# Create the complete machine-learning pipeline.
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split the dataset: 80% training and 20% testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train the model and generate predictions.
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")

print(f"MAE:       {mae:,.2f}")
print(f"MSE:       {mse:,.2f}")
print(f"RMSE:      {rmse:,.2f}")
print(f"R-squared: {r2:.4f}")

# Save model evaluation metrics.
metrics = pd.DataFrame([{
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}])

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "model_evaluation.csv"),
    index=False
)

# Save actual and predicted house prices.
prediction_results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_pred
})

prediction_results.to_csv(
    os.path.join(OUTPUT_DIR, "prediction_results.csv"),
    index=False
)


# ============================================================
# 12. GRAPH 4 - ACTUAL VS PREDICTED HOUSE PRICES
# ============================================================

plot_points = prediction_results.copy()

if len(plot_points) > 5000:
    plot_points = plot_points.sample(
        n=5000,
        random_state=42
    )

plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=plot_points,
    x="actual",
    y="predicted",
    alpha=0.4,
    s=18
)

# The dashed line represents perfect predictions.
min_value = min(
    plot_points["actual"].min(),
    plot_points["predicted"].min()
)

max_value = max(
    plot_points["actual"].max(),
    plot_points["predicted"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs. Predicted House Values")
plt.xlabel("Actual House Value (USD)")
plt.ylabel("Predicted House Value (USD)")

save_plot("04_actual_vs_predicted.png")


# ============================================================
# 13. FINISH
# ============================================================

print("\n" + "=" * 60)
print("PROJECT RUN COMPLETED")
print("=" * 60)

print(f"Results saved in the '{OUTPUT_DIR}' folder.")

print("\nGraphs generated:")
print("1. 01_house_price_distribution.png")
print("2. 02_income_vs_house_price.png")
print("3. 03_correlation_heatmap.png")
print("4. 04_actual_vs_predicted.png")

print("\nCSV files generated:")
print("1. eda_missing_values.csv")
print("2. eda_descriptive_statistics.csv")
print("3. eda_outlier_analysis.csv")
print("4. model_evaluation.csv")
print("5. prediction_results.csv")