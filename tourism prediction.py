import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://github.com/ngawang-zangphel/tourism/blob/main/bhutan-tourism-data.xlsx?raw=true"

tourism_data = pd.read_excel(url, sheet_name=0)

#3. Data Cleaning
# Drop the first two rows to remove metadata
df_cleaned = tourism_data.iloc[2:].reset_index(drop=True)

# Rename columns using the second row as headers
df_cleaned.columns = ["metric", "category", "unit"] + list(df_cleaned.iloc[0, 3:].values)

# Drop the duplicate header row
df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)

# Replace placeholders like ".." with NaN
df_cleaned.replace("..", pd.NA, inplace=True)

# Convert year columns to numeric values
year_columns = df_cleaned.columns[3:]
df_cleaned[year_columns] = df_cleaned[year_columns].apply(pd.to_numeric, errors="coerce")

# Correct year column names
df_cleaned.columns = ["metric", "category", "unit"] + list(range(1995, 2023))

#4. Data Visualization
df = df_cleaned

df_filtered = df[df['category'] == 'inbound_arrivals_by_region'].set_index('metric')

# Select years as columns (from 1995 to 2022)
years = df_filtered.columns[2:]

# Transpose for plotting
df_plot = df_filtered[years].T

# Plotting the data
plt.figure(figsize=(12, 6))
for region in df_plot.columns:
    plt.plot(df_plot.index, df_plot[region], marker='o', label=region)

# Formatting the plot
plt.xlabel("Year")
plt.ylabel("Inbound Arrivals (Thousands)")
plt.title("Inbound Tourism Arrivals by Region (1995-2022)")
df_plot.index = pd.to_numeric(df_plot.index, errors='coerce')  # Converts non-numeric values to NaN
plt.xticks(np.arange(df_plot.index.min(), df_plot.index.max() + 1, 1), rotation=45)
plt.legend(title="Region", loc='upper left')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#2. Bar Graph
num_regions = len(df_plot.columns)
fig, axes = plt.subplots(nrows=num_regions, ncols=1, figsize=(8, 4 * num_regions))


for i, region in enumerate(df_plot.columns):
    axes[i].bar(df_plot.index, df_plot[region])
    axes[i].set_xlabel("Year")
    axes[i].set_ylabel("Inbound Arrivals (Thousands)")
    axes[i].set_title(f"Inbound Arrivals for {region} (1995-2022)")


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#5. Data Prediction
tourism_data = df

# Remove NaN metrics
tourism_data = tourism_data.dropna(subset=["metric"])
df = tourism_data[tourism_data["category"] == "inbound_arrivals_by_region"]

# Get unique metrics
metrics = df["metric"].dropna().unique()

# Store predictions in a dictionary
predictions_dict = {}

# Iterate through each metric
for metric in metrics:
    df_metric = df[df["metric"] == metric].drop(columns=["category", "unit"], errors="ignore")

    # Drop non-numeric columns and reshape data
    df_metric = df_metric.drop(columns=["metric"], errors="ignore").melt(var_name="Year", value_name="Value").dropna()

    # Ensure Year is an integer
    df_metric["Year"] = pd.to_numeric(df_metric["Year"], errors="coerce").astype("Int64")

    # Drop NaN values and sort
    df_metric = df_metric.dropna().sort_values("Year")

    # Check if data exists
    if df_metric.empty or len(df_metric) < 2:
        print(f"⚠️ Skipping metric '{metric}' due to insufficient data.")
        continue

    # Extract features (X) and target (y)
    X = df_metric["Year"].values.reshape(-1, 1)
    y = df_metric["Value"].values

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # 🔥 Predict for years 2022 to  only
    future_years = np.array(range(2022, datetime.now().year + 5)).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Store predictions
    predictions_dict[metric] = {
        "Year": future_years.flatten(),
        "Predicted Value": future_predictions
    }

# Convert predictions to DataFrame in original format
predicted_df = pd.DataFrame()

for metric, data in predictions_dict.items():
    # Get category and unit from the original dataset
    meta_info = df[df["metric"] == metric][["metric", "category", "unit"]].drop_duplicates()

    # Create DataFrame with years as columns
    pred_data = pd.DataFrame([data["Predicted Value"]], columns=map(str, data["Year"]))
    pred_data.insert(0, "metric", metric)  # Add metric column
    pred_data.insert(1, "category", meta_info["category"].values[0] if not meta_info.empty else "Unknown")
    pred_data.insert(2, "unit", meta_info["unit"].values[0] if not meta_info.empty else "Unknown")

    # Append to final DataFrame
    predicted_df = pd.concat([predicted_df, pred_data], ignore_index=True)

# ✅ Ensure year columns are sorted numerically
year_columns = sorted([col for col in predicted_df.columns if col.isdigit()], key=int)
fixed_columns = ["metric", "category", "unit"] + year_columns
predicted_df = predicted_df[fixed_columns]

# Export to Excel
output_filename = input("Enter your file name: ")
predicted_df.to_excel(f"{output_filename}.xlsx", index=False)
print(f"Data exported as {output_filename}")

# 🎯 Plot the predictions
plt.figure(figsize=(10, 5))

for metric, data in predictions_dict.items():
    plt.plot(data["Year"], data["Predicted Value"], marker="o", linestyle="-", label=metric)

plt.xlabel("Year")
plt.ylabel("Predicted Value")
plt.title("Predicted Metrics (2022-2032)")
plt.legend(loc="upper left")
plt.grid(True)
plt.xticks(range(2022, datetime.now().year + 5))
plt.show()

#6. Evaluation
evaluation_results = {}
for metric in metrics:
    X = df_metric["Year"].values.reshape(-1, 1)
    y = df_metric["Value"].values

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # 🎯 Evaluate Model Performance
    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    # Store evaluation results
    evaluation_results[metric] = {"MAE": mae, "MSE": mse, "R² Score": r2}

evaluation_data = pd.DataFrame.from_dict(evaluation_results, orient='index')
print(evaluation_data)


