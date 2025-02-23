import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
import io

# Title
st.title("Tourism Data Analysis and Prediction Till Year 2032")

data_options = {
    "Use Default Data": "default_data",
    "Upload Your Own": "upload_data",
    "Insert Url": "insert_url",
}

data_option = st.radio("Choose Data Source:", list(data_options.keys()), key="data_source_option")
selected_key = data_options[data_option]
default_data_url = "https://github.com/ngawang-zangphel/tourism/blob/main/assets/data/bhutan-tourism-data.xlsx"

# File Upload
if selected_key == "upload_data":
    # Default Data
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
elif selected_key == "default_data":
    uploaded_file = f"{default_data_url}?raw=true"
    st.text(f"The Default Data will be fetched from")
    st.markdown(default_data_url)
elif selected_key == "insert_url":
    uploaded_file = st.text_input("Enter URL to fetch data from:")

if selected_key == "upload_data" or selected_key == "insert_url":
    st.text(f"Your Table Format should be like below")
    st.image("https://github.com/ngawang-zangphel/tourism/blob/main/assets/images/image.png?raw=true", caption="Tourism Data Overview")

# Check if the url is valid or not
def is_valid_url(url):
    result = urlparse(url)
    return all([result.scheme, result.netloc])

# Predict Tourism Data
def generate_tourism_data():
    tourism_data = pd.read_excel(uploaded_file, sheet_name=0)

    # Data Cleaning
    df_cleaned = tourism_data.iloc[2:].reset_index(drop=True)
    df_cleaned.columns = ["metric", "category", "unit"] + list(df_cleaned.iloc[0, 3:].values)
    df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)
    df_cleaned.replace("..", pd.NA, inplace=True)
    df_cleaned[df_cleaned.columns[3:]] = df_cleaned[df_cleaned.columns[3:]].apply(pd.to_numeric, errors="coerce")
    df_cleaned.columns = ["metric", "category", "unit"] + list(range(1995, 2023))

    # Filter data
    df = df_cleaned[df_cleaned['category'] == 'inbound_arrivals_by_region'].set_index('metric')
    years = df.columns[2:]
    df_plot = df[years].T

    # Visualization
    st.subheader("Inbound Tourism Arrivals by Region (1995-2022)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for region in df_plot.columns:
        ax.plot(df_plot.index, df_plot[region], marker='o', label=region)
    ax.set_xlabel("Year")
    ax.set_ylabel("Inbound Arrivals (Thousands)")
    ax.legend(title="Region", loc='upper left')
    ax.grid(True)

    # Ensure xticks are integers
    ax.set_xticks(np.arange(1995, 2023, 1))  # Set step size to 1 year
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.xticks(rotation=90) 

    st.pyplot(fig)

    # Bar Graph
    st.subheader("Region-wise Inbound Arrivals")
    for region in df_plot.columns:
        fig, ax = plt.subplots()
        ax.bar(df_plot.index, df_plot[region])
        ax.set_xlabel("Year")
        ax.set_ylabel("Inbound Arrivals (Thousands)")
        ax.set_title(f"Inbound Arrivals for {region} (1995-2022)")

        # Ensure xticks are integers
        ax.set_xticks(np.arange(1995, 2023, 1))  # Set step size to 1 year
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        plt.xticks(rotation=90)  
        st.pyplot(fig)

    # Predictions
    st.subheader("Predictions for Future Years")
    predictions_dict = {}

    for metric in df.index:
        df_metric = df.loc[[metric]].drop(columns=["category", "unit"], errors="ignore").melt(var_name="Year", value_name="Value").dropna()
        df_metric["Year"] = pd.to_numeric(df_metric["Year"], errors="coerce").astype("Int64")
        df_metric = df_metric.dropna().sort_values("Year")

        if df_metric.empty or len(df_metric) < 2:
            continue

        X = df_metric["Year"].values.reshape(-1, 1)
        y = df_metric["Value"].values
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.arange(2022, datetime.now().year + 5, dtype=int).reshape(-1, 1)
        future_predictions = model.predict(future_years)
        predictions_dict[metric] = {"Year": future_years.flatten(), "Predicted Value": future_predictions}

    # Show predictions
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, data in predictions_dict.items():
        ax.plot(data["Year"], data["Predicted Value"], marker="o", linestyle="-", label=metric)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Predicted Metrics (2022-2032)")

    # Ensure xticks are integers
    ax.set_xticks(np.arange(2022, datetime.now().year + 5, 1))  # Set step size to 1 year
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    ax.legend(loc="upper left")
    ax.grid(True)
    st.pyplot(fig)

    # Model Evaluation
    st.subheader("Model Evaluation")
    evaluation_results = {}
    for metric in df.index:
        if metric not in predictions_dict:
            continue
        X = df_metric["Year"].values.reshape(-1, 1)
        y = df_metric["Value"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred_train = model.predict(X)
        mae = mean_absolute_error(y, y_pred_train)
        mse = mean_squared_error(y, y_pred_train)
        r2 = r2_score(y, y_pred_train)
        evaluation_results[metric] = {"MAE (Mean Absolute Error)": mae, "MSE (Mean Squared Error)": mse, "R² Score (Coefficient of Determination)": r2}
    evaluation_data = pd.DataFrame.from_dict(evaluation_results, orient='index')
    st.dataframe(evaluation_data)

    st.text("MAE: Measures the average absolute difference between predicted and actual values. Lower is better.")
    st.text("MSE: Similar to MAE but squares the differences, making larger errors more significant. Lower is better.")
    st.text("R² Score: Measures how well the model explains variance in the data. Ranges from -∞ to 1, where 1 is a perfect fit, 0 means no predictive power, and negative values indicate a poor fit.")

    # Export predictions
    st.info("Exporting and Download Data is not yet implemented")
    st.subheader("Export Predictions")
    if st.button("Generate Predictions and Download", disabled=True):
    # Prepare the DataFrame for export
        predicted_df = pd.DataFrame()
        for metric, data in predictions_dict.items():
            meta_info = df.loc[[metric], ["category", "unit"]].drop_duplicates()
            pred_data = pd.DataFrame([data["Predicted Value"]], columns=map(str, data["Year"]))
            pred_data.insert(0, "metric", metric)
            pred_data.insert(1, "category", meta_info.iloc[0, 0] if not meta_info.empty else "Unknown")
            pred_data.insert(2, "unit", meta_info.iloc[0, 1] if not meta_info.empty else "Unknown")
            predicted_df = pd.concat([predicted_df, pred_data], ignore_index=True)

        # Save the predictions to a BytesIO object (no need to save to a file on disk)
        predicted_df.to_excel(predicted_df, index=True)
        st.success("Predictions are ready for download!")

if st.button("Process Uploaded File"):
    if selected_key == "insert_url":
        if is_valid_url(uploaded_file):
            generate_tourism_data()
        else:
            st.error("Invalid URL. Please enter a valid URL.")
    else:     
        if uploaded_file:
            generate_tourism_data()
        else:
            st.error("No file uploaded.")
