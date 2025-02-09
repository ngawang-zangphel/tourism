import pandas as pd

df = pd.read_excel('/Users/nawang/Downloads/bhutan-tourism-data.xlsx')

df_cleaned = df.iloc[2:].reset_index(drop=True)

df_cleaned.columns = ["metric", "category", "unit"] + list(df_cleaned.iloc[0, 3:].values)

df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)

df_cleaned.replace("..", '0', inplace=True)

# Convert year columns to numeric values
year_columns = df_cleaned.columns[3:]
df_cleaned[year_columns] = df_cleaned[year_columns].apply(pd.to_numeric, errors="coerce")

# Correct year column names
df_cleaned.columns = ["metric", "category", "unit"] + list(range(1995, 2023))

# Save with NaNs intact
df_cleaned.to_excel("cleaned_tourism.xlsx", index=False, engine='openpyxl')

print(df_cleaned.head())

df.to_excel("output.xlsx", index=True)
