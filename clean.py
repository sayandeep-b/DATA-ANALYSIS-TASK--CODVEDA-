import pandas as pd

file_path = r"C:\Users\SAYANDEEP\Desktop\DATA ANLYTICS CSV\iris.csv"
df = pd.read_csv(file_path)

print(" First 5 rows:\n", df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Missing values per column:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

if 'species' in df.columns:
    df['species'] = df['species'].str.strip().str.lower()  

print("\n Cleaned Dataset Info:")
print(df.info())

output_path = r"C:\Users\SAYANDEEP\Desktop\DATA ANLYTICS CSV\iris_cleaned.csv"
df.to_csv(output_path, index=False)

print("\n Data cleaning done. Cleaned file saved as 'iris_cleaned.csv'")
