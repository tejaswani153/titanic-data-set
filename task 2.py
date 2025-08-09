import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

# --- Step 1: Pick file interactively ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Titanic CSV File",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected. Exiting...")
    exit()

# --- Step 2: Load the dataset ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
    exit()

# Clean column names (remove spaces, unify case)
df.columns = df.columns.str.strip()

print("\nColumn names in the dataset:")
print(df.columns.tolist())

# --- Step 3: Data Cleaning ---
# Handle Age
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
elif 'age' in df.columns:
    df.rename(columns={'age': 'Age'}, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
else:
    print("\nWarning: 'Age' column not found. Skipping age cleaning.")

# Handle Embarked
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin if exists
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\nData Info after Cleaning:")
df.info()

# --- Step 4: EDA ---
if 'Survived' in df.columns:
    print("\n--- Exploratory Data Analysis ---")
    survival_rate = df['Survived'].value_counts(normalize=True) * 100
    print(f"\nSurvival Rate:\n{survival_rate}")

    if 'Sex' in df.columns:
        print("\nSurvival by Gender:")
        print(df.groupby('Sex')['Survived'].mean() * 100)

    if 'Pclass' in df.columns:
        print("\nSurvival by Passenger Class:")
        print(df.groupby('Pclass')['Survived'].mean() * 100)

# --- Step 5: Visualization ---
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

if {'Sex', 'Survived'}.issubset(df.columns):
    plt.subplot(2, 2, 1)
    sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
    plt.title('Survival Rate by Gender')

if {'Pclass', 'Survived'}.issubset(df.columns):
    plt.subplot(2, 2, 2)
    sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
    plt.title('Survival Rate by P-Class')

if 'Age' in df.columns:
    plt.subplot(2, 2, 3)
    sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution')

    if 'Survived' in df.columns:
        plt.subplot(2, 2, 4)
        sns.histplot(x='Age', hue='Survived', data=df, multiple='stack', bins=20, palette='RdBu')
        plt.title('Survival by Age')

plt.tight_layout()
plt.show()

# --- Step 6: Final Observations ---
print("\n--- Key Patterns Identified ---")
if 'Sex' in df.columns:
    print("- Females had a significantly higher survival rate than males.")
if 'Pclass' in df.columns:
    print("- Passengers in First Class had a much higher survival rate than those in Third Class.")
if 'Age' in df.columns:
    print("- Children (younger age) appear to have a higher survival rate compared to other age groups.")
