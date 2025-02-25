import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Function to load and preview the dataset
def load_data(filepath):
    """Loads the dataset and returns a DataFrame."""
    df = pd.read_csv(filepath)
    print("Data Loaded Successfully!\n")
    print("Dataset Preview:")
    print(df.head())
    return df

# 2. Function for Data Cleaning
def clean_data(df):
    """Cleans the dataset by handling missing values and duplicates."""
    df_cleaned = df.dropna()
    print(f"\nDropped missing values. \nNew shape: {df_cleaned.shape}")
    
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Dropped duplicates. \nFinal shape: {df_cleaned.shape}")

    return df_cleaned

# 3. Function to display data summary
def data_summary(df):
    """Provides info on data types and summary statistics."""
    print("\nDataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

# 4. univariate_analysis 
def univariate_analysis(df, numerical_cols):
    """Plots multiple types of charts for numerical features side by side."""
    for col in numerical_cols:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'\n\n\nUnivariate Analysis for {col}', fontsize=16)

        # Histogram
        sns.histplot(df[col], kde=True, bins=30, ax=axes[0, 0])
        axes[0, 0].set_title(f'Histogram of {col}')

        # Boxplot
        sns.boxplot(x=df[col], ax=axes[0, 1])
        axes[0, 1].set_title(f'Boxplot of {col}')

        # Violin Plot
        sns.violinplot(x=df[col], ax=axes[1, 0])
        axes[1, 0].set_title(f'Violin Plot of {col}')

        # Density Plot
        sns.kdeplot(df[col], fill=True, ax=axes[1, 1])
        axes[1, 1].set_title(f'Density Plot of {col}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# 5. Function to plot Correlation Heatmap
def correlation_heatmap(df):
    """Plots a correlation heatmap for numerical features."""
    print("\n\n")
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# 6. Bivariate Analysis:
def bivariate_analysis(df):
    """
    Bivariate Analysis: Plots 5 different charts to show relationships between features and Machine Failure.
    """
    print(f"\n\n Bivariate Analysis\n\n")
    plt.figure(figsize=(5,3.5))
    sns.boxplot(x='Type', y='Air temperature [K]',hue = "Machine failure", data=df, palette='Set2')
    plt.title('Boxplot: Air Temperature vs Machine Type')
    plt.show()

    plt.figure(figsize=(5,3.5))
    sns.violinplot(x='Type', y='Rotational speed [rpm]', hue = "Machine failure", data=df, palette='muted')
    plt.title('Violin Plot: Rotational speed by Machine Type ')
    plt.show()
    
    plt.figure(figsize=(4,3))
    sns.relplot(x='Type', y ='Torque [Nm]', hue = "Machine failure", data=df, palette='Set2')
    plt.title('Scatter Plot : Torque vs Machine Break Down By Friction')
    plt.show()

    plt.figure(figsize=(5,3.5))
    sns.countplot(x="Type", hue="Machine failure", data=df, palette='Set2')
    plt.title('Count Plot: Machine Type vs Count Machine Type')
    plt.show()

    plt.figure(figsize=(5,3.5))
    for label in df['Machine failure'].unique():
        sns.kdeplot(data=df[df['Machine failure'] == label], x='Process temperature [K]', fill=True, label=f'Machine Failure = {label}')
    plt.title('KDE Plot: Temperature Distribution by Machine Failure')
    plt.legend()
    plt.show()

# 7. Function for Failure Mode Analysis (if applicable)
def failure_mode_analysis(df, failure_modes):
    """Plots the frequency of different failure modes."""
    failure_counts = df[failure_modes].sum().sort_values(ascending=False)
    plt.figure(figsize=(6, 3))
    sns.barplot(x=failure_counts.index, y=failure_counts.values,palette='Set2')
    plt.title('Failure Modes Frequency')
    plt.xlabel('Failure Mode')
    plt.ylabel('Count')
    plt.show()

# Main Function which Contains all the sub function into it.
def main():
    filepath = r"https://raw.githubusercontent.com/Rishabh45/Tata_Steel_Machine_Failure_Analysis/refs/heads/main/train%20(2).csv"
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    df = load_data(filepath)

    df_cleaned = clean_data(df)

    data_summary(df_cleaned)

    # Univariate Analysis
    univariate_analysis(df_cleaned, numerical_cols)

    # correlation heatmap
    correlation_heatmap(df_cleaned)

    # Bivariate Analysis 
    bivariate_analysis(df_cleaned)

    # Failure Mode Analysis
    failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] 
    failure_mode_analysis(df_cleaned, failure_modes) 

# Calling the main function.
if __name__ == "__main__":
    main()

