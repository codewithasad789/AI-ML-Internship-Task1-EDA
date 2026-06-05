# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = sns.load_dataset('iris')

# Shape of dataset
print("Shape:", iris.shape)

# Column names
print("\nColumns:")
print(iris.columns)

# First 5 rows
print("\nFirst 5 Rows:")
print(iris.head())

# Dataset information
print("\nDataset Info:")
print(iris.info())

# Summary statistics
print("\nSummary Statistics:")
print(iris.describe())

# Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.show()

# Histograms
iris.hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()

# Box Plots
plt.figure(figsize=(10,6))
sns.boxplot(data=iris)
plt.title("Box Plot for Outlier Detection")
plt.show()
