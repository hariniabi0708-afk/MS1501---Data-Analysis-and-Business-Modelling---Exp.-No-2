# Exp No.2 : Exploratory Data Analysis (EDA)

## Aim
To perform Exploratory Data Analysis (EDA) on the heart disease dataset to understand the data distribution, relationships, and key patterns useful for business insights and model building.

## Dataset
The dataset used for this experiment is `heartdisease.csv`. It contains medical and demographic information such as age, sex, cholesterol levels (`totChol`), `BMI`, blood pressure, and target values (`TenYearCHD`) indicating heart disease presence.

## Algorithm
1. Import required Python libraries (`numpy`, `pandas`, `seaborn`, `matplotlib`).
2. Load the dataset into a Pandas DataFrame.
3. Display the dataset structure, shape, and descriptive statistics.
4. Check for missing values and duplicate rows.
5. Examine correlations among numeric features using a correlation matrix and heatmap.
6. Visualize univariate distributions using histograms, boxplots, and countplots.
7. Analyze bivariate and multivariate relationships using scatter plots and pairplots.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Basic Exploration** | `df.info()` | Inspect dataset structure and data types |
| | `df.describe()` | Generate descriptive statistics |
| | `df.isnull().sum()` | Detect missing values in each column |
| | `df.duplicated().sum()` | Identify duplicate rows |
| | `df.corr()` | Find correlations between numeric variables |
| **Univariate Analysis** | `sns.histplot()` | Plot histogram (distribution of single variable) |
| | `sns.boxplot()` | Identify outliers using boxplot |
| | `sns.countplot()` | Count occurrences of categorical variables |
| **Bivariate Analysis** | `sns.scatterplot()` | Display relationship between two variables |
| | `sns.heatmap()` | Show correlation matrix visually |
| **Multivariate Analysis** | `sns.pairplot()` | Show pairwise relationships across variables |
| **Visualization Controls** | `plt.figure()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.show()` | Configure and display plotted charts |

## Program & Output
# Step 1: Import essential libraries
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Improve the default appearance of plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# Step 2: Load the dataset
# Keep heartdisease.csv in the same folder as this notebook.
file_path = "heartdisease.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
display(df.head())


# Step 3: Basic exploration

# Inspect dataset structure and data types
print("Dataset Information:")
df.info()

# Generate descriptive statistics for numerical columns
print("\\nDescriptive Statistics:")
display(df.describe())

# Check number of rows and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display first 10 rows
print("\\nFirst 10 Rows:")
display(df.head(10))


# Step 4: Check data quality

# Check for missing values in each column
print("Missing Values in Each Column:")
display(df.isnull().sum())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print("Number of Duplicate Rows:", duplicate_count)


# Step 5: Correlation analysis

# Calculate correlation between numeric columns
correlation_matrix = df.corr(numeric_only=True)
print("Correlation Matrix:")
display(correlation_matrix)

# Visualize correlation using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.tight_layout()
plt.show()


# Step 6: Univariate analysis

# Histogram for Age distribution (if the column is available)
if "age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"].dropna(), kde=True, bins=20, color="teal")
    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'age' is not available in this dataset.")

# Boxplot for Cholesterol (to detect possible outliers)
if "totChol" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["totChol"])
    plt.title("Boxplot of Cholesterol Levels")
    plt.xlabel("Cholesterol")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'totChol' is not available in this dataset.")

# Countplot for Gender (if available)
if "sex" in df.columns:
    plt.figure(figsize=(5, 4))
    sns.countplot(x="sex", data=df)
    plt.title("Count of Gender")
    plt.xlabel("Sex (check dataset coding)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'sex' is not available in this dataset.")


# Step 7: Bivariate analysis

# Scatter plot: Age vs Cholesterol, with heart disease target as hue
required_columns = {"age", "totChol", "TenYearCHD"}
if required_columns.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="age",
        y="totChol",
        data=df,
        hue="TenYearCHD",
        palette="coolwarm"
    )
    plt.title("Age vs Cholesterol (by Heart Disease Target)")
    plt.xlabel("Age")
    plt.ylabel("Cholesterol")
    plt.tight_layout()
    plt.show()
else:
    print("Scatter plot skipped. Required columns:", sorted(required_columns))


# Step 8: Multivariate analysis

# Pairplot for multiple relationships
selected_columns = ["age", "totChol", "BMI", "TenYearCHD"]
available_columns = [col for col in selected_columns if col in df.columns]

if len(available_columns) >= 2:
    sns.pairplot(df[available_columns].dropna(), diag_kind="kde")
    plt.suptitle("Pairwise Relationships between Variables", y=1.02)
    plt.show()
else:
    print("Pairplot skipped. Fewer than two selected columns are available.")


## Result
