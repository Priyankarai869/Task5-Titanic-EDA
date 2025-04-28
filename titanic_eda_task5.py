# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Titanic Dataset
df = pd.read_csv('train.csv')   

# Step 3: Basic Info
print("First 5 rows:")
print(df.head())

print("\nData Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())
 
print("\nSurvived Value Counts:")
print(df['Survived'].value_counts())

print("\nSex Value Counts:")
print(df['Sex'].value_counts())

print("\nPclass Value Counts:")
print(df['Pclass'].value_counts())

# Step 4: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5A: Pairplot
sns.pairplot(df, hue='Survived')  # hue color based on 'Survived' (0 or 1)
plt.suptitle('Pairplot of Titanic Data', y=1.02)  # Title for pairplot
plt.show()

# Step 5B: Heatmap of Correlation Matrix
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Step 6: Histogram of Age
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Step 7: Boxplot (Age vs Pclass)
plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Boxplot of Age by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# Step 8: Scatterplot (Fare vs Age)
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Scatterplot of Fare vs Age')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived')
plt.show()



