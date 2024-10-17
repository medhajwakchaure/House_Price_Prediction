import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set the style for Seaborn plots
sns.set(style="whitegrid")

# 1. Load the dataset
df = pd.read_csv('AmesHousing.csv')

# 2. Handle missing values
columns_to_drop = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu']
df.drop(columns=columns_to_drop, inplace=True)

# Fill missing values in categorical columns with mode
garage_columns = ['Garage Cond', 'Garage Qual', 'Garage Finish', 'Garage Yr Blt', 'Garage Type']
for col in garage_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

basement_columns = ['Bsmt Exposure', 'BsmtFin Type 2', 'BsmtFin Type 1', 'Bsmt Qual', 'Bsmt Cond']
for col in basement_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing values in numerical columns with median
df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(df['Mas Vnr Area'].median())
numerical_columns = ['Total Bsmt SF', 'Garage Cars', 'Garage Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF']
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing values in 'Electrical' with mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# 3. Feature Engineering
df = pd.get_dummies(df, drop_first=True)

# Ensure that the column names match your dataset
df['TotalSF'] = df['Gr Liv Area'] + df['Total Bsmt SF']
df['HouseAge'] = df['Yr Sold'] - df['Year Built']
df['TotalBath'] = (df['Full Bath'] + (0.5 * df['Half Bath']) +
                   df['Bsmt Full Bath'] + (0.5 * df['Bsmt Half Bath']))

# 4. Train-test split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features (optional, if using models sensitive to scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the shapes of the training and testing sets
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Testing set: {X_test.shape}, {y_test.shape}')


### Exploratory Data Analysis (EDA) ###

# 6.1. Visualizing the Distribution of House Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, color='blue', bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# 6.2. Analyzing Relationships Between Features and House Prices

# Scatter Plot: TotalSF vs SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalSF', y='SalePrice', data=df, color='green')
plt.title('Total Square Footage vs Sale Price')
plt.xlabel('Total Square Footage')
plt.ylabel('Sale Price')
plt.show()

# Scatter Plot: HouseAge vs SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HouseAge', y='SalePrice', data=df, color='red')
plt.title('House Age vs Sale Price')
plt.xlabel('House Age')
plt.ylabel('Sale Price')
plt.show()

# Correlation Matrix: Visualizing correlations between features and SalePrice
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# 6.3. Investigating Outliers

# Boxplot: SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x='SalePrice', data=df)
plt.title('Boxplot of Sale Price')
plt.show()

# Identifying potential outliers in TotalSF vs SalePrice
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalSF', y='SalePrice', data=df)
plt.title('Scatter Plot of TotalSF vs SalePrice (Outliers Highlighted)')
plt.xlabel('TotalSF')
plt.ylabel('SalePrice')

# Highlighting outliers
outliers = df[(df['TotalSF'] > 6000) & (df['SalePrice'] < 200000)]
sns.scatterplot(x=outliers['TotalSF'], y=outliers['SalePrice'], color='red', s=100, label="Outliers")
plt.legend()
plt.show()

# Handling outliers: Removing the identified outliers
df = df[~((df['TotalSF'] > 6000) & (df['SalePrice'] < 200000))]

print(f"Data shape after removing outliers: {df.shape}")
