import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('AmesHousing.csv')

# 2. Handle missing values
columns_to_drop = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu']
df.drop(columns=columns_to_drop, inplace=True)

# Fill missing values in categorical columns with mode
garage_columns = ['Garage Cond', 'Garage Qual', 'Garage Finish', 'Garage Yr Blt', 'Garage Type']
for col in garage_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

basement_columns = ['Bsmt Exposure', 'BsmtFin Type 2', 'BsmtFin Type 1', 'Bsmt Qual', 'Bsmt Cond']
for col in basement_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill missing values in numerical columns with median
df['Lot Frontage'].fillna(df['Lot Frontage'].median(), inplace=True)
df['Mas Vnr Area'].fillna(df['Mas Vnr Area'].median(), inplace=True)
numerical_columns = ['Total Bsmt SF', 'Garage Cars', 'Garage Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF']
for col in numerical_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing values in 'Electrical' with mode
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

# 3. Feature Engineering
df = pd.get_dummies(df, drop_first=True)
df['TotalSF'] = df['Gr Liv Area'] + df['Total Bsmt SF']
df['HouseAge'] = df['Yr Sold'] - df['Year Built']
df['TotalBath'] = (df['Full Bath'] + (0.5 * df['Half Bath']) +
                   df['Bsmt Full Bath'] + (0.5 * df['Bsmt Half Bath']))

# 4. Train-test split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (Optional) Scale features if necessary
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Testing set: {X_test.shape}, {y_test.shape}')
