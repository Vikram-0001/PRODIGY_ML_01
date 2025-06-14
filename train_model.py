import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Load and prepare data
df = pd.read_csv('train.csv')
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model and metrics
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'mse': mse, 'r2': r2}, f)
