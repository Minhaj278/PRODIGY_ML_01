import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
df = pd.read_csv("house_prices.csv")

# Select relevant features and target variable
features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = df['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Optionally, you can evaluate the model here
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.2f}")
