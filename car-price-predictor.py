# project name: Car Price Predictor With Machine Learning
# Author: Mominur Rahman
# Date: 24-11-2023
# Version: 1.0
# Description: This project is used to predict the price of a car based on various features such as Car Name, Present Price, selling price, owner,Driven kilometers, year, fuel type, transmission, and mileage.
# GitHub Repo: https://github.com/mominurr/oibsip_task3
# LinkedIn: https://www.linkedin.com/in/mominur-rahman-145461203/

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from datetime import datetime

# Define a function to visualize the data
def data_visualization(df):
    """
    Visualize the data in the dataset using different plots.
    
    Plots:
        - Heatmap: Correlation matrix for understanding feature relationships.
        - Bar plot: Selling price trends over the years.
        - Scatter plot: Present price vs. selling price relationship.

    Args:
        df: The dataset to be visualized.
    """


    # Calculate correlation matrix
    correlation_matrix = df.corr()
    # Create a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()

    # Create a bar plot
    plt.figure(figsize=(16, 11))
    plt.bar(df['Year'], df['Selling_Price'])
    plt.xlabel('Year')
    plt.ylabel('Selling Price')
    plt.title('Selling Price Trends Over Years')
    plt.xticks(rotation=90)
    plt.savefig('selling_price_trends.png')
    plt.show()

    
    # Create a scatter plot
    plt.figure(figsize=(16, 11))
    # Plot the scatter points with dx='Present_Price'ifferent colors
    plt.scatter(df['Present_Price'], df['Present_Price'], color='green', label='Present Price')
    plt.scatter(df['Present_Price'], df['Selling_Price'], color='red', label='Selling Price')
    plt.xlabel('Present Price')
    plt.ylabel('Selling Price')
    plt.title('Scatter Plot of Present Price vs. Selling Price')
    plt.legend()
    plt.savefig('present_price_vs_selling_price.png')
    plt.show()

    # # Create a pair plot
    # sns.pairplot(df)
    # plt.title('Pair Plot')
    # plt.savefig('pair_plot.png')
    # plt.show()


# This function is used to predict the price of a car based on various features
def price_prediction():
    # Load the CSV file
    df=pd.read_csv('car data.csv')

    # Display basic dataset information
    print("\nFirst 5 rows of the dataset: \n")
    print(df.head())

    print("\n\nLast 5 rows of the dataset: \n")
    print(df.tail())

    print("\n\nShape of the dataset: \n")
    print(df.shape)

    print("\n\nData types of the dataset: \n")
    print(df.dtypes)

    print("\n\nNull values in the dataset: \n")
    print(df.isnull().sum())

    print("\n\nUnique values in the dataset: \n")
    print(df.nunique())

    print("\n\nDescriptive statistics of the dataset: \n")
    print(df.describe())
    
    print("\n\nInformation about the dataset: \n")
    print(df.info())
    
    
    # print("\n\nPrint Unique values list in the Car_Name column: \n")
    # print(df['Car_Name'].unique())
   

    # Label Encoding
    label_encoder = LabelEncoder()
    df['Car_Name'] = label_encoder.fit_transform(df['Car_Name'])
    df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])
    df['Selling_type'] = label_encoder.fit_transform(df['Selling_type'])
    df['Transmission'] = label_encoder.fit_transform(df['Transmission'])
    print("\n\n First 5 rows of the dataset after Label Encoding: \n")
    print(df.head())

    # get the year difference between the current year and the year of the car
    current_year = datetime.now().year
    df['Year_Diff'] = current_year - df['Year']
    
    # data visualization for best understanding of the data. So we will use correlation matrix and  other visualizations.
    data_visualization(df)

    # Split the data into training and testing sets. 
    # We will use random_state=42 for reproducibility
    # We will use 80% of the data for training and 20% for testing.
    # Test_size=0.2 means 20% of the data will be used for testing.
    # X is the input features and Y is the target variable

    X=df[["Car_Name","Year_Diff","Present_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner"]]
    Y=df["Selling_Price"]
    X=X.values
    Y=Y.values
    # Model Selection and Training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a Random Forest Regressor model
    MODEL=RandomForestRegressor(random_state=42)

    # Fit the model
    MODEL.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = MODEL.predict(X_test)

    

    # show the predicted values and the actual values relationship in graph for better understanding of the model
    # Plot the actual vs. predicted values
    plt.figure(figsize=(16, 11))
    # Plot the scatter points with different colors
    plt.scatter(Y_test, Y_test, color='blue', label='Actual Values')
    plt.scatter(Y_test, Y_pred, color='red', label='Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.savefig('actual_vs_predicted.png')
    plt.show()

    # Calculate the mean absolute error
    MAE = mean_absolute_error(Y_test, Y_pred)
    print("\nModel Evaluation:\n")
    print("Mean Absolute Error: ", MAE)

    # Calculate the mean squared error
    MSE = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error: ", MSE)

    # Calculate the root mean squared error
    RMSE = np.sqrt(MSE)
    print("Root Mean Squared Error: ", RMSE)

    # # Calculate the R-squared score
    R2 = r2_score(Y_test, Y_pred)
    print("R-squared Score: ", R2, "\n")

    # Save the trained model
    joblib.dump(MODEL, 'car_price_prediction_model.pkl')




# Here the code executes
if __name__ == '__main__':
    print("Welcome to Car Price Prediction With Machine Learning!\n")
    price_prediction()

    



