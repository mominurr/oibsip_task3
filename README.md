# Car Price Prediction With Machine Learning

## Project Overview
This project focuses on predicting the selling price of cars using machine learning techniques. The dataset used for this project contains information about various car features, such as the car name, year of manufacture, present price, kilometers driven, fuel type, selling type, transmission, and the number of previous owners. The goal is to train a machine learning model that can accurately predict the selling price based on these features.

## Data Analysis
The dataset is loaded and analyzed to gain insights into its structure and characteristics. This includes exploring the first and last few rows of the dataset, checking its shape, data types, presence of null values, unique values, and descriptive statistics.

## Data Visualization
A function `data_visualization` is defined to visualize the data using different plots:

- **Correlation Heatmap**: Displays the correlation matrix of the dataset, providing insights into the relationships between features.
- **Selling Price Trends Over Years (Bar Plot)**: Illustrates the trend of selling prices over the years.
- **Scatter Plot of Present Price vs. Selling Price**: Shows the relationship between present price and selling price.

Also, use a scatter plot for the actual values and predicted values relationship in the graph for better understanding of the model.

## Script Details
The script includes the necessary libraries and two main functions:

1. `data_visualization`: Visualizes the dataset using various plots.
2. `price_prediction`: Loads the dataset, performs data preprocessing (including label encoding), visualizes the data, splits it into training and testing sets, trains a Random Forest Regressor model, and evaluates the model's performance. The trained model is saved for future use.

**Model Use**: After training, this model is used for prediction. For prediction, run `app.py` file.

## Video Representation
Check out the video representation of the project for a more interactive and engaging overview: [Unemployment Analysis Video](https://youtu.be/8iTeDf1O8DQ)

## Requirements
Ensure you have the following libraries installed to run the script:

- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy
- joblib

You can install these dependencies using pip:

    pip install pandas matplotlib seaborn scikit-learn numpy joblib
or

    pip install -r requirements.txt
    
## Usage
To use this project, follow these steps:
1. Ensure you have Python installed on your machine.
2. **For Training:**
   - Clone the repository: `git clone https://github.com/mominurr/oibsip_task3.git`
   - Install the required libraries: `pip install -r requirements.txt`
   - Run the script: `python car-price-predictor.py`
3. **For Prediction:**
   - Run the script: `python app.py`

## Conclusion
This project demonstrates the process of predicting car prices using machine learning. The Random Forest Regressor model shows promising results in predicting selling prices based on the provided features. The visualizations help in understanding the relationships between different variables in the dataset, aiding in feature selection and model interpretation. The trained model (`car_price_prediction_model.pkl`) can be used for future predictions.

##Author:
[Mominur Rahman](https://github.com/mominurr)
