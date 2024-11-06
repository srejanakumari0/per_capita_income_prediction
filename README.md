**Per Capita Income Prediction**
This project involves predicting the per capita income of different regions using machine learning techniques. The goal is to build a model that can forecast income based on various economic factors. This is a regression problem that demonstrates the application of data preprocessing, feature engineering, and machine learning algorithms.

#Table of Contents
#Project Overview
#Technologies Used
#Dataset
Installation and Setup
Usage
Model Evaluation
License
**Project Overview**
In this project, I worked with a real-world dataset to predict per capita income using regression models. The dataset contains various socio-economic factors that contribute to the income levels of different regions.

Key Steps:
#Data Preprocessing: Cleaned and prepared the data for analysis.
#Feature Engineering: Extracted relevant features and transformed the data for better model performance.
#Modeling: Built multiple regression models, including Linear Regression and Random Forest Regressor, to predict per capita income.
#Evaluation: Assessed the performance of the models using common metrics like Mean Squared Error (MSE) and R-squared.
#Visualization: Used data visualization tools to analyze the trends and predictions.
**Technologies Used**
Python: The core programming language for this project.
Libraries:
Pandas for data manipulation and analysis.
NumPy for numerical operations.
Matplotlib and Seaborn for data visualization.
Scikit-learn for machine learning models and evaluation.
Dataset
The dataset used for this project is a collection of economic factors that can influence income, including:

GDP (Gross Domestic Product)
Literacy rate
Population size
Urbanization (percent of urban population)
Educational levels
The dataset is pre-processed for missing values and outliers to ensure accurate predictions.

Installation and Setup
To run this project locally, follow these steps:

**Clone the repository:**

bash
Copy code
git clone https://github.com/srejanakumari0/per_capita_income_prediction.git
Navigate to the project directory:

bash
Copy code
cd per_capita_income_prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the notebook:

You can open and run the Jupyter Notebook (income_prediction.ipynb) to explore the entire workflow.

bash
Copy code
jupyter notebook income_prediction.ipynb
Usage
Once you open the Jupyter notebook, follow the steps to:

Load the dataset.
Perform data cleaning and preprocessing.
Train the regression models.
Visualize the results and interpret the predictions.
You can modify the dataset or model parameters to experiment and improve the results.

Model Evaluation
After training the models, the following evaluation metrics were used to assess performance:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
R-squared (R²): Represents the proportion of variance explained by the model.
Model performance results:

Linear Regression Model: Achieved an R² score of X.XX.
Random Forest Regressor: Achieved an R² score of Y.YY.
**License**
This project is licensed under the MIT License - see the LICENSE file for details.
