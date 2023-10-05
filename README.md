# Data Preprocessing and Modeling in Python

This Python script is designed to help you with data preprocessing and modeling for various machine learning tasks. It offers functionalities for handling common data preprocessing tasks like handling categorical variables, missing values, and feature selection, as well as building and evaluating predictive models for both classification and regression problems.

## Getting Started

Follow these steps to get started with this script:

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```
   git clone https://github.com/dineshghadge2002/Auto-Preprocessing-Machine-learing-model.git
   ```

2. **Install Dependencies**

   Ensure you have the necessary Python libraries installed. You can install them using pip:

   ```
   pip install pandas matplotlib seaborn numpy scikit-learn
   ```

3. **Run the Script**

   Run the script by executing the `data_pre_model.py` file:

   ```
   python data_pre_model.py
   ```

## Usage

Here's how to use the script effectively:

1. **Dataset Input**

   - When prompted, enter the path or URL of your dataset. The script will load the dataset into a Pandas DataFrame.

2. **Selecting the Target Column**

   - Enter the name of the target column for your predictive modeling task. The script will set this column as the target variable (dependent variable) and remove it from the feature set.

3. **Handling Categorical Variables**

   - The script automatically encodes categorical variables using Label Encoding if they have fewer than 100 unique values.

4. **Feature Selection**

   - The script performs basic feature selection by removing columns that have string values or if the number of unique values in a column is equal to the number of data points (assuming it's not the target variable).

5. **Handling Missing Values**

   - If there are missing values in the dataset, the script calculates the mean for each numeric column and fills the missing values with the respective column's mean.

6. **Modeling and Evaluation**

   - If the target variable is binary or has two unique values, the script performs Logistic Regression for classification tasks.
   - If the target variable is numeric or has more than two unique values, the script performs Linear Regression for regression tasks.
   - The script calculates and displays various evaluation metrics such as accuracy, confusion matrix, MAE, MSE, RMSE, and R-squared.

7. **Results and Analysis**

   - The script provides insights into the performance of the chosen model and displays key metrics to help you assess the quality of your predictions.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

