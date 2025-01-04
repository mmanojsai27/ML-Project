𝘼𝙗𝙨𝙩𝙧𝙖𝙘𝙩

This study leverages a dataset containing monthly temperature readings across different locations to build a predictive model. Using Random Forest regression, we aim to forecast encoded categorical outcomes based on monthly temperature features. The project emphasizes preprocessing, model evaluation, and visualization to derive meaningful insights from the data.

 
𝙆𝙚𝙮𝙬𝙤𝙧𝙙𝙨

Temperature Prediction

Random Forest

Feature Engineering

Regression Analysis

Data Preprocessing

Machine Learning


𝘼𝙗𝙤𝙪𝙩 𝘿𝙖𝙩𝙖𝙨𝙚𝙩

The dataset contains monthly temperature data for various locations. Key details:

Columns: Unnamed: 0 (categorical target variable), Jan to Dec (monthly temperatures in degrees Celsius).

Size: 33 rows and 13 columns.

Preprocessing: Non-numeric characters were removed from monthly temperature columns, and the categorical target column was encoded numerically using Label Encoding.

𝙋𝙧𝙤𝙥𝙤𝙨𝙚𝙙 𝘼𝙡𝙜𝙤𝙧𝙞𝙩𝙝𝙢

The following steps outline the proposed method:

1.Data Cleaning:
Handle non-numeric values in temperature columns.
Encode categorical target variables numerically.

2.Feature Selection:
Use Jan to Dec as predictor variables.
Unnamed: 0 serves as the target variable.

3.Model Training:
Train a Random Forest Regressor with default parameters.
Split the dataset into 80% training and 20% testing.

4.Evaluation Metrics:
Mean Squared Error (MSE)
R-squared (R²)

5.Visualization:
Predictions vs. True Values
Feature Importance (using SHAP)
Error metrics comparison


𝙍𝙚𝙨𝙪𝙡𝙩𝙨 𝙖𝙣𝙙 𝙋𝙚𝙧𝙛𝙤𝙧𝙢𝙖𝙣𝙘𝙚

The proposed model achieved 87.5% classification accuracy metrics: Training MSE, Validation MSE, Training R², Validation R², Simulated F1-Score

Key Visualizations:
Distribution of predictions and true values
Feature importance ranking using SHAP values
Correlation heatmap among monthly temperature features


𝙍𝙚𝙥𝙧𝙤𝙙𝙪𝙘𝙞𝙗𝙞𝙡𝙞𝙩𝙮

The analysis can be reproduced using the provided dataset and code. Random seed (random_state=42) ensures consistency in model training and testing splits.

𝘿𝙚𝙥𝙚𝙣𝙙𝙚𝙣𝙘𝙞𝙚𝙨 𝙖𝙣𝙙 𝙍𝙚𝙦𝙪𝙞𝙧𝙚𝙢𝙚𝙣𝙩𝙨

Python Libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn

shap

𝙏𝙤 𝙞𝙣𝙨𝙩𝙖𝙡𝙡 𝘿𝙚𝙥𝙚𝙣𝙙𝙚𝙣𝙘𝙞𝙚𝙨, 𝙧𝙪𝙣

Install the required libraries using the following command:

pip install numpy pandas matplotlib seaborn scikit-learn shap

𝙎𝙚𝙧𝙫𝙚𝙧 𝙖𝙣𝙙 𝙃𝙖𝙧𝙙𝙬𝙖𝙧𝙚 𝙍𝙚𝙦𝙪𝙞𝙧𝙚𝙢𝙚𝙣𝙩𝙨

Hardware:

Minimum: Dual-core processor, 4 GB RAM
Recommended: Quad-core processor, 8 GB RAM

Software:

Operating System: Windows, Linux, or macOS
Python 3.7 or later
Jupyter Notebook or any Python IDE for running scripts


