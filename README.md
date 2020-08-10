# project3

## Classifying Wines - Fine or Not?

**Created by Will Stith**  
**08/09/2020**

**Introduction**

This folder contains files comprising my third project for Metis. The project is a classification model for wine quality based on the physicochemical traits of the wine. Specifically, the dataset used to construct the model contains over 6000 examples of both red and white wines of the Portuguese Vinho Verde variety. The data was loaded into Postgres where minimal feature engineering was performed. SQLAlchemy was then used to export unique rows of the dataset into a Pandas DataFrame to be modeled using Python. Modeling began by fitting six base classification models (K-Nearest Neighbors, Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and Gaussian Naive Bayes) and scoring upon four different classification metrics (F1, ROC, precision, and recall). Additional model tuning was then performed with the random forest model, the best-performing model according to every metric. In particular, the issue of class imbalance was addressed by applying SMOTE resampling. The final model's F1 score was 0.561 with an ROC-AUC of 0.851. In addition, a Streamlit app was built for the model, which allows a user to input the characteristics of a new wine and test whether the model classifies that wine as fine or not.

**Requirements**

Python 3.?
Necessary modules (listed below)

**Necessary modules**

- pickle
- pandas
- seaborn
- matplotlib
- numpy
- sklearn
- SQLAlchemy
- copy
- Streamlit
- PIL


**Data**

The data was downloaded from the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/wine+quality as two separate CSV files - one for red wine (1600 rows) and one for white wine (4899 rows). Each wine has 14 attributes describing its physicochemical characteristics (e.g. density, pH, alcohol, chlorides). There are no labels associated with individual wines, and in fact there were more than 1000 duplicate rows in the original data. Of special important is the 'quality' feature, which is the average of 3 ratings for the wine's quality (presumably rounded down, as every value was a whole number, and there were no perfect 10's). The target variable for this project was "goodbad", which was a binary feature based on whether or not the wine's quality rating was greater than 6.

**Contents**

*project3_data_intake.ipynb* - contains code which uses SQLAlchemy to read data from the psql database into a Pandas DataFrame.  
*project3_modeling.ipynb* - contains code for building the classification model, as well as some additional data cleaning.  
*wine_streamlit.py* - contains code for the Streamlit app associated with the model.  
*project3_psql_work.txt* - contains code for creating psql tables from the project data as well as the psql queries used to clean and modify tables.  
*sl_wine_df.p* - pickled save of the data used in running the Streamlit app.  
*sl_wine_model.p* - pickled save of the random forest model used in running the Streamlit app.  
*wine_df_from_sql.p* - pickled save of the wine DataFrame after exporting from psql.  
*wine_bottles.png* - png image for Streamlit page.

**Instructions**

Download data from the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/wine+quality. Follow the instructions in project3_psql_work to load the data into psql and query/modify the table. Then convert the resulting table into a Pandas DatFrame by following the project3_data_intake.ipynb notebook. Finally, use the project3_modeling.ipynb notebook to recreate the random forest model from this project. If you wish to test novel wines for quality according to this model, enter "streamlit run wine_streamlit.py" in a terminal to launch the Streamlit app. Follow instructions within to input your wine's features.

**Acknowledgements**

Thank you to all my Metis instructors and TAs for their support and guidance for this project and others. Additionally, I'd like to thank my fellow Metis students for positively contributing to the learning environment. Special thanks to Paolo Cortez and colleagues at the University of Minho in Guimaraes, Portugal for making the data used in this project publicly available, as well as to the University of California, Irvine for maintaining their Machine Learning Repository which hosted the data.  
    Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
    A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal
    @2009
