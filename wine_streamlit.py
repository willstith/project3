#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import imblearn.over_sampling
from sklearn.svm import SVC
import time
from sklearn.metrics import precision_score,recall_score,precision_recall_curve,f1_score,fbeta_score,roc_auc_score
import pickle


#import the data
with open('sl_wine_df.p','rb') as read_file:
    data = pickle.load(read_file)

#with open('wine_df_from_sql.p','rb') as read_file:
#    data = pickle.load(read_file)
image = Image.open("wine_bottles.png")
st.title("Is Your Fine Wine or Not?")
st.image(image, use_column_width=True)

#checking the data
st.write("This application will try to predict whether your wine is fine or not. Give it a try!")
check_data = st.checkbox("See training data preview")
if check_data:
    st.write(data.head(10))
st.write("Now enter in your wine's physicochemical traits.")

#input the numbers
fixed_acidity = st.slider("Fixed acidity",float(data.fixed_acidity.min()),float(data.fixed_acidity.max()),float(data.fixed_acidity.mean()) )
volatile_acidity = st.slider("Volatile acidity",float(data.volatile_acidity.min()),float(data.volatile_acidity.max()),float(data.volatile_acidity.mean()) )
citric_acid = st.slider("Citric Acid",float(data.citric_acid.min()),float(data.citric_acid.max()),float(data.citric_acid.mean()) )
residual_sugar = st.slider("Residual Sugar",float(data.residual_sugar.min()),float(data.residual_sugar.max()),float(data.residual_sugar.mean()) )
chlorides = st.slider("Chlorides",float(data.chlorides.min()),float(data.chlorides.max()),float(data.chlorides.mean()) )
free_so2 = st.slider("Free sulfur dioxide",float(data.free_so2.min()),float(data.free_so2.max()),float(data.free_so2.mean()) )
unfree_so2 = st.slider("Unfree sulfur dioxide (total - free)",float(data.unfree_so2.min()),float(data.unfree_so2.max()),float(data.unfree_so2.mean()) )
density = st.slider("Density",float(data.density.min()),float(data.density.max()),float(data.density.mean()) )
ph = st.slider("pH",float(data.ph.min()),float(data.ph.max()),float(data.ph.mean()) )
sulphates = st.slider("Sulphates",float(data.sulphates.min()),float(data.sulphates.max()),float(data.sulphates.mean()) )
alcohol = st.slider("Alcohol by volume",float(data.alcohol.min()),float(data.alcohol.max()),float(data.alcohol.mean()) )
variety = st.slider("Red or white",int(data.variety.min()),int(data.variety.max()),int(data.variety.min()) )

features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_so2, unfree_so2, 
                     density, ph, sulphates, alcohol, variety]).reshape(1,-1)

#split data
X_cols = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar',
       'chlorides','free_so2','density','ph','sulphates',
       'alcohol','variety','unfree_so2']
X = data[X_cols]
y = data['goodbad']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=23)

#modelling step
#import model
# setup desired ratios for SMOTE
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
ratio = {1 : n_pos * 4, 0 : n_neg}

#scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = imblearn.over_sampling.SMOTE(sampling_strategy=ratio, random_state = 23)
X_tr_smote, y_tr_smote = smote.fit_sample(X_train_scaled, y_train)

rf_smote = RandomForestClassifier(random_state=23)
rf_smote.fit(X_tr_smote, y_tr_smote)
rf_smote.predict(X_test_scaled)
predictions = rf_smote.predict(scaler.transform(features))

#checking prediction
if st.button("Assess me!"):
    fine_or_not = 'is fine!'
    if int(predictions) == 0:
        fine_or_not = "doesn't make the cut."
    st.header("This wine {}".format(fine_or_not))
#    st.subheader("Your range of prediction is USD {} - USD {}".format(int(predictions-errors),int(predictions+errors) ))