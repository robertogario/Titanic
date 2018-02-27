# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:32:12 2018

@author: gary.roberts
"""

import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas.plotting import parallel_coordinates
from pandas.plotting import radviz
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from titanic_utils import DataFrameSelector
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

def print_section(text):
    print("\n")
    print("----------------------------------------")
    print(str(text))
    print("----------------------------------------")

print_section("Loading Data")
orig_data = pd.read_csv(r"C:\Source\Py_Projects\Titanic\data\train.csv")

print(df.head())
print(df.describe())
print(df.corr())

print_section("Dropping unnecessary columns")
# make a copy without columns that aren't useful for ML
df = orig_data.drop("Name", axis=1).drop("Cabin", axis=1).drop("PassengerId", axis=1)

print_section("Splitting data and labels")
titanic_data = df
# Copy labels. They will be removed from the original set during the pipeline.
titanic_labels = df["Survived"].copy()

print_section("Data cleansing pipeline")
# The line below can be used to check the learning. If it is uncommented
# in place of the other num_attribs row, it should result in a score of
# 100% for the model.
#num_attribs = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]

# Pipeline to process data begins here.
num_attribs = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Sex"]
total_attribs = num_attribs + cat_attribs

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler',StandardScaler()),
    ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

# Run pipeline
titanic_prepared = full_pipeline.fit_transform(titanic_data)

# This section is just because once the categorical data has been 
# enumerated it makes sense to recheck correlations.
print_section("Check correlations in normalised data")
tit_prep_pd = pd.DataFrame(data=titanic_prepared, columns=total_attribs)
tit_prep_pd_labelled = tit_prep_pd.assign(Survived=titanic_labels.values)
# Correlation of the prepared data allows us to check correlation of numerics
print(tit_prep_pd_labelled.corr())
sns.pairplot(tit_prep_pd_labelled[num_attribs + list(["Survived"])], hue="Survived", size = 2.5)

# Split training and testing data and labels in the usual way.
print_section("Split data, train and test model")
X_train, X_test, y_train, y_test = \
        train_test_split(titanic_prepared, titanic_labels, test_size=.3, random_state=42)
        
clf = LinearSVC(random_state=0)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print("Test Accuracy: {0:.2f}%".format(score * 100.0))

#predictions = clf.predict(<test data>)