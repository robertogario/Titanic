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
from titanic_utils import DataFrameSelector
from sklearn.pipeline import FeatureUnion

orig_data = pd.read_csv(r"C:\Source\Py_Projects\Titanic\data\train.csv")

# make a copy without columns that aren't useful for ML
df = orig_data.drop("Name", axis=1).drop("Cabin", axis=1).drop("PassengerId", axis=1)

titanic_data = df
titanic_labels = df["Survived"].copy()

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

titanic_prepared = full_pipeline.fit_transform(titanic_data)

#print(df.head())

#print(df.describe())

print(df.corr())
# It appears that the strongest correlations with Survival are on class, fare, 

#attributes = ["Survived","Pclass", "Fare", "Sex"]
#scatter_matrix(df, alpha=0.2)

#sns.pairplot(df[attributes], hue="Survived", size= 2.5)
#sns.swarmplot(df[attributes])

#df.plot(kind="scatter", x="Fare", y="Pclass", c="Survived", alpha=0.1)

#parallel_coordinates(df[attributes], "Survived")
#radviz(df[attributes],"Survived")

tit_prep_pd = pd.DataFrame(data=titanic_prepared, columns=total_attribs)
tit_prep_pd_labelled = tit_prep_pd.assign(Survived=titanic_labels.values)
sns.pairplot(tit_prep_pd_labelled[num_attribs + list(["Survived"])], hue="Survived", size = 2.5)
