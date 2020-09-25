import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_housing_data(house_path):
    return pd.read_csv(house_path)


housing = load_housing_data("chap2/housing.csv")  # Loading the data
# getting familiar with the data
# housing.head()

# housing.info()

# housing["ocean_proximity"].value_counts()

# housing.describe()

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

np.random.seed(42)  # For making np functions output same values everytime

housing["income_cat"] = pd.cut(housing["median_income"],  # makes a new income category column
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing.head()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # returns a objects that can be used

for train_index, test_index in split.split(housing, housing["income_cat"]):  # yeilds train and test that are stratified
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# visualizing data
housing = strat_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  # see the density

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,  # A more sophisticated scatter plot
#             s=housing["population"] / 100, label="population", figsize=(10, 7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#             )
# plt.legend()

cor_matrix = housing.corr()

# cor_matrix.head()

# cor_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))  # pandas plots every attribute against every attribute

# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

cor_matrix = housing.corr()
# cor_matrix["median_house_value"].sort_values(ascending=False)

# cleaning data

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# filling missing data

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

imputer.statistics_  # It is now trained
housing_num.median().values  # check if the imputer is trained

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
# converting textual data into numerical values

housing_cat = housing[["ocean_proximity"]]
# housing_cat.head(10)

# simple encoding
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# housing_cat_encoded[:10]
# ordinal_encoder.categories_

# 1hot encoding
# cat_encoder = OneHotEncoder()

# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# housing_cat_1hot.toarray()

# custom transformers
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

# pipelining the data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room=True)),
    ('std_scaler', StandardScaler()),
])

# housing_num_tr = num_pipeline.fit_transform(housing_num)
# housing_num_tr

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared

# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_ready = full_pipeline.transform(some_data)
lin_reg.predict(some_data_ready)
some_labels

# calculating root mean squared error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

# Decision trees
des_tree = DecisionTreeRegressor()
des_tree.fit(housing_prepared, housing_labels)

tree_prediction = des_tree.predict(housing_prepared)
mean_squared_error(housing_labels, tree_prediction)

# k-folds Cross validation
scores = cross_val_score(des_tree, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(score):
    print("Scores:", score)
    print("Mean:", score.mean())
    print("Standard deviation:", score.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

display_scores(np.sqrt(-lin_scores))

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_score = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
display_scores(np.sqrt(-forest_score))

joblib.dump(forest_reg, "random_forest.pkl")

my_model_loaded = joblib.load("random_forest.pkl")
forest_score = cross_val_score(my_model_loaded, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
display_scores(np.sqrt(-forest_score))
