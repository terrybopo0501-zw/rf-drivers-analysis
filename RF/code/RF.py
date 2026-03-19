#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
from pprint import pprint
import os
import dask.dataframe as ddf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import (
    PartialDependenceDisplay,
    partial_dependence,
    permutation_importance,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from RF_operations import *


# In[13]:


# split test and validation datasets
exclude_columns = []
pred_attr = "DGWS"
output_dir = "/home/terry/working/groundwater/grace/deep_groundwater/figure/random_forest/project/RF/model/Predictors/"
csv_dir = "/home/terry/working/groundwater/grace/deep_groundwater/figure/random_forest/project/RF/model/Predictors/total.csv"
x_train, x_test, y_train, y_test, predictor_name_dict = split_train_test_ratio(
    predictor_csv=csv_dir,
    exclude_columns=[],
    pred_attr="DGWS",
    test_size=0.3,
    random_state=0,
    outdir=output_dir,
)


# In[3]:


## k-fold parameters
global classifier
optimized_param_dict = hyperparameter_optimization(
    x_train,
    y_train,
    model="rf",
    folds=10,
    n_iter=1,
    random_search=True,
    repeatedstratified=False,
)


# In[3]:


## model training
# n_estimators = optimized_param_dict["n_estimators"]
# max_depth = optimized_param_dict["max_depth"]
# max_features = optimized_param_dict["max_features"]
# min_samples_leaf = optimized_param_dict["min_samples_leaf"]
# min_samples_split = optimized_param_dict["min_samples_split"]
classifier = RandomForestClassifier(
    n_estimators=400,
    max_features=10,
    min_samples_leaf=20,
    min_samples_split=8,
    max_depth=13,
    max_samples=None,
    max_leaf_nodes=None,
    random_state=0,
    bootstrap=True,
    class_weight="balanced",
    n_jobs=-1,
    oob_score=True,
)
classifier = classifier.fit(x_train, y_train)


# In[4]:


## accuracy evaluation
y_train_pred = classifier.predict(x_train)
y_pred = classifier.predict(x_test)
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_pred)

column_labels = [
    np.array(["Predicted", "Predicted", "Predicted"]),
    np.array(["<16.3 mm/yr", "16.3-42.5 mm/yr", ">42.5 mm/yr"]),
]
index_labels = [
    np.array(["Actual", "Actual", "Actual"]),
    np.array(["<16.3 mm/yr", "16.3-42.5 mm/yr", ">42.5 mm/yr"]),
]
cm_df_train = pd.DataFrame(cm_train, columns=column_labels, index=index_labels)
cm_df_test = pd.DataFrame(cm_test, columns=column_labels, index=index_labels)
print(cm_df_train, "\n")
print(cm_df_test, "\n")

font = 11
label = np.array(["<16.3", "16.3-42.5", ">42.5"])
disp = ConfusionMatrixDisplay(cm_test, display_labels=label)
disp.plot(cmap="YlGn")
for labels in disp.text_.ravel():
    labels.set_fontsize(font)
disp.ax_.set_ylabel("True Class", fontsize=font)
disp.ax_.set_xlabel("Predicted Class", fontsize=font)
plot_name = 'accuracy.png'
plt.savefig((plot_name), dpi=400)
print("Test confusion matrix saved")

micro_precision = round(precision_score(y_test, y_pred, average="micro"), 2)
micro_recall = round(recall_score(y_test, y_pred, average="micro"), 2)
micro_f1 = round(f1_score(y_test, y_pred, average="micro"), 2)
print(micro_precision, "\n")
print(micro_recall, "\n")
print(micro_f1, "\n")

### important plot
predictor_dict = {
    0: "crop",
    1: "rain",
    2: "irrigation",
    3: "salt",
    4: "distance",
    5: "pop",
    6: "DGWS",
}
x_train_df = pd.DataFrame(x_train)
x_train_df = x_train_df.rename(columns=predictor_dict)
col_labels = np.array(x_train_df.columns)
importance = np.array(classifier.feature_importances_)
imp_dict = {"feature_names": col_labels, "feature_importance": importance}
imp_df = pd.DataFrame(imp_dict)
imp_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
plt.figure(figsize=(10, 8))
plt.rcParams["font.size"] = 14
print(imp_df["feature_names"])
sns.barplot(x=imp_df["feature_names"], y=imp_df["feature_importance"], color="black")
plt.xticks(rotation=90)
plt.ylabel("Variable Importance")
plt.xlabel("Variables")
plt.tight_layout()
print(importance)
# plt.savefig((accuracy_dir + '/' + predictor_imp_keyword + '_pred_importance.png'), dpi=600)
# print('Feature importance plot saved')


# In[5]:


## permutation importance
from sklearn.metrics import f1_score, make_scorer

predictor_cols = pd.read_csv(
    "/home/terry/working/groundwater/grace/deep_groundwater/figure/random_forest/project/RF/model/Predictors/X_test.csv"
).columns
f1_macro = make_scorer(f1_score, average="macro")
# Permutation importance on test set
result_test = permutation_importance(
    classifier,
    x_test,
    y_test,
    n_repeats=30,
    random_state=0,
    n_jobs=-1,
    scoring=f1_macro,
)

sorted_importances_idx = result_test.importances_mean.argsort()
importances = pd.DataFrame(
    result_test.importances[sorted_importances_idx].T,
    columns=predictor_cols[sorted_importances_idx],
)
plt.figure(figsize=(10, 8))
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importance (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Relative change in accuracy")
ax.figure.tight_layout()
print(np.mean(importances,axis=0))


# In[6]:


## shap
import shap

feature_names = [
    "crop",
    "rain",
    "irrigation",
    "salt",
    "distance",
    "pop",
    "AI",
    "building",
    "CGI",
    "clay",
    "soil_m",
    "tem",
]
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(x_test)
#shape_class3 = shap_values[:, :, 2]
shape_class3=np.abs(shap_values).mean(axis=2)
shap_importance = np.abs(shape_class3).mean(axis=0)
total_importance = shap_importance.sum()
contribution_pct = (shap_importance / total_importance) * 100
results_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Contribution_%": contribution_pct,
    }
)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, 6))
wedges, texts, autotexts = plt.pie(
    results_df["Contribution_%"],
    labels=results_df["Feature"],
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
)
plt.savefig("shap.pdf", format="pdf", dpi=600, bbox_inches="tight")
print(contribution_pct[0])


# In[11]:


shape_class3=np.abs(shap_values).mean(axis=2)
shape_class3=shap_values[:,:,2]
shap_importance = np.abs(shape_class3).mean(axis=0)
total_importance = shap_importance.sum()
contribution_pct = (shap_importance / total_importance) * 100
results_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Contribution_%": contribution_pct,
    }
)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, 6))
wedges, texts, autotexts = plt.pie(
    results_df["Contribution_%"],
    labels=results_df["Feature"],
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
)
print(shap_importance)
np.savetxt('shape.csv',shap_importance,delimiter=',',fmt='%.3f')


# ## print(result_test.importances_mean.sum())

# In[8]:


print(np.abs(importances).mean(axis=0))
#print(result_test)


# In[17]:


micro_precision = precision_score(y_test, y_pred, average=None)
micro_recall = recall_score(y_test, y_pred, average=None)
micro_f1 = f1_score(y_test, y_pred, average=None)
print(micro_precision, "\n")
print(micro_recall, "\n")
print(micro_f1, "\n")

