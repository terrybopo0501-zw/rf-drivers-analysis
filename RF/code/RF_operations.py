import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
import dask.dataframe as ddf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from lightgbm import LGBMClassifier


def split_train_test_ratio(predictor_csv, exclude_columns=[], pred_attr='DGWS', test_size=0.3, random_state=0,
                           outdir=None, verbose=True):
  
    input_df = pd.read_csv(predictor_csv)
    predictor_name_dict = {'crop': 'crop', 'rain': 'rain',
                           'irrigation': 'irrigation', 'salt': 'salt', 'distance': 'distance',
                           'pop': 'pop','AI':'AI', 'building':'building','CGI':'CGI','clay':'clay','soil_m':'soil_m','tem':'tem', 'DGWS': 'DGWS'}

    input_df = input_df.rename(columns=predictor_name_dict)
    drop_columns = exclude_columns + [pred_attr]
    x = input_df.drop(columns=drop_columns)
    y = input_df[pred_attr]
    if verbose:
        print('Dropping Columns-', exclude_columns)
        print('Predictors:', x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        shuffle=True, stratify=y)

    if outdir:
        x_train_df = pd.DataFrame(x_train)
        x_train_df.to_csv(os.path.join(outdir, 'X_train.csv'), index=False)

        y_train_df = pd.DataFrame(y_train)
        y_train_df.to_csv(os.path.join(outdir, 'y_train.csv'), index=False)

        x_test_df = pd.DataFrame(x_test)
        x_test_df.to_csv(os.path.join(outdir, 'X_test.csv'), index=False)

        y_test_df = pd.DataFrame(y_test)
        y_test_df.to_csv(os.path.join(outdir, 'y_test.csv'), index=False)

    return x_train, x_test, y_train, y_test, predictor_name_dict


def hyperparameter_optimization(x_train, y_train, model='rf', folds=10, n_iter=70, random_search=True,
                                repeatedstratified=False):
   
    global classifier
    param_dict = {'rf':
                      {'n_estimators': [100, 200, 300, 400, 500],
                       'max_depth': [8, 12, 13, 14],
                       'max_features': [6, 7, 9, 10],
                       'min_samples_leaf': [5e-4, 1e-5, 1e-3, 6, 12, 20, 25],
                       'min_samples_split': [6, 7, 8, 10]
                       },
                  'gbdt':
                      {'num_leaves': [31, 63, 100, 200],
                       'max_depth': [10, 12, 15, 20],
                       'learning_rate': [0.01, 0.05],
                       'n_estimators': [100, 200, 300],
                       'subsample': [1, 0.9],
                       'min_child_samples': [20, 25, 30, 35, 50]}
                  }

    print('Classifier Name:', model)
    pprint(param_dict[model])

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=0, n_jobs=-1, bootstrap=True, oob_score=True,
                                            class_weight='balanced')
    elif model == 'gbdt':
        classifier = LGBMClassifier(boosting_type='gbdt', objective='multiclass', class_weight='balanced',
                                    importance_type='split', random_state=0, n_jobs=-1)


    if repeatedstratified:
        kfold = RepeatedStratifiedKFold(n_splits=folds, n_repeats=10, random_state=0)
    else:
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    if random_search:
        CV = RandomizedSearchCV(estimator=classifier, param_distributions=param_dict[model], n_iter=n_iter,
                                cv=kfold, verbose=1, random_state=0, n_jobs=-1,
                                scoring='f1_macro', refit=True, return_train_score=True)
    else:
        CV = GridSearchCV(estimator=classifier, param_grid=param_dict[model], cv=kfold, verbose=1, n_jobs=-1,
                          scoring='f1_macro', refit=True, return_train_score=True)

    CV.fit(x_train, y_train)

    print('\n')
    print('best parameters for macro f1 value ', '\n')
    pprint(CV.best_params_)
    print('\n')
    print('mean_test_macro_f1_score', round(CV.cv_results_['mean_test_score'][CV.best_index_], 2))
    print('mean_train_macro_f1_score', round(CV.cv_results_['mean_train_score'][CV.best_index_], 2))

    if model == 'rf':
        optimized_param_dict = {'n_estimators': CV.best_params_['n_estimators'],
                                 'max_depth': CV.best_params_['max_depth'],
                                 'max_features': CV.best_params_['max_features'],
                                 'min_samples_leaf': CV.best_params_['min_samples_leaf'],
                                'min_samples_split': CV.best_params_['min_samples_split']
                                }

        return optimized_param_dict

    elif model == 'gbdt':
        optimized_param_dict = {'num_leaves': CV.best_params_['num_leaves'],
                                'max_depth': CV.best_params_['max_depth'],
                                'learning_rate': CV.best_params_['learning_rate'],
                                'n_estimators': CV.best_params_['n_estimators'],
                                'subsample': CV.best_params_['subsample'],
                                'min_child_samples': CV.best_params_['min_child_samples']}

        return optimized_param_dict
        
