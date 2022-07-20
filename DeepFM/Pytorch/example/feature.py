import os
import gzip
import shutil
import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


def feature_selection(data):
    print("\n\n2. feature selection ")

    sparse_features = data.columns.tolist()
    sparse_features.remove('click')
    sparse_features.remove('hour')
    dense_features = ['hour']

    print("sparse feature :", sparse_features)
    print("dense feature :", dense_features)
    print("target :", 'click')

    return data, sparse_features, dense_features

def feature_encoding(data, sparse_features, dense_features):

    print("\n\n3-1. feature encoding ")
    print("categorical value to numeric label")
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("numeric value Minmax scaling ")
    mms = MinMaxScaler(feature_range=(0, 1)) ### date 더 최근일 수록 더 큰 숫자가 입력됨 
    data[dense_features] = mms.fit_transform(data[dense_features])

    return data

def feature_format_deepfm(data, sparse_features, dense_features, embedding_dim):

    print(f"\n\n3-2. feature embedding - embedding size {embedding_dim}")
    spar_feat_list = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim) for i, feat in enumerate(sparse_features)]
    dense_feat_list = [DenseFeat(feat, 1, ) for feat in dense_features]
    fixlen_feature_columns = spar_feat_list + dense_feat_list

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    return dnn_feature_columns, linear_feature_columns, feature_names
