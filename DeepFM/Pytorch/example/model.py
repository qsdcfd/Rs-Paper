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



def modeling(linear_feature_columns, dnn_feature_columns,
             batch_size, num_epoch, val_ratio, test_rato, l2_decay_val, random_seed):
    
    print(f"\n\n5. Modeling")
    model = DeepFM(linear_feature_columns=linear_feature_columns,  
               dnn_feature_columns=dnn_feature_columns, 
               l2_reg_linear=l2_decay_val, l2_reg_embedding=l2_decay_val, l2_reg_dnn=l2_decay_val,
               dnn_dropout=0.5, 
               dnn_use_bn = True,
               dnn_hidden_units=(32, 16),
               task='binary',
               seed=random_seed, device=device)


    model.compile("adam", "binary_crossentropy", 
                metrics=["binary_crossentropy", "auc"], )


    return model


if __name__ == "__main__":
    batch_size = 1000
    num_epoch = 20
    val_ratio = 0.1
    test_rato = 0.1
    random_seed = 2022
    l2_decay_val = 1e-01
    embedding_dim = 5

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'


    data = data_load()
    target = ['click']

    data, sparse_features, dense_features = feature_selection(data)
    data = feature_encoding(data, sparse_features, dense_features)

    dnn_feature_columns, linear_feature_columns, feature_names = feature_format_deepfm(data, sparse_features, dense_features, embedding_dim)

    train, test, train_model_input, test_model_input = data_split(data, test_rato, 
                                                                  feature_names, random_seed)

    model = modeling(linear_feature_columns, dnn_feature_columns,
             batch_size, num_epoch, val_ratio, test_rato, l2_decay_val, random_seed)
    
    model.fit(train_model_input, train[target].values,
            batch_size=batch_size, epochs=num_epoch, verbose=2, validation_split=val_ratio)
    
    eval_test(model, test_model_input, batch_size)