import pandas as pd
import numpy as np

import os
import gzip
import shutil
import glob

from sklearn.model_selection import train_test_split


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

! pip install deepctr-torch

def data_load():
    print("\n\n1. data load ")
    data_path = "/content/drive/MyDrive/Colab Notebooks/2022_recom_study/ctr_sample_dataset/abazu_dataset/"
    data = pd.read_csv(data_path + "avazu_sample_10.csv")
    display(data.head(3))
    print(data.columns)
    print(data.shape) 
    return data

def data_split(data, test_rato, feature_names, random_seed):
    print(f"\n\n4. data split (test ratio - {test_rato})")
    train, test = train_test_split(data, test_size=test_rato, random_state = random_seed)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    return train, test, train_model_input, test_model_input