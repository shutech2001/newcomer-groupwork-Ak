import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
#追加したimport
from sklearn.decomposition import TruncatedSVD
import random
import time

def check_args(args):
    if args.train is None:
        print(parser.print_help())
        exit(1)
    if args.test is not None and args.out is None:
        print(parser.print_help())
        exit(1)

def generate_input(df, window_radius=1):  #「window_radius*2+1」ごとに区切ってリスト
    _data = []
    for _, item in df.iterrows():
        seq = item.sequence
        length = len(seq)

        seq = ("_" * window_radius) + seq + ("_" * window_radius) #add spacer
        for resn in range(length):
            _in = list(seq[resn:resn+window_radius*2+1])
            _data.append(_in)
    return _data

def generate_label(df):
    label = []
    for _, item in df.iterrows():
        ss = item.label
        for resn, _label in enumerate(ss):
            label.append(int(_label))
    return np.array(label)

if __name__ == "__main__":

    # start = time.time()

    parser = argparse.ArgumentParser(description="example program")
    parser.add_argument("-train", help="path to training data (required)")
    parser.add_argument("-test", help="path to test data (optional)")
    parser.add_argument("-out", help="path to predicted information for test data (required only if --test is set)")
    parser.add_argument("--window_radius", type=int, default=10)
    args = parser.parse_args()

    check_args(args)

    ###### 1. data preparation ######

    # read csv files
    train_val_df = pd.read_csv(args.train)  #訓練用データ
    test_df      = pd.read_csv(args.test) if (args.test is not None) else None  #教師用データ

    # split into train dataset and validation dataset (not train-test splitting)
    train_df, val_df = train_test_split(train_val_df, random_state=0)  #訓練用データを分割
    del train_val_df

    # encode an amino acids sequence into a numerical vector
    # MUST use the same transformer for all data without refit

    # extract subsequence
    window_radius = args.window_radius
    train_data_ = generate_input(train_df, window_radius)
    onehot = OneHotEncoder().fit(train_data_)
    n = 460
    pca = TruncatedSVD(n_components=n)
    train_data  = onehot.transform(train_data_)
    del train_data_
    y_train = generate_label(train_df)
    del train_df
    pca.fit(train_data)
    X_train = pca.transform(train_data)

    val_data_   = generate_input(val_df, window_radius)
    y_val   = generate_label(val_df)
    del val_df
    val_data    = onehot.transform(val_data_)
    del val_data_
    X_val = pca.transform(val_data)

    test_data_  = generate_input(test_df, window_radius) if (test_df is not None) else None
    test_data   = onehot.transform(test_data_) if (test_data_ is not None) else None
    X_test = pca.transform(test_data) if (test_data_ is not None) else None

    # extract label information
    # Note: NO LABEL INFORMATION for test dataset

    # test_label = None

    ###### 2. model construction (w/ training dataset) ######

    lr = LogisticRegression(random_state=1, max_iter=300)
    model = lr.fit(X_train, y_train)

    ###### 3. model evaluation (w/ validation dataset) ######

    score = model.score(X_val, y_val)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # print('window_radius: %d'%(window_radius))
    # print('n: %d'%(n))
    print('Q2 accuracy: %.4f'%(score))
    print('AUC: %.4f'%(auc))

    ###### 4. prediction for test dataset ######

    if (test_df is not None) and (X_test is not None):  #これ以降で出力の仕方を設定

        predicted = model.predict_proba(X_test)[:, 1]

        sequence_id_list    = []
        residue_number_list = []
        for _, item in test_df.iterrows():
            sequence_id = item.sequence_id
            sequence    = item.sequence
            for i, aa in enumerate(sequence):
                sequence_id_list.append(sequence_id)
                residue_number_list.append(i+1) #0-origin to 1-origin

        predicted_df = pd.DataFrame.from_dict({  #「from_dict」でdataframeにして出力
            "sequence_id": sequence_id_list,
            "residue_number": residue_number_list,
            "predicted_value": predicted,
            })
        predicted_df.to_csv(args.out, index=None)

    # elapsed_time = time.time() - start
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
