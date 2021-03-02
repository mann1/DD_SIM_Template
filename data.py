import os
import pickle
import numpy as np
from sklearn.datasets import make_regression


def read_pickle(file_name):
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects

def generate_dataset():

    cwd = os.getcwd()
    DATASET_PATH = os.path.join(cwd, "datasets")
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    X, Y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
    data_file = os.path.join(DATASET_PATH, "data.pickle")
    pickle.dump(X, open(data_file, "wb"))
    target_file = os.path.join(DATASET_PATH, "target.pickle")
    pickle.dump(Y, open(target_file, "wb"))
    print(f"Downloaded and extracted at {DATASET_PATH}")

    return DATASET_PATH

def split_dataset(DATASET_PATH = '', train_data = 300, val_data = 50):
    # Specify path to the downloaded folder
    classes = 3

    # Creating train directory
    train_dir = os.path.join(DATASET_PATH, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Creating val directory
    val_dir = os.path.join(DATASET_PATH, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    data_file = os.path.join(DATASET_PATH, "data.pickle")
    target_file = os.path.join(DATASET_PATH, "target.pickle")

    X = read_pickle(data_file)
    Y = read_pickle(target_file)
        
    # Creating destination files (train and val)
    train_data_file = os.path.join(train_dir, "train_data.pickle")
    train_target_file = os.path.join(train_dir, "train_target.pickle")
        
    val_data_file = os.path.join(val_dir, "val_data.pickle")
    val_target_file = os.path.join(val_dir, "val_target.pickle")

    assert(train_data < len(X)), "Train_data size larger than the data size"
    X_train = X[:train_data]
    Y_train = Y[:train_data]

    if train_data+val_data < len(X):
        X_val = X[train_data:train_data+val_data]
        Y_val = Y[train_data:train_data+val_data]
    else:
        X_val = X[train_data:]
        Y_val = Y[train_data:]

    pickle.dump(X_train, open(train_data_file, "wb"))
    pickle.dump(Y_train, open(train_target_file, "wb"))

    pickle.dump(X_val, open(val_data_file, "wb"))
    pickle.dump(Y_val, open(val_target_file, "wb"))

def get_dataset_stats(DATASET_PATH = ''):
    """
        This utility gives the following stats for the dataset:
        feature average:

        NOTE: You should have enough memory to load complete dataset
    """
    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'val')

    train_data_file = os.path.join(train_dir, "train_data.pickle")
    train_target_file = os.path.join(train_dir, "train_target.pickle")

    val_data_file = os.path.join(val_dir, "val_data.pickle")
    val_target_file = os.path.join(val_dir, "val_target.pickle")

    X_train = read_pickle(train_data_file)
    Y_train = read_pickle(train_target_file)

    X_val = read_pickle(val_data_file)
    Y_val = read_pickle(val_target_file)


    assert len(X_train) == len(Y_train)
    assert len(X_val) == len(Y_val)

    print("feature samples: ",X_train[0])
    sum = 0
    for index_j in range(len(X_train[0])):
        for index_i in range(len(X_train)):
            sum += X_train[index_i][index_j]
        print(f"feature {index_j}. has average of  {sum/len(X_train)}.")
    print("target samples: ", Y_train[0])
    sum = 0
    for index_j in range(len(Y_train[0])):
        for index_i in range(len(Y_train)):
            sum += Y_train[index_i][index_j]
        print(f"target {index_j}. has average of  {sum/len(Y_train)}.")
    return train_dir, val_dir

if __name__ == "__main__":
    
    DATASET_PATH = generate_dataset()
    # Number of images required in train and val sets
    train_images = 700
    val_images = 200
    split_dataset(DATASET_PATH=DATASET_PATH, train_data = train_images, val_data = val_images)
    get_dataset_stats(DATASET_PATH=DATASET_PATH)
