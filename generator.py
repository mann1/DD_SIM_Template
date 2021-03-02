import os, pickle
import numpy as np
import tensorflow as tf

def read_pickle(file_name):
    with (open(file_name, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects

class Generator(tf.keras.utils.Sequence):

    def __init__(self, DATASET_PATH, BATCH_SIZE=32):
        """ Initialize Generator object.

        Args
            DATASET_PATH           : Path to folder containing individual folders named by their class names
            BATCH_SIZE             : The size of the batches to generate.
        """

        self.batch_size = BATCH_SIZE
        self.load_data(DATASET_PATH)
        self.create_data_batches()
    
    def load_data(self, DATASET_PATH):
        cwd = os.getcwd()
        DATA_PATH = os.path.join(cwd, DATASET_PATH)

        if DATASET_PATH == 'datasets/train':
            data_file = os.path.join(DATA_PATH, "train_data.pickle")
            target_file = os.path.join(DATA_PATH, "train_target.pickle")
        elif DATASET_PATH == 'datasets/val':
            data_file = os.path.join(DATA_PATH, "val_data.pickle")
            target_file = os.path.join(DATA_PATH, "val_target.pickle")

        self.data = read_pickle(data_file)
        self.target = read_pickle(target_file)

        
        assert len(self.data) == len(self.target)

    def create_data_batches(self):

        # Divide data and target into groups of BATCH_SIZE
        self.data_batchs = [[self.data[x % len(self.data)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.data), self.batch_size)]
        self.target_batchs = [[self.target[x % len(self.target)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.target), self.batch_size)]

    
    def __len__(self):
        """
        Number of batches for each Epoch.
        """

        return len(self.data_batchs)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        if index >= len(self.data_batchs):
            index = index % len(self.data_batchs)
        data_batch = self.data_batchs[index]
        target_batch = self.target_batchs[index]

        return np.array(data_batch), np.array(target_batch)

if __name__ == "__main__":

    train_generator = Generator('datasets/train')
    val_generator = Generator('datasets/val')
    print(len(train_generator))
    print(len(val_generator))
    data_batch, target_batch = train_generator.__getitem__(0)
    print(data_batch.shape)
    print(target_batch.shape)
    