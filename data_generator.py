import numpy as np
import tensorflow as tf 

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_list, labels, data_dir, batch_size=32, n_channels=3, n_classes=1, 
        shuffle=True, data_flag = 'train', dim= None, **augmentation_kwargs):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_flag = data_flag
        self.n_channels = n_channels
        self.augmentor= tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_kwargs)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_list))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of data for a batch
        data_list_temp = [self.data_list[k] for k in batch_indexes]


        # Generate data
        X, y= self.__data_generation(data_list_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype= np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)
        # Generate data
        for i, idx in enumerate(data_list_temp):
            # Store sample
            X[i,]= #load data
            y[i,] = self.labels[idx]
        
        if(self.data_flag=='train'):
            X, y= self.augmentor.flow(X,y=y,shuffle=True, batch_size=self.batch_size)[0]
        return X, y

