import tensorflow as tf
from tensorflow.keras import backend as K

class PreRecF1score():
    def __init__(self, num_classes, labels_list):
    '''Arguments:
          num_classes: Number of classes
          label_lists: list of names of classes
    '''
        self.num_classes = num_classes
        self.labels_list = labels_list
        self.metrics = []

    @tf.function
    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @tf.function
    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @tf.function
    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def precision_classwise(self, index):
        def precision_call(y_true, y_pred):
            true = y_true[:,index]
            pred = y_pred[:,index]
            return self.precision_m(true, pred)
        precision_call.__name__ = 'pre_' + self.labels_list[index]
        return precision_call

    
    def recall_classwise(self, index):
        def recall_call(y_true, y_pred):
            true = y_true[:,index]
            pred = y_pred[:,index]
            return self.recall_m(true, pred)
        recall_call.__name__ = 'rec_' + self.labels_list[index]
        return recall_call

    
    def f1_classwise(self, index):
        def f1_call(y_true, y_pred):
            true = y_true[:,index]
            pred = y_pred[:,index]
            pre = self.precision_m(true, pred)
            rec = self.recall_m(true, pred)
            return 2*((pre*rec)/(pre+rec+K.epsilon()))
        f1_call.__name__ = 'f1_' + self.labels_list[index]
        return f1_call

    def metrics_call(self, mode='avg'):
    '''Arguments:
            mode:  One of {'avg', 'full'}.
            Default is 'avg'.
            - 'avg' for average Precision, Recall & F1-Score
            - 'full' for class-wise descriptive Precision, Recall & F1-Score
    '''
        if(mode=='full'):    
            self.metrics = [self.precision_classwise(i) for i in range(self.num_classes)]
            self.metrics += [self.recall_classwise(i) for i in range(self.num_classes)]
            self.metrics += [self.f1_classwise(i) for i in range(self.num_classes)]
        elif (mode=='avg'):
            self.metrics= []
        self.metrics.append(self.precision_m)
        self.metrics.append(self.recall_m)
        self.metrics.append(self.f1_m)
        return self.metrics