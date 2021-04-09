import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch
        
        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0],1)
        it = x.shape[0]/batch_size
        if it - int(it) != 0:
            it = int(it) + 1
        else:
            it = int(it)
        for each_time in range(epochs):
            loss = 0
            idxs = np.arange(x.shape[0])
            np.random.shuffle(idxs)
            x = x[idxs]
            y = y[idxs]
            for i in range(it):
                x_iter = x[i*batch_size:(i+1)*batch_size]
                y_iter = y[i*batch_size:(i+1)*batch_size]
                pred_y = self._sigmoid(np.dot(x_iter,self.W)).reshape(x_iter.shape[0],1)
                loss -= np.sum(y_iter * np.log(pred_y + epsilon) + (1 - y_iter) * np.log(1 - pred_y + epsilon))
                diff = sum(x_iter*(pred_y - y_iter) / x_iter.shape[0]).reshape(self.num_features,1)
                self.W = optim.update(self.W,diff,lr)
            loss = loss / x.shape[0]
            
            
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        y_pred = self._sigmoid(np.dot(x,self.W))
        for idx in range(len(y_pred)):
            if y_pred[idx] < threshold:
                y_pred[idx] = 0
            else:
                y_pred[idx] = 1
        y_predicted = y_pred

        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None
        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.e ** (-x))
        # ============================================================
        return sigmoid
