import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))
    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch
        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y=y.reshape(x.shape[0],1)
        it = x.shape[0]/batch_size
        if it - int(it) != 0:
            it = int(it) + 1
        else:
            it = int(it)
        for each_time in range(epochs):
            loss=0
            idxs = np.arange(x.shape[0])
            np.random.shuffle(idxs)
            x = x[idxs]
            y = y[idxs]
            for i in range(it):
                x_iter = x[i*batch_size:(i+1)*batch_size]
                y_iter = y[i*batch_size:(i+1)*batch_size]
                y_pred = self.forward(x_iter)
                error = y_iter - y_pred
                loss += np.sum(error ** 2)
                diff = ((-2/x_iter.shape[0])*sum(x_iter*error)).reshape(self.num_features,1)
                self.W = optim.update(self.W, diff, lr)
            loss /= x.shape[0]
            final_loss=loss
        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        # ============================================================
        return y_predicted
