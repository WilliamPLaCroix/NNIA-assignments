# Parts of the skeleton of this file is adapted from a UdS HLCV Course Assignment SS2021
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

"""
DO NOT CHANGE ANYTHING INSIDE THE BLOCK QUOTES
"""

# constants, do not change
torch.manual_seed(35)
np.random.seed(23)


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        """
        Q1.2
        TODO: implement the function to reshape images into a vector of 'new_size', change the return accordingly

        :param img: original image Tensor of shape (C,H,W) or (H,W,C)
        :return: flattened image Tensor in new_size
        """
        return torch.flatten(img)


def get_cifar10_dataset(val_size=5000, batch_size=128):
    """
    Load and transform the CIFAR10 dataset. Make Validation set. Create dataloaders for
    train, test, validation sets.

    :param val_size: size of the validation partition
    :param batch_size: number of samples in a batch
    :return:
    """

    """
    Q1.2
    Compose the instances of transpose functions into a vector. For normalization, use mean=0.5 and sd=0.5.
    e.g. transform = transforms.Compose([..., ..., ...])
    
    """
    # TODO: START YOUR CODE HERE: Compose the transform function

    transform = transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5),
        ReshapeTransform(32*32*3)])  # replace None with your code here

    # TODO: Load the train_set and test_set from PyTorch; replace the None to your solution

    train_set = torchvision.datasets.CIFAR10(root="cifar-10-batches-py", train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root="cifar-10-batches-py", train=False, transform=transform, download=True)
    """
    DO NOT CHANGE THE CODE BELOW
    """
    classes = train_set.classes

    """
    Q1.3 
    Split our train data into train_set and val_set.
    Use val_size (10% of the data) for validation.
    Use the DataLoader module from PyTorch for this part.
    """
    # TODO: Split your data and define train_loader, test_loader, val_loader; replace the None with your solution
    train_size = len(train_set) - val_size
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size)

    """END OF YOUR CODE -- DO NOT CHANGE THE RETURN"""
    return train_loader, test_loader, val_loader, classes


"""
Q2 - Q3 Tasks are inside the class NeuralNetowrkModel below. 
You're not allowed to use PyTorch built-in functions beyond this point.
Only Numpy and standard library built-in operations are allowed.

DO NOT CHANGE THE EXISTING CODE UNLESS SPECIFIED
"""


class NeuralNetworkModel:
    """
    A two-layer fully-connected neural network. The model has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.

    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # DO NOT CHANGE ANYTHING IN HERE

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """

        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength (lambda).

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = 0.
        """
        Q2.2
        Perform the forward pass, computing the class probabilities for the 
        input. Store the result in the scores variable, which should be an array  
        of shape (N, C).                                                        
        """
        # TODO: START OF YOUR CODE BELOW Eq 3-7

        # N == number of samples
        # D == number of features or dimensions per sample
        # Eq 3:

        # Eq 4:

        # Eq 5:

        # Eq 6:

        # Eq 7:


        """
        DO NOT TOUCH THE CODE BELOW
        """
        try:
            assert np.all(np.isclose(np.sum(a_3, axis=1), 1.0))  # check that scores for each sample add up to 1
        except AssertionError:
            print(f'scores after softmax: \n{a_3}')
            print(f'sum of scores for all class: {np.sum(a_3, axis=1)}')

        scores = a_3

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = 0.

        """
        Q2.2 Continued
        TODO: Finish the forward pass, and compute the loss. This should include
        both the data loss and L2 regularization for W1 and W2. Store the result 
        in the variable loss, which should be a scalar. Use the Softmax classifier loss.                                                          
        """
        # TODO: START OF YOUR CODE BELOW Eq 11-13
        # Implement the loss for softmax output layer

        # Eq 11
        
        # Eq 12
        
        # Eq 13

        """
        DO NOT TOUCH THE CODE BELOW
        """
        # Backward pass: compute gradients
        grads = {}
        """
        Q3.2: Compute the backward pass, computing the derivatives of the weights and biases. 
        Store the results in the grads dictionary (defined above). 
        
        For example, grads['W1'] should store the gradient on W1, and be a matrix of same size.
        """
        # TODO: START OF YOUR CODE:Backpropagation Eq 16, 18-23

        # Eq 16:


        # Eq 18, 19


        # Eq 20:

        grads['W2'] = None  # change None to your gradient wrt W2

        # Eq 21-23 correspond to the derivatives in Q4 of Q 2.3.1. which you have to come up with
        # Eq 21: gradients wrt to b2

        grads['b2'] = None  # change None to your gradient wrt b2

        # Eq 22: gradients wrt to W1

        grads['W1'] = None  # change None to your gradient wrt W1

        # Eq 23: gradient wrt to b1
        grads['b1'] = None  # change None to your gradient wrt b1

        """END OF YOUR CODE: DO NOT CHANGE THE RETURN*****"""

        return loss, grads

