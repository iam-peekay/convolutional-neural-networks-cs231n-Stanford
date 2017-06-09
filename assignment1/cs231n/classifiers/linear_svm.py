import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # gradient with respect to wyi: count the number of classes that didn't 
        # meet the desired margin (and hence contributed to the loss function) 
        # and then the data vector xixi scaled by this number is the gradient.
        
        # Whenever you compute loss vector for x[i], ith training example and 
        # get some nonzero loss, that means you should move your weight vector 
        # for the incorrect class (j != y[i]) away by x[i], and at the same time, 
        # move the weights or hyperplane for the correct class (j==y[i]) near x[i]. 
        # By parallelogram law,  w + x lies in between w and x. So this way w[y[i]] 
        # tries to come nearer to x[i] each time it finds loss>0. 
        # Thus, dW[:,y[i]] += -X[i] and dW[:,j] += X[i] is done in the loop, 
        # but while update, we will do in direction of decreasing gradient, 
        # so we are essentially adding X[i] to correct class weights and 
        # going away by X[i] from weights that miss classify.
        
        dW[:, y[i]] += -X[i]
        dW[:, j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W) #(N, C)
  
  # get the correct class scores by indexing (within scores) on the value of the correct class in y
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) # (N, 1)
  
  # calculate margin for all class scores
  margins = np.maximum(0, scores - correct_class_scores + 1)
  # set the margin for the correct class to 0 
  margins[range(num_train), list(y)] = 0
 
  loss = np.sum(margins) / num_train
  
  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  coeff_matrix = np.zeros((num_train, num_classes))
  # create a binary index of anything which has margin over 0
  coeff_matrix[margins > 0] = 1
  # zero out for the actual class values since we don't want to account for them
  coeff_matrix[range(num_train), list(y)] = 0
  # a sum over j != y_i
  coeff_matrix[range(num_train), list(y)] = -np.sum(coeff_matrix, axis=1)
  
  dW = (X.T).dot(coeff_matrix)
  dW = dW/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
