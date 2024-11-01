"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, variance, p = mixture # mu is (K, d)

    # Calculate the normal distribution matrix
    norm_dist = np.exp(-1 / (2 * variance) * np.linalg.norm(X[:, None] - mu, axis=2) ** 2) / ((2 * np.pi * variance) ** (d / 2)) # X[:, None] adds a new axis to X, 
                                                                                                                                 # transforming it into a shape of (n, 1, d). 
                                                                                                                                 # Why? To enable broadcasting in the subtraction step.
                                                                                                                                 # axis = 2 specifies norm computation
                                                                                                                                 # Results in (n, K) matrix

    # Compute the posterior probabilities (soft counts)
    posterior_prob = (p * norm_dist) / np.sum(p * norm_dist, axis=1, keepdims=True) # axis = 1 sum across rows, keepdims to ensure shape is the same as input

    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(np.sum(p * norm_dist, axis=1)))

    return posterior_prob, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    K = post.shape[1]
    n_hat = np.sum(post, axis=0)
    mu_hat = (post.T @ X) / n_hat.reshape(-1, 1)
    p_hat = n_hat / n
    
    norms = np.linalg.norm(X[:, None] - mu_hat, ord=2, axis=2)**2 # ord = 2 specifies L2 norm
    variance_hat = np.sum(post * norms, axis=0) / (n_hat * d)

    return GaussianMixture(mu_hat, variance_hat, p_hat)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    n, d = X.shape
    old_log_likelihood = None
    new_log_likelihood = None
    while old_log_likelihood is None or (new_log_likelihood - old_log_likelihood > 1e-6 * np.abs(new_log_likelihood)):
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture) # Estabish new posteriors
        mixture = mstep(X, post) # Use new posteriors to recalibrate parameters (mu, var, p)

    return mixture, post, new_log_likelihood