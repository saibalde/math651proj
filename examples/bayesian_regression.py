import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

try:
    os.makedirs("../figures/", exist_ok=True)
except:
    raise OSError

def compute_weight_posterior(prior_avg, prior_var, noise_std, x_data, y_data):
    m = prior_avg.shape[0]

    Phi = special.eval_legendre(np.arange(m)[:, np.newaxis], x_data[np.newaxis, :])

    post_var = np.linalg.inv(np.linalg.inv(prior_var) + 1.0 / noise_std**2 * np.matmul(Phi, Phi.T))
    post_avg = np.matmul(post_var, np.linalg.solve(prior_var, prior_avg) + 1.0 / noise_std**2 * np.matmul(Phi, y_data))

    return post_avg, post_var

def compute_function_dist_from_weight_dist(x_eval, weight_avg, weight_var, noise_std):
    m = weight_avg.shape[0]

    Phi = special.eval_legendre(np.arange(m)[:, np.newaxis], x_eval[np.newaxis, :])

    f_eval_avg = np.matmul(Phi.T, weight_avg)
    f_eval_std = np.sqrt((Phi * np.matmul(weight_var, Phi)).sum(axis=0) + noise_std**2)

    return f_eval_avg, f_eval_std

def f_true(x):
    return np.sin(np.pi * x)

def run_test(n_data, noise_std, m, prior_std, seed=None, figname=None):
    if seed is not None:
        np.random.seed(seed)

    # generate data
    x_data = 2.0 * np.random.rand(n_data) - 1.0

    f_data = f_true(x_data)
    y_data = f_data + noise_std * np.random.randn(*x_data.shape)

    # fix prior
    prior_avg = np.zeros(m, dtype=np.float)
    prior_var = prior_std**2 * np.eye(m)

    # compute posterior
    post_avg, post_var = compute_weight_posterior(prior_avg, prior_var, noise_std, x_data, y_data)

    # determine function evaluation points
    x_eval = np.linspace(-1.0, 1.0, 201)
    f_eval = f_true(x_eval)

    # determine function posterior
    f_prior_avg, f_prior_std = compute_function_dist_from_weight_dist(x_eval, prior_avg, prior_var, noise_std)
    f_post_avg, f_post_std = compute_function_dist_from_weight_dist(x_eval, post_avg, post_var, noise_std)

    # plot figure
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), squeeze=False)

    ax[0, 0].plot(x_eval, f_eval, color='b', linestyle='--', linewidth=2)
    ax[0, 0].plot(x_eval, f_prior_avg, color='r')
    ax[0, 0].fill_between(x_eval, f_prior_avg - 2.0 * f_prior_std, f_prior_avg + 2.0 * f_prior_std, color='r', alpha=0.2)
    ax[0, 0].set_xlabel(r'$x$', fontsize=14)
    ax[0, 0].set_ylabel(r'$y$', fontsize=14)

    ax[0, 1].plot(x_eval, f_eval, color='b', linestyle='--', linewidth=2)
    ax[0, 1].scatter(x_data, y_data, color='g')
    ax[0, 1].plot(x_eval, f_post_avg, color='r')
    ax[0, 1].fill_between(x_eval, f_post_avg - 2.0 * f_post_std, f_post_avg + 2.0 * f_post_std, color='r', alpha=0.2)
    ax[0, 1].set_ylim([-2.0, 2.0])
    ax[0, 1].set_xlabel(r'$x$', fontsize=14)
    ax[0, 1].set_ylabel(r'$y$', fontsize=14)

    plt.tight_layout()

    if figname is not None:
        fig.savefig('../figures/' + figname + '.pdf')
    else:
        fig.show()
    plt.close(fig)

if __name__ == '__main__':
    n_data, noise_std, m, prior_std, seed = 10, 0.5, 6, 4.0, 20200314
    run_test(n_data, noise_std, m, prior_std, seed, 'bayesian_regression_1')

    n_data, noise_std, m, prior_std, seed = 50, 0.5, 6, 4.0, 20200314
    run_test(n_data, noise_std, m, prior_std, seed, 'bayesian_regression_2')

    n_data, noise_std, m, prior_std, seed = 100, 0.5, 6, 4.0, 20200314
    run_test(n_data, noise_std, m, prior_std, seed, 'bayesian_regression_3')
