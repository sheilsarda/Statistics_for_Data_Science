import numpy as np
import scipy.special as sp
import scipy.stats as st
from matplotlib import pyplot as plt
from time import sleep


def generate_mu():
    # Simulate True Mean of Normal Distribution
    mu_array = np.linspace(-4, 4, 1000)
    return mu_array


def generate_alpha():
    # Simulate Probability of Type 1 Errors
    alpha_error = np.linspace(0.005, 0.15, 1000)
    return alpha_error


def generate_n():
    # Simulate Various Sample Sizes
    max_n = 50
    n_array = np.linspace(1, max_n, max_n)
    return n_array


def generate_var():
    # Simulate Different Variances
    sigma_squared_array = np.linspace(0.001, 20, 1000)
    return sigma_squared_array


def compute_power(mu, z_alpha, n, sigma):
    # Power is probability of rejecting the null hypothesis
    # when it is wrong

    # Find acceptance and rejection regions in terms of sample means
    Xbar_lower_bound = -z_alpha * (sigma / n ** 0.5)
    Xbar_upper_bound = z_alpha * (sigma / n ** 0.5)

    # Find power in terms of Z value
    Z_left_tail = (Xbar_lower_bound - mu) / (sigma / n ** 0.5)
    Z_right_tail = (Xbar_upper_bound - mu) / (sigma / n ** 0.5)

    # Compute and return power
    power = abs(sp.ndtr(Z_left_tail)) + (1 - abs(sp.ndtr(Z_right_tail)))
    return power


def plot_power(scenario):
    if (scenario == 'vary_mu'):
        mu_array = generate_mu()
        z_alpha = 1.96  # two-tailed test value for alpha = 0.05
        n = 15  # number of observations
        sigma = 2  # standard deviation
        power_array = []
        for mu in mu_array:
            power = compute_power(mu, z_alpha, n, sigma)
            power_array.append(power)

        plt.plot(mu_array, power_array)
        plt.suptitle(
            r'Power vs Population Mean where $H_0 : \mu = 0, H_a : \mu \neq 0$' + '\n' +
            r'Distribution Parameters: $\alpha = 0.05, n = 15, \sigma^2 = 4$')
        plt.xlabel(r'Population Mean $\mu_0$')

    elif (scenario == 'vary_alpha'):
        alpha_array = generate_alpha()
        mu = 3.0  # population mean
        n = 15  # number of observations
        sigma = 2  # standard deviation
        power_array = []
        for alpha in alpha_array:
            z_alpha = abs(st.norm.ppf(alpha / 2.0))
            power = compute_power(mu, z_alpha, n, sigma)
            power_array.append(power)

        plt.plot(alpha_array, power_array)
        plt.suptitle(
            r'Power vs Significance Level where $H_0 : \mu = 0, H_a : \mu \neq 0$' + '\n' +
            r'Distribution Parameters: $\mu = 3, n = 15, \sigma^2 = 4$')
        plt.xlabel(r'Significance Level $\alpha$')

    elif (scenario == 'vary_n'):
        n_array = generate_n()
        mu = 3.0  # population mean
        z_alpha = 1.96  # two-tailed test value for alpha = 0.05
        sigma = 2  # standard deviation
        power_array = []
        for n in n_array:
            power = compute_power(mu, z_alpha, n, sigma)
            power_array.append(power)

        plt.plot(n_array, power_array)
        plt.suptitle(
            r'Power vs Sample Size where $H_0 : \mu = 0, H_a : \mu \neq 0$' + '\n' +
            r'Distribution Parameters: $\mu = 3, \alpha = 0.05, \sigma^2 = 4$')
        plt.xlabel(r'Sample Size $n$')

    elif (scenario == 'vary_sigma_squared'):
        sigma_squared_array = generate_var()
        z_alpha = 1.96  # two-tailed test value for alpha = 0.05
        n = 15  # number of observations
        mu = 3.0  # population mean
        power_array = []
        for sigma_squared in sigma_squared_array:
            sigma = sigma_squared ** 0.5
            power = compute_power(mu, z_alpha, n, sigma)
            power_array.append(power)

        plt.plot(sigma_squared_array, power_array)
        plt.suptitle(
            r'Power vs Variance where $H_0 : \mu = 0, H_a : \mu \neq 0$' + '\n' +
            r'Distribution Parameters: $\mu = 3, \alpha = 0.05, n = 15$')
        plt.xlabel(r'Variance $\sigma^2$')

    plt.ylabel(r'Power of Test (1 - $\beta$)')
    plt.show()


def main():
    scenario_array = ['vary_mu',
                      'vary_alpha',
                      'vary_n',
                      'vary_sigma_squared'
                      ]
    for scenario in scenario_array:
        plot_power(scenario)
    return


if __name__ == '__main__':
    main()
