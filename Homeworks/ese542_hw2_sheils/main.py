import csv
import numpy as np
import scipy.stats as st

fname = "data_HW2.csv"


def computeMeanWithoutInbuilt(data_array):
    sum = 0.0
    for d in data_array:
        sum += d
    sample_mean = sum / len(data_array)
    return sample_mean


def computeVarianceWithoutInbuilt(data_array):
    sample_mean = computeMeanWithoutInbuilt(data_array)
    total_squared_deviation = 0.0
    for d in data_array:
        total_squared_deviation += (d - sample_mean) ** 2
    sample_variance = total_squared_deviation / len(data_array)
    return sample_variance


with open(fname) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    string_data = [r[0] for r in reader]
    data_array = np.asarray(string_data, dtype=np.float64, order="C")

    sample_mean_without_inbuilt = computeMeanWithoutInbuilt(data_array)
    sample_mean_with_inbuilt = np.mean(data_array)
    sample_mean_deviation = np.abs(sample_mean_without_inbuilt - sample_mean_with_inbuilt)

    sample_variance_without_inbuilt = computeVarianceWithoutInbuilt(data_array)
    sample_variance_with_inbuilt = np.nanvar(data_array)
    sample_variance_deviation = np.abs(sample_variance_without_inbuilt - sample_variance_with_inbuilt)

    ninety_percent_confidence_interval = st.t.interval(alpha=0.90, df=len(data_array) - 1, loc=np.mean(data_array),
                                                       scale=0.25)

    print("Mean computed without inbuilt functions    : " + str(sample_mean_without_inbuilt))
    print("Mean using inbuilt numpy function          : " + str(sample_mean_with_inbuilt))
    print("Deviation between both methods             : " + str(sample_mean_deviation))
    print("==========")
    print("Variance computed without inbuilt functions: " + str(sample_variance_without_inbuilt))
    print("Variance using inbuilt numpy function      : " + str(sample_variance_with_inbuilt))
    print("Deviation between both methods             : " + str(sample_variance_deviation))
    print("==========")
    print("90% Confidence Interval Bounds             : " + str(ninety_percent_confidence_interval))
