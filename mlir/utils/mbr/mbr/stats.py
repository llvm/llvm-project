"""This file contains functions related to interpreting measurement results
of benchmarks.
"""
import configparser
import numpy as np
import os


def has_enough_measurements(measurements):
    """Takes a list/numpy array of measurements and determines whether we have
    enough measurements to make a confident judgement of the performance. The
    criteria for determining whether we have enough measurements is as follows.
    1. Whether enough time, defaulting to 1 second, has passed.
    2. Whether we have a max number of measurements, defaulting to a billion.

    If 1. is true, 2. doesn't need to be true.
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.cfg"))
    if "stats" in config:
        stats_dict = {
            "max_number_of_measurements": int(
                float(config["stats"]["max_number_of_measurements"])
            ),
            "max_time_for_a_benchmark_ns": int(
                float(config["stats"]["max_time_for_a_benchmark_ns"])
            ),
        }
    else:
        stats_dict = {
            "max_number_of_measurements": 1e9,
            "max_time_for_a_benchmark_ns": 1e9,
        }
    return (
        np.sum(measurements) >= stats_dict["max_time_for_a_benchmark_ns"]
        or np.size(measurements) >= stats_dict["max_number_of_measurements"]
    )
