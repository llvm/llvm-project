"""This file contains the main function that's called by the CLI of the library.
"""

import os
import sys
import time

import numpy as np

from discovery import discover_benchmark_modules, get_benchmark_functions
from stats import has_enough_measurements


def main(top_level_path, stop_on_error):
    """Top level function called when the CLI is invoked.
    """
    if "::" in top_level_path:
        if top_level_path.count("::") > 1:
            raise AssertionError(f"Invalid path {top_level_path}")
        top_level_path, benchmark_function_name = top_level_path.split("::")
    else:
        benchmark_function_name = None

    if not os.path.exists(top_level_path):
        raise AssertionError(
            f"The top-level path {top_level_path} doesn't exist"
        )

    modules = [module for module in discover_benchmark_modules(top_level_path)]
    benchmark_dicts = []
    for module in modules:
        benchmark_functions = [
            function for function in
            get_benchmark_functions(module, benchmark_function_name)
        ]
        for benchmark_function in benchmark_functions:
            try:
                compiler, runner = benchmark_function()
            except (TypeError, ValueError):
                error_message = (
                    f"benchmark_function '{benchmark_function.__name__}'"
                    f" must return a two tuple value (compiler, runner)."
                )
                if stop_on_error is False:
                    print(error_message, file=sys.stderr)
                    continue
                else:
                    raise AssertionError(error_message)
            measurements_ns = np.array([])
            if compiler:
                start_compile_time_s = time.time()
                try:
                    compiled_callable = compiler()
                except Exception as e:
                    error_message = (
                        f"Compilation of {benchmark_function.__name__} failed"
                        f" because of {e}"
                    )
                    if stop_on_error is False:
                        print(error_message, file=sys.stderr)
                        continue
                    else:
                        raise AssertionError(error_message)
                total_compile_time_s = time.time() - start_compile_time_s
                runner_args = (compiled_callable,)
            else:
                total_compile_time_s = 0
                runner_args = ()
            while not has_enough_measurements(measurements_ns):
                try:
                    measurement_ns = runner(*runner_args)
                except Exception as e:
                    error_message = (
                        f"Runner of {benchmark_function.__name__} failed"
                        f" because of {e}"
                    )
                    if stop_on_error is False:
                        print(error_message, file=sys.stderr)
                        # Recover from runner error by breaking out of this loop
                        # and continuing forward.
                        break
                    else:
                        raise AssertionError(error_message)
                if not isinstance(measurement_ns, int):
                    error_message = (
                        f"Expected benchmark runner function"
                        f" to return an int, got {measurement_ns}"
                    )
                    if stop_on_error is False:
                        print(error_message, file=sys.stderr)
                        continue
                    else:
                        raise AssertionError(error_message)
                measurements_ns = np.append(measurements_ns, measurement_ns)

            if len(measurements_ns) > 0:
                measurements_s = [t * 1e-9 for t in measurements_ns]
                benchmark_identifier = ":".join([
                    module.__name__,
                    benchmark_function.__name__
                ])
                benchmark_dicts.append(
                    {
                        "name": benchmark_identifier,
                        "compile_time": total_compile_time_s,
                        "execution_time": list(measurements_s),
                    }
                )

    return benchmark_dicts
