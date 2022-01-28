"""This file contains functions for discovering benchmark functions. It works
in a similar way to python's unittest library.
"""
import configparser
import importlib
import os
import pathlib
import re
import sys
import types


def discover_benchmark_modules(top_level_path):
    """Starting from the `top_level_path`, discover python files which contains
    benchmark functions. It looks for files with a specific prefix, which
    defaults to "benchmark_"
    """
    config = configparser.ConfigParser()
    config.read(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")
    )
    if "discovery" in config.sections():
        filename_prefix = config["discovery"]["filename_prefix"]
    else:
        filename_prefix = "benchmark_"
    if re.search(fr"{filename_prefix}.*.py$", top_level_path):
        # A specific python file so just include that.
        benchmark_files = [top_level_path]
    else:
        # A directory so recursively search for all python files.
        benchmark_files = pathlib.Path(
            top_level_path
        ).rglob(f"{filename_prefix}*.py")
    for benchmark_filename in benchmark_files:
        benchmark_abs_dir = os.path.abspath(os.path.dirname(benchmark_filename))
        sys.path.append(benchmark_abs_dir)
        module_file_name = os.path.basename(benchmark_filename)
        module_name = module_file_name.replace(".py", "")
        module = importlib.import_module(module_name)
        yield module
        sys.path.pop()


def get_benchmark_functions(module, benchmark_function_name=None):
    """Discover benchmark functions in python file. It looks for functions with
    a specific prefix, which defaults to "benchmark_".
    """
    config = configparser.ConfigParser()
    config.read(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")
    )
    if "discovery" in config.sections():
        function_prefix = config["discovery"].get("function_prefix")
    else:
        function_prefix = "benchmark_"

    module_functions = []
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if (
            isinstance(attribute, types.FunctionType)
            and attribute_name.startswith(function_prefix)
        ):
            module_functions.append(attribute)

    if benchmark_function_name:
        # If benchmark_function_name is present, just yield the corresponding
        # function and nothing else.
        for function in module_functions:
            if function.__name__ == benchmark_function_name:
                yield function
    else:
        # If benchmark_function_name is not present, yield all functions.
        for function in module_functions:
            yield function
