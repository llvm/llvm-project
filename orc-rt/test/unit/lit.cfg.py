# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "ORC-RT-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.orc_rt_obj_root, "unittests")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")
