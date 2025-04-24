# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = "Offload-Unit-{}".format(config.offload_platform)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

config.environment = {"OFFLOAD_UNITTEST_PLATFORM": config.offload_platform}

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.library_dir, "unittests")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, ".unittests")
