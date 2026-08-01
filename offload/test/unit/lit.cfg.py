# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = "Offload-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# Add the tools bin dir to the search path so the JIT tests can
# find it. Prefer the tools from the configured LLVM install or
# bootstrapping build.
if config.bin_llvm_tools_dir:
    old_path = config.environment.get("PATH")
    config.environment["PATH"] = (
        f"{config.bin_llvm_tools_dir}{os.path.pathsep}{old_path}"
        if old_path
        else config.bin_llvm_tools_dir
    )

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.unittest_dir
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, ".unittests")
