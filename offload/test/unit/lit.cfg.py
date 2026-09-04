# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = "Offload-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []


def prepend_executable_path(path):
    old_path = config.environment.get("PATH")
    config.environment["PATH"] = (
        f"{path}{os.path.pathsep}{old_path}" if old_path else path
    )


# Add the tools bin dir to the search path so the JIT tests can
# find it. Prefer the tools from the configured LLVM install or
# bootstrapping build.
if config.bin_llvm_tools_dir:
    prepend_executable_path(config.bin_llvm_tools_dir)

# The CUDA plugin invokes 'ptxas' via the PATH when JIT-compiling PTX for
# NVPTX devices. Add the CUDA bin dir to the search path so the unit tests can
# find it, mirroring what the main lit.cfg does for the lit tests.
if config.cuda_path:
    prepend_executable_path(f"{config.cuda_path}{os.path.sep}bin")

# Windows doesn't have rpath so make sure runtime deps for the unittest
# executable, such as LLVMOffload.dll, are found.
if config.operating_system == "Windows" and config.library_dir:
    prepend_executable_path(config.library_dir)

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.unittest_dir
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, ".unittests")
