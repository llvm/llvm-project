# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import sys

import lit.formats
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "lldb-unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where unit test binaries are located.
# test_exec_root: The root path where tests should be run.
config.test_source_root = os.path.join(config.lldb_obj_root, "unittests")
config.test_exec_root = config.test_source_root

# One of our unit tests dynamically links against python.dll, and on Windows
# it needs to be able to find it at runtime.  This is fine if Python is on your
# system PATH, but if it's not, then this unit test executable will fail to run.
# We can solve this by forcing the Python directory onto the system path here.
llvm_config.with_system_environment(
    [
        "HOME",
        "PATH",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
    ]
)
llvm_config.with_environment("PATH", os.path.dirname(sys.executable), append_path=True)

# Enable sanitizer runtime flags.
config.environment["ASAN_OPTIONS"] = "detect_stack_use_after_return=1"
config.environment["TSAN_OPTIONS"] = "halt_on_error=1"

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")
