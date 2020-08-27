# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' integration test runner.

# name: The name of this integration test suite.
config.name = 'MLIR_INTEGRATION'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as integration test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where integration tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where integration tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'integration_test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%mlir_src_root', config.mlir_src_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the integration testsuite.
config.excludes = ['CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir]
tools = [
    'mlir-opt',
    'mlir-cpu-runner',
]

# The following tools are optional.
tools.extend([
    ToolSubst(
        '%mlir_integration_test_dir',
        config.mlir_integration_test_dir,
        unresolved='ignore'),
])

llvm_config.add_tool_substitutions(tools, tool_dirs)
