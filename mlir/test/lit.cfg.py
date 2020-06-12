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
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MLIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.toy', '.ll', '.tc']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(("%mlir_src_root", config.mlir_src_root))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir]
tools = [
    'mlir-opt',
    'mlir-tblgen',
    'mlir-translate',
    'mlir-edsc-builder-api-test',
]

# The following tools are optional
tools.extend([
    ToolSubst('toy-ch1', unresolved='ignore'),
    ToolSubst('toy-ch2', unresolved='ignore'),
    ToolSubst('toy-ch3', unresolved='ignore'),
    ToolSubst('toy-ch4', unresolved='ignore'),
    ToolSubst('toy-ch5', unresolved='ignore'),
    ToolSubst('%cuda_wrapper_library_dir', config.cuda_wrapper_library_dir, unresolved='ignore'),
    ToolSubst('%linalg_test_lib_dir', config.linalg_test_lib_dir, unresolved='ignore'),
    ToolSubst('%mlir_runner_utils_dir', config.mlir_runner_utils_dir, unresolved='ignore'),
    ToolSubst('%rocm_wrapper_library_dir', config.rocm_wrapper_library_dir, unresolved='ignore'),
    ToolSubst('%vulkan_wrapper_library_dir', config.vulkan_wrapper_library_dir, unresolved='ignore')
])

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope"
