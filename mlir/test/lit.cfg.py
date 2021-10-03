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
config.suffixes = ['.td', '.mlir', '.toy', '.ll', '.tc', '.py', '.yaml', '.test']

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
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt',
                   'lit.cfg.py', 'lit.site.cfg.py']

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
    'mlir-lsp-server',
    'mlir-capi-ir-test',
    'mlir-capi-pass-test',
    'mlir-cpu-runner',
    'mlir-linalg-ods-yaml-gen',
    'mlir-reduce',
]

# The following tools are optional
tools.extend([
    ToolSubst('toy-ch1', unresolved='ignore'),
    ToolSubst('toy-ch2', unresolved='ignore'),
    ToolSubst('toy-ch3', unresolved='ignore'),
    ToolSubst('toy-ch4', unresolved='ignore'),
    ToolSubst('toy-ch5', unresolved='ignore'),
    ToolSubst('%linalg_test_lib_dir', config.linalg_test_lib_dir, unresolved='ignore'),
    ToolSubst('%mlir_runner_utils_dir', config.mlir_runner_utils_dir, unresolved='ignore'),
    ToolSubst('%spirv_wrapper_library_dir', config.spirv_wrapper_library_dir, unresolved='ignore'),
    ToolSubst('%vulkan_wrapper_library_dir', config.vulkan_wrapper_library_dir, unresolved='ignore'),
    ToolSubst('%mlir_integration_test_dir', config.mlir_integration_test_dir, unresolved='ignore'),
])

python_executable = config.python_executable
# Python configuration with sanitizer requires some magic preloading. This will only work on clang/linux.
# TODO: detect Darwin/Windows situation (or mark these tests as unsupported on these platforms).
if "asan" in config.available_features and "Linux" in config.host_os:
  python_executable = f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {config.python_executable}"
tools.extend([
  ToolSubst('%PYTHON', python_executable, unresolved='ignore'),
])

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope --allow-unused-prefixes=false"


# LLVM can be configured with an empty default triple
# by passing ` -DLLVM_DEFAULT_TARGET_TRIPLE="" `.
# This is how LLVM filters tests that require the host target
# to be available for JIT tests.
if config.target_triple:
    config.available_features.add('default_triple')

# Add the python path for both the source and binary tree.
# Note that presently, the python sources come from the source tree and the
# binaries come from the build tree. This should be unified to the build tree
# by copying/linking sources to build.
if config.enable_bindings_python:
    llvm_config.with_environment('PYTHONPATH', [
        os.path.join(config.mlir_obj_root, 'python_packages', 'mlir_core'),
        os.path.join(config.mlir_obj_root, 'python_packages', 'mlir_test'),
    ], append_path=True)

if config.enable_assertions:
    config.available_features.add('asserts')
else:
    config.available_features.add('noasserts')
