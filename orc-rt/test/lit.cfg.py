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


# name: The name of this test suite.
config.name = "ORCRT"

# TODO: Consolidate the logic for turning on the internal shell by default for all LLVM test suites.
# See https://github.com/llvm/llvm-project/issues/106636 for more details.
#
# We prefer the lit internal shell which provides a better user experience on failures
# unless the user explicitly disables it with LIT_USE_INTERNAL_SHELL=0 env var.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [
    ".ll",
    ".test",
    ".c",
]
# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.orcrt_obj_root, "test")
llvm_config.with_environment(
    "PATH",
    os.path.join(config.orcrt_obj_root, "tools", "orc-executor"),
    append_path=True)
config.substitutions.append(("%PATH%", config.environment["PATH"]))
# config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%llvm_src_root", config.llvm_src_root))
# config.substitutions.append(("%host_cxx", config.host_cxx))
# config.substitutions.append(("%host_cc", config.host_cc))
if config.llvm_rt_tools_dir:
    config.llvm_tools_dir = config.llvm_rt_tools_dir
llvm_config.use_default_substitutions()


# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py.in",
]
