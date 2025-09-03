# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

# This is the top level configuration. Most of these configuration options will
# be overriden by individual lit configuration files in the test
# subdirectories. Anything configured here will *not* be loaded when pointing
# lit at on of the subdirectories.

config.name = "lldb"
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.lldb_obj_root, "test")

# We prefer the lit internal shell which provides a better user experience on
# failures and is faster unless the user explicitly disables it with
# LIT_USE_INTERNAL_SHELL=0 env var.

use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

if use_lit_shell:
    os.environ["LIT_USE_INTERNAL_SHELL"] = "1"
