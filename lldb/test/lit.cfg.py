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
