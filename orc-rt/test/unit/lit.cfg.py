# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

config.name = "ORC-RT-Unit"
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")
config.test_exec_root = os.path.join(config.orc_rt_obj_root, "unittests")
config.test_source_root = config.test_exec_root
