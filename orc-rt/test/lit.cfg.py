# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "ORC-RT"
config.test_format = lit.formats.ShTest()
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.orc_rt_obj_root, "test")
config.suffixes = [
    ".test"
]

llvm_config.with_environment(
    "PATH",
    os.path.join(config.orc_rt_obj_root, "tools", "orc-executor"),
    append_path=True)

llvm_config.use_default_substitutions()
