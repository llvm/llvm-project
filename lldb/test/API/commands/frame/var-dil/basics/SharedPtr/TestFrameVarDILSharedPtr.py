"""
Make sure 'frame var' using DIL parser/evaluator works for shared/weak  pointers.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILSharedPtr(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "Set a breakpoint here",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("ptr_node.__ptr_",
                    type="std::shared_ptr<NodeS>::element_type *")
        self.expect_var_path("ptr_node.__ptr_->value", value="1")
        self.expect_var_path("ptr_node.__ptr_->next.__ptr_->value", value="2")

        self.expect_var_path("ptr_int.__ptr_",
                    type="std::shared_ptr<int>::element_type *")
        self.expect_var_path("*ptr_int.__ptr_", value="1")
        self.expect_var_path("ptr_int_weak.__ptr_",
                    type="std::weak_ptr<int>::element_type *")
        self.expect_var_path("*ptr_int_weak.__ptr_", value="1")
