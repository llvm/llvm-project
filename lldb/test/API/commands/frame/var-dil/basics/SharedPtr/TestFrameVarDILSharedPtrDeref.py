"""
Make sure 'frame var' using DIL parser/evaluator works for shared pointer deref.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILSharedPtrDeref(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "Set a breakpoint here",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("ptr_node->value", value="1")
        self.expect_var_path("ptr_node->next->value", value="2")
        self.expect_var_path("(*ptr_node).value", value="1")
        self.expect_var_path("(*(*ptr_node).next).value", value="2")
