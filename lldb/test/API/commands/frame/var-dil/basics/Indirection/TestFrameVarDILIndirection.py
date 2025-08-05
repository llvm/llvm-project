"""
Make sure 'frame var' using DIL parser/evaluator works for indirection.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time


class TestFrameVarDILIndirection(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("*p", value="1")
        self.expect_var_path("p", type="int *")
        self.expect_var_path("*my_p", value="1")
        self.expect_var_path("my_p", type="myp")
        self.expect_var_path("*my_pr", type="int *")
        self.expect_var_path("my_pr", type="mypr")

        self.expect(
            "frame variable '*1'",
            error=True,
            substrs=["Unexpected token: <'1' (numeric_constant)>"],
        )
        self.expect(
            "frame variable '*val'",
            error=True,
            substrs=[
                "dereference failed: not a pointer, reference or array type: (int) val"
            ],
        )
