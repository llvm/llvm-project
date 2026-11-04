"""
Make sure 'frame var' using DIL parser/evaluator works for bit fields.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time


class TestFrameVarDILBitField(TestBase):
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
        self.expect_var_path("bf.a", value="1023")
        self.expect_var_path("bf.b", value="9")
        self.expect_var_path("bf.c", value="false")
        self.expect_var_path("bf.d", value="true")

        self.expect_var_path("abf.a", value="1023")
        self.expect_var_path("abf.b", value="'\\x0f'")
        self.expect_var_path("abf.c", value="3")

        # Perform an operation to ensure we actually read the value.
        # Address-of is not allowed for bit-fields.
        self.expect(
            "frame variable '&bf.a'",
            error=True,
            substrs=["'bf.a' doesn't have a valid address"],
        )
