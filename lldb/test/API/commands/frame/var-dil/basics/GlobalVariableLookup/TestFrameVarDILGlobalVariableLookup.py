"""
Make sure 'frame var' using DIL parser/evaultor works for local variables.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time


class TestFrameVarDILGlobalVariableLookup(TestBase):
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
        self.expect_var_path("globalVar", type="int", value="-559038737")  # 0xDEADBEEF
        self.expect_var_path("globalPtr", type="int *")
        self.expect_var_path("globalRef", type="int &")
        self.expect_var_path("::globalVar", value="-559038737")
        self.expect_var_path("::globalPtr", type="int *")
        self.expect_var_path("::globalRef", type="int &")

        self.expect(
            "frame variable 'externGlobalVar'",
            error=True,
            substrs=["use of undeclared identifier"],
        )  # 0x00C0FFEE
        self.expect(
            "frame variable '::externGlobalVar'",
            error=True,
            substrs=["use of undeclared identifier"],
        )  # ["12648430"])

        self.expect_var_path("ns::globalVar", value="13")
        self.expect_var_path("ns::globalPtr", type="int *")
        self.expect_var_path("ns::globalRef", type="int &")
        self.expect_var_path("::ns::globalVar", value="13")
        self.expect_var_path("::ns::globalPtr", type="int *")
        self.expect_var_path("::ns::globalRef", type="int &")
