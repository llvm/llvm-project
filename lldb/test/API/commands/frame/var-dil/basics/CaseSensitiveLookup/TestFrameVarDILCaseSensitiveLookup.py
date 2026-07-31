"""
Test that DIL matches variables correctly for case-sensitive languages.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILCaseSensitiveLookup(TestBase):
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

        self.expect(
            "frame var GlobaLVaR",
            error=True,
            substrs=["use of undeclared identifier 'GlobaLVaR'"],
        )
        self.expect(
            "frame var GLOBALVAR",
            error=True,
            substrs=["use of undeclared identifier 'GLOBALVAR'"],
        )
        self.expect(
            "frame var globalvar",
            error=True,
            substrs=["use of undeclared identifier 'globalvar'"],
        )

        self.expect_var_path("testVariable", type="int", value="3")

        self.expect(
            "frame var TestVaRiable",
            error=True,
            substrs=["use of undeclared identifier 'TestVaRiable'"],
        )
        self.expect(
            "frame var testvariable",
            error=True,
            substrs=["use of undeclared identifier 'testvariable'"],
        )
        self.expect(
            "frame var TESTVARIABLE",
            error=True,
            substrs=["use of undeclared identifier 'TESTVARIABLE'"],
        )
