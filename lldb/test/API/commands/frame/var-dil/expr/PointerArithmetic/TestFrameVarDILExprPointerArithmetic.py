"""
Test DIL pointer arithmetic.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILExprPointerArithmetic(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_pointer_arithmetic(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect_var_path("+array", type="int *")
        self.expect_var_path("+array_ref", type="int *")
        self.expect_var_path("+p_int0", type="int *")
        self.expect(
            "frame var -- '-p_int0'",
            error=True,
            substrs=["invalid argument type 'int *' to unary expression"],
        )
