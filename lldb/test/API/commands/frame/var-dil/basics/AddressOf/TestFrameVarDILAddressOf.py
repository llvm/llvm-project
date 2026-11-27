"""
Test DIL address calculation.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILGlobalVariableLookup(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_var_path("&x", type="int *")
        self.expect_var_path("r", type="int &")
        self.expect_var_path("&r", type="int &*")
        self.expect_var_path("pr", type="int *&")
        self.expect_var_path("&pr", type="int *&*")
        self.expect_var_path("my_pr")
        self.expect_var_path("&my_pr", type="mypr *")
        self.expect_var_path("&globalVar", type="int *")
        self.expect_var_path("&s_str", type="const char **")
        self.expect_var_path("&argc", type="int *")

        # Check that '&' is not allowed in simple mode, but allowed in legacy mode
        frame = thread.GetFrameAtIndex(0)
        simple = frame.GetValueForVariablePathWithMode("&x", lldb.eDILModeSimple)
        legacy = frame.GetValueForVariablePathWithMode("&x", lldb.eDILModeLegacy)
        self.assertFailure(simple.GetError())
        self.assertSuccess(legacy.GetError())
