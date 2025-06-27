"""
Test DIL address calculation.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILGlobalVariableLookup(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def expect_var_path(self, expr, compare_to_framevar=False, value=None, type=None):
        value_dil = super().expect_var_path(expr, value=value, type=type)
        if compare_to_framevar:
            self.runCmd("settings set target.experimental.use-DIL false")
            value_frv = super().expect_var_path(expr, value=value, type=type)
            self.runCmd("settings set target.experimental.use-DIL true")
            self.assertEqual(value_dil.GetValue(), value_frv.GetValue())

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("&x", True, type="int *")
        self.expect_var_path("r", True, type="int &")
        self.expect_var_path("&r", True, type="int &*")
        self.expect_var_path("pr", True, type="int *&")
        self.expect_var_path("&pr", True, type="int *&*")
        self.expect_var_path("my_pr", True)
        self.expect_var_path("&my_pr", True, type="mypr *")
        self.expect_var_path("&globalVar", True, type="int *")
        self.expect_var_path("&s_str", True, type="const char **")
        self.expect_var_path("&argc", True, type="int *")
