"""
Test DIL pointer arithmetic.
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

    def test_dereference(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("*p_int0", True, value="0")
        self.expect_var_path("*cp_int5", True, value="5")
        self.expect_var_path("*rcp_int0", True, type="const int *")
        self.expect_var_path("*offset_p", True, value="5")
        self.expect_var_path("*offset_pref", True, type="int *")
        self.expect_var_path("**pp_int0", value="0")
        self.expect_var_path("&**pp_int0", type="int *")
        self.expect_var_path("*array", value="0")
        self.expect(
            "frame var '&*p_null'",
            error=True,
            substrs=["doesn't have a valid address"],
        )
        self.expect(
            "frame var '&*p_void'",
            error=True,
            substrs=["dereference failed: (void *) p_void"],
        )
