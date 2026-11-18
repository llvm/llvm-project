"""
Test that variables of zero-size types without a location have a value.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCZeroSizeVariables(TestBase):
    # The MSVC ABI gives an empty C struct a nonzero size, and CodeView omits
    # variables that have no location.
    @skipIfWindows
    def test(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "take")
        self.expect_var_path("param", type="Empty", children=[])
        self.expect_var_path("local", type="Empty", children=[])
        self.expect_var_path("zero_len", type="int[0]", children=[])
        self.expect_var_path("x", type="int", value="41")
