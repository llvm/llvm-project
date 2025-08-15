"""
Test that we properly print multiple types.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators


class TestTypeLookupAnonStruct(TestBase):
    def test_lookup_anon_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_var_path("unnamed_derived.y", value="2")
        self.expect_var_path("unnamed_derived.z", value="13")
        self.expect(
            'frame variable "derb.x"',
            error=True,
            substrs=['"x" is not a member of "(DerivedB) derb"']
        )
        self.expect(
            'frame variable "derb.y"',
            error=True,
            substrs=['"y" is not a member of "(DerivedB) derb"']
        )
        self.expect_var_path("derb.w", value="14")
        self.expect_var_path("derb.k", value="15")
        self.expect_var_path("derb.a.x", value="1")
        self.expect_var_path("derb.a.y", value="2")

        self.expect_var_path("multi1.m", value="16")
        self.expect_var_path("multi1.y", value="30")

        self.expect_var_path("multi2.i", value="42")
        self.expect_var_path("multi2.w", value="23")
        self.expect_var_path("multi2.a.x", value="1")
        self.expect_var_path("multi2.a.y", value="2")
        self.expect_var_path("multi2.y", value="2")
        self.expect_var_path("multi2.n", value="7")
