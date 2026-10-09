"""
Tests that bool types work
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPPBoolTestCase(TestBase):
    def test_with_run_command(self):
        """Test that bool types work in the expression parser"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// breakpoint 1", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr(
            "bool second_bool = my_bool; second_bool",
            result_type="bool",
            result_value="false",
        )
        self.expect_expr("my_bool = true", result_type="bool", result_value="true")
