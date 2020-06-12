"""
Test that lldb persistent variables works correctly.
"""



import lldb
from lldbsuite.test.lldbtest import *


class PersistentVariablesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_persistent_variables(self):
        """Test that lldb persistent variables works correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.c"))

        self.runCmd("expr int $i = i")
        self.expect_expr("$i == i", result_type="bool", result_value="true")
        self.expect_expr("$i + 1", result_type="int", result_value="6")
        self.expect_expr("$i + 3", result_type="int", result_value="8")
        self.expect_expr("$1 + $2", result_type="int", result_value="14")
        self.expect_expr("$3", result_type="int", result_value="14")
        self.expect_expr("$2", result_type="int", result_value="8")
        self.expect_expr("(int)-2", result_type="int", result_value="-2")
        self.expect_expr("$4", result_type="int", result_value="-2")
        self.expect_expr("$4 > (int)31", result_type="bool", result_value="false")
        self.expect_expr("(long)$4", result_type="long", result_value="-2")
