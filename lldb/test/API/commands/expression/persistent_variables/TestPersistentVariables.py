"""
Test that lldb persistent variables works correctly.
"""


import lldb
from lldbsuite.test.lldbtest import *


class PersistentVariablesTestCase(TestBase):
    def test_persistent_variables(self):
        """Test that lldb persistent variables works correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

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

        # Try assigning an existing persistent veriable with a numeric name.
        self.expect(
            "expr int $2 = 1234",
            error=True,
            substrs=[
                "Error [IRForTarget]: Names starting with $0, $1, ... are reserved for use as result names"
            ],
        )
        # $2 should still have its original value.
        self.expect_expr("$2", result_type="int", result_value="8")

        # Try assigning an non-existing persistent veriable with a numeric name.
        self.expect(
            "expr int $200 = 3",
            error=True,
            substrs=[
                "Error [IRForTarget]: Names starting with $0, $1, ... are reserved for use as result names"
            ],
        )
        # Test that $200 wasn't created by the previous expression.
        self.expect(
            "expr $200", error=True, substrs=["use of undeclared identifier '$200'"]
        )

        # Try redeclaring the persistent variable with the same type.
        # This should be rejected as we treat them as if they are globals.
        self.expect(
            "expr int $i = 123",
            error=True,
            substrs=["error: redefinition of persistent variable '$i'"],
        )
        self.expect_expr("$i", result_type="int", result_value="5")

        # Try redeclaring the persistent variable with another type. Should
        # also be rejected.
        self.expect(
            "expr long $i = 123",
            error=True,
            substrs=["error: redefinition of persistent variable '$i'"],
        )
        self.expect_expr("$i", result_type="int", result_value="5")

        # Try assigning the persistent variable a new value.
        self.expect("expr $i = 55")
        self.expect_expr("$i", result_type="int", result_value="55")
