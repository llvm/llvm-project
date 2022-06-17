"""
Test default template arguments.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestDefaultTemplateArgs(TestBase):

    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Declare a template with a template argument that has a default argument.
        self.expect("expr --top-level -- template<typename T = int> struct $X { int v; };")

        # The type we display to the user should omit the argument with the default
        # value.
        result = self.expect_expr("$X<> x; x",  result_type="$X<>")
        # The internal name should also always show all arguments (even if they
        # have their default value).
        self.assertEqual(result.GetTypeName(), "$X<int>")

        # Test the template but this time specify a non-default value for the
        # template argument.
        # Both internal type name and the one we display to the user should
        # show the non-default value in the type name.
        result = self.expect_expr("$X<long> x; x", result_type="$X<long>")
        self.assertEqual(result.GetTypeName(), "$X<long>")

        # Test that the formatters are using the internal type names that
        # always include all template arguments.
        self.expect("type summary add '$X<int>' --summary-string 'summary1'")
        self.expect_expr("$X<> x; x", result_summary="summary1")
        self.expect("type summary add '$X<long>' --summary-string 'summary2'")
        self.expect_expr("$X<long> x; x", result_summary="summary2")
