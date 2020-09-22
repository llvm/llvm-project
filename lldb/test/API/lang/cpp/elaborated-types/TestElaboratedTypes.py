"""
Test elaborated types (e.g. Clang's ElaboratedType or TemplateType sugar).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        # Add a type formatter for 'Struct'.
        self.expect("type summary add Struct --summary-string '(summary x=${var.x})'")
        # Check that creating an expr with an elaborated type ('::Struct')
        # triggers our formatter for 'Struct' while keeping the elaborated type
        # as the display type.
        result = self.expect_expr("::Struct s; s.x = 4; s",
                                  result_type="::Struct",
                                  result_summary="(summary x=4)")
        # Test that a plain elaborated type is only in the display type name but
        # not in the full type name.
        self.assertEqual(result.GetTypeName(), "Struct")

        # Test the same for template types (that also only act as sugar to better
        # show how the template was specified by the user).

        # Declare a template that can actually be instantiated.
        # FIXME: The error message here is incorrect.
        self.expect("expr --top-level -- template<typename T> struct $V {};",
                    error=True, substrs=["Couldn't find $__lldb_expr() in the module"])
        result = self.expect_expr("$V<::Struct> s; s",
                                  result_type="$V< ::Struct>")
        self.assertEqual(result.GetTypeName(), "$V<Struct>")
