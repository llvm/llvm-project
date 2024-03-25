"""
Tests using a class type that originated from a member function evaluation
with the same class type outside a member function.

When evaluating expressions in a class, LLDB modifies the type of the current
class by injecting a "$__lldb_expr" member function into the class. This
function should not cause the type to become incompatible with its original
definition without the "$__lldb_expr" member.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break in member function", lldb.SBFileSpec("main.cpp")
        )

        # Evaluate an expression in a member function. Store the type of the
        # 'this' pointer in a persistent variable.
        self.expect_expr("A $p = *this; $p", result_type="A")

        breakpoint = self.target().BreakpointCreateBySourceRegex(
            "// break in main", lldb.SBFileSpec("main.cpp")
        )
        self.assertNotEqual(breakpoint.GetNumResolvedLocations(), 0)
        threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint)
        self.assertEqual(len(threads), 1)

        # Evaluate expressions in the main function. Use the persistent type
        # of "A" that came from an evaluation in a member function with a
        # normal "A" type and make sure that LLDB can still evaluate
        # expressions that reference both types at the same time.
        self.expect_expr("$p.i + a.i", result_type="int", result_value="2")
        self.expect_expr("a.i + $p.i", result_type="int", result_value="2")
        self.expect_expr("a = $p; a.i", result_type="int", result_value="1")
