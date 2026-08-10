"""
Test evaluating expressions whose result is an rvalue with a non-trivial
destructor.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # An lvalue result is turned into a '$__lldb_expr_result_ptr' and needs
        # no destructor.
        self.expect_expr("f.x", result_type="int", result_value="42")

        # An rvalue result is turned into a static '$__lldb_expr_result' whose
        # destructor gets registered with atexit/__cxa_atexit.
        self.expect_expr(
            "make_foo()",
            result_type="Foo",
            result_children=[ValueCheck(name="x", value="42")],
        )
        self.expect_expr(
            "make_widget()",
            result_type="Widget",
            result_children=[ValueCheck(name="x", value="47")],
        )

        # Make sure evaluating the expressions didn't leave a dangling
        # destructor registered in the inferior.
        target.DeleteAllBreakpoints()
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
