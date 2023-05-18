import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueAPIVoidStarTestCase(TestBase):
    def test(self):
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Break at this line", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)

        # Verify that the expression result for a void * behaves the same way as the
        # variable value.

        var_val = frame.FindVariable("void_ptr")
        self.assertSuccess(var_val.GetError(), "Var version made correctly")

        expr_val = frame.EvaluateExpression("void_ptr")
        self.assertSuccess(expr_val.GetError(), "Expr version succeeds")

        # The pointer values should be equal:
        self.assertEqual(var_val.unsigned, expr_val.unsigned, "Values are equal")

        # Both versions should have valid AddressOf, and they should be the same.

        val_addr_of = var_val.AddressOf()
        self.assertNotEqual(val_addr_of, lldb.LLDB_INVALID_ADDRESS, "Var addr of right")

        expr_addr_of = expr_val.AddressOf()
        self.assertNotEqual(
            expr_addr_of, lldb.LLDB_INVALID_ADDRESS, "Expr addr of right"
        )

        # The AddressOf values should also be equal.
        self.assertEqual(expr_addr_of.unsigned, val_addr_of.unsigned, "Addr of equal")
