"""
Test expression evaluation in a frame whose `self` type cannot be realized.

`Lib.Root` is defined in one module and extended from another.  Reflection
metadata for `Lib.Root` is unavailable and Lib's AST is not reachable either, so
LLDB has a valid *location* for `self` but no usable value for it.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftCrossModuleExtensionSelf(TestBase):
    @requireNotEmbeddedSwift
    @swiftTest
    def test_unusable_self_does_not_break_frame(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        # Verify that `self` is indeed not available.
        self_var = self.frame().FindVariable("self")
        self.assertTrue(self_var.IsValid(), "expected a 'self' variable")
        self.assertEqual(self_var.GetType().GetName(), "Lib.Root")
        self.assertTrue(
            self_var.GetError().Fail(),
            "expected 'self' to have no usable value, got %s"
            % self_var.GetValue(),
        )

        # A local variable is not a member of `self`, so it must still resolve --
        # and materialize -- even though `self` was dropped from the wrapper.
        factor = self.frame().EvaluateExpression("factor")
        self.assertSuccess(factor.GetError())
        self.assertEqual(factor.GetValue(), "7")

        # Expressions that need `self` must fail with a diagnostic about `self`.
        error = self.frame().EvaluateExpression("self").GetError().GetCString() or ""
        self.assertIn("cannot find 'self' in scope", error)
        self.assertNotIn("$__lldb_injected_self", error)
