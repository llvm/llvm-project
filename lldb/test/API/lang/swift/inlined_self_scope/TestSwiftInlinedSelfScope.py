"""
Test expression evaluation when stopped inside an inlined function.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftInlinedSelfScope(TestBase):
    def assert_stopped_in_inlined_frame(self, thread, name):
        frame = thread.GetSelectedFrame()
        self.assertTrue(
            frame.IsInlined(),
            "expected to be stopped in an inlined frame, but stopped in '%s'; "
            "the optimizer may no longer be inlining %s"
            % (frame.GetFunctionName(), name),
        )
        return frame

    @skipEmbeddedSwift
    @swiftTest
    def test_self_scope_in_inlined_frames(self):
        self.build()

        # An inlined *free function* has no `self` of its own, and the
        # caller's `self` must not leak into scope. 
        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self,
            "break here in free function",
            lldb.SBFileSpec("main.swift"),
        )
        self.assert_stopped_in_inlined_frame(thread, "freeFunction()")

        value = thread.GetSelectedFrame().EvaluateExpression("self.count")
        error = value.GetError().GetCString() or ""
        self.assertIn("cannot find 'self' in scope", error)
        # The regression: the wrapper referenced an undeclared variable.
        self.assertNotIn("$__lldb_injected_self", error)

        # Sanity check: An inlined *method* keeps its own `self`, which stays usable.
        threads = lldbutil.continue_to_source_breakpoint(
            self,
            process,
            "break here in method",
            lldb.SBFileSpec("main.swift"),
        )
        self.assertEqual(len(threads), 1, "expected one thread at the breakpoint")
        self.assert_stopped_in_inlined_frame(threads[0], "Tester.inlinedMethod()")

        self.expect("expression self.count", substrs=["41"])
        self.expect("expression count", substrs=["41"])
