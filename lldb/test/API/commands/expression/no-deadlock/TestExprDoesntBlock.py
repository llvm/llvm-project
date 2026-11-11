"""
Test that expr will time out and allow other threads to run if it blocks.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
class ExprDoesntDeadlockTestCase(TestBase):
    @add_test_categories(["basic_process"])
    def test_with_run_command(self):
        """Test that expr will time out and allow other threads to run if it blocks."""
        self.build()

        # Now create a breakpoint at source line before call_me_to_get_lock
        # gets called.
        _, _, thread, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("locking.cpp")
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)

        frame0 = thread.GetFrameAtIndex(0)

        var = frame0.EvaluateExpression("call_me_to_get_lock(get_int())")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsSigned(0), 567)
