import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import re

class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test step-in to async functions"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        _, process, _, _ = lldbutil.run_to_source_breakpoint(self, 'await', src)

        # When run with debug info enabled builds, this prevents stepping from
        # stopping in Swift Concurrency runtime functions.
        self.runCmd("settings set target.process.thread.step-avoid-libraries libswift_Concurrency.dylib")

        # All thread actions are done on the currently selected thread.
        thread = process.GetSelectedThread

        num_async_steps = 0
        while True:
            stop_reason = thread().stop_reason
            if stop_reason == lldb.eStopReasonNone:
                break
            elif stop_reason == lldb.eStopReasonPlanComplete:
                # Run until the next `await` breakpoint.
                process.Continue()
            elif stop_reason == lldb.eStopReasonBreakpoint:
                caller_before = thread().frames[0].function.GetDisplayName()
                thread().StepInto()
                caller_after = thread().frames[1].function.GetDisplayName()
                self.assertEqual(caller_after, caller_before)
                num_async_steps += 1

        self.assertGreater(num_async_steps, 0)
