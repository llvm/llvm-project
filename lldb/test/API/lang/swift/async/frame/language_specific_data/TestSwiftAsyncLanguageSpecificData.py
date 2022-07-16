import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestCase(lldbtest.TestBase):

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test SBFrame.GetLanguageSpecificData() in async functions"""
        self.build()

        # This test uses "main" as a prefix for all functions, so that all will
        # be stopped at. Setting a breakpoint on "main" results in a breakpoint
        # at the start of each coroutine "funclet" function.
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(self, "main")

        # Skip the real `main`. All other stops will be Swift async functions.
        self.assertEqual(thread.frames[0].name, "main")
        process.Continue()

        regex_bpno = lldbutil.run_break_set_by_regexp(self, "^main[[:digit:]]")

        while process.state == lldb.eStateStopped:
            thread = process.GetSelectedThread()
            frame = thread.frames[0]
            data = frame.GetLanguageSpecificData()
            is_async = data.GetValueForKey("IsSwiftAsyncFunction").GetBooleanValue()
            self.assertTrue(is_async)

            process.Continue()

        bkpt = target.FindBreakpointByID(regex_bpno)
        self.assertEqual(bkpt.GetHitCount(), bkpt.num_locations)
        for location in bkpt.locations:
            self.assertEqual(location.GetHitCount(), 1)
