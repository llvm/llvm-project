import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_disable_enable(self):
        self.build()
        _, _, thread, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        loc_id = self._stop_location_id(thread)
        self.assertTrue(bp.FindLocationByID(loc_id).IsEnabled())
        self.expect("breakpoint disable .", startstr="1 breakpoints disabled.")
        self.assertFalse(bp.FindLocationByID(loc_id).IsEnabled())
        self.expect("breakpoint enable .", startstr="1 breakpoints enabled.")
        self.assertTrue(bp.FindLocationByID(loc_id).IsEnabled())

    def test_delete(self):
        self.build()
        _, _, thread, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        loc_id = self._stop_location_id(thread)
        self.expect(
            "breakpoint delete .",
            startstr="0 breakpoints deleted; 1 breakpoint locations disabled",
        )
        self.assertFalse(bp.FindLocationByID(loc_id).IsEnabled())

    def test_error_not_breakpoint_stop(self):
        self.build()
        _, _, thread, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        loc_id = self._stop_location_id(thread)
        self.assertTrue(bp.FindLocationByID(loc_id).IsEnabled())
        thread.StepOver()
        self.assertNotEqual(thread.stop_reason, lldb.eStopReasonBreakpoint)
        self.expect(
            "breakpoint disable .",
            error=True,
            startstr="error: current thread is not stopped at a breakpoint",
        )
        self.assertTrue(bp.FindLocationByID(loc_id).IsEnabled())

    def test_error_no_process(self):
        self.build()
        target = self.createTestTarget()
        target.BreakpointCreateByLocation("main.c", 2)
        self.expect("breakpoint disable .", error=True, substrs=["no current thread"])

    def _stop_location_id(self, thread: lldb.SBThread) -> int:
        # At a breakpoint stop, the stop reason data has the following structure:
        #   [bp1_id, loc1_id, bp2_id, loc2_id, ...]
        self.assertEqual(
            thread.GetStopReasonDataCount(),
            2,
            "stop should be for one breakpoint only",
        )
        return thread.GetStopReasonDataAtIndex(1)
