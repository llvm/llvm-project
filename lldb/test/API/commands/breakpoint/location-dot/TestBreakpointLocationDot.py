import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_disable_enable(self):
        self.build()
        _, _, _, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        self.assertTrue(bp.FindLocationByID(1).IsEnabled())
        self.expect("breakpoint disable .", startstr="1 breakpoints disabled.")
        self.assertFalse(bp.FindLocationByID(1).IsEnabled())
        self.expect("breakpoint enable .", startstr="1 breakpoints enabled.")
        self.assertTrue(bp.FindLocationByID(1).IsEnabled())

    def test_delete(self):
        self.build()
        _, _, _, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        self.expect(
            "breakpoint delete .",
            startstr="0 breakpoints deleted; 1 breakpoint locations disabled",
        )
        self.assertFalse(bp.FindLocationByID(1).IsEnabled())

    def test_error_not_breakpoint_stop(self):
        self.build()
        _, _, thread, bp = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        self.assertTrue(bp.FindLocationByID(1).IsEnabled())
        thread.StepOver()
        self.assertNotEqual(thread.stop_reason, lldb.eStopReasonBreakpoint)
        self.expect(
            "breakpoint disable .",
            error=True,
            startstr="error: current thread is not stopped at a breakpoint",
        )
        self.assertTrue(bp.FindLocationByID(1).IsEnabled())

    def test_error_no_process(self):
        self.build()
        target = self.createTestTarget()
        target.BreakpointCreateByLocation("main.c", 2)
        self.expect("breakpoint disable .", error=True, substrs=["no current thread"])
