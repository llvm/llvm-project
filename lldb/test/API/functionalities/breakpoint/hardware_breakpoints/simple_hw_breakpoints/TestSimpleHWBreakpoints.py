import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from functionalities.breakpoint.hardware_breakpoints.base import *


class SimpleHWBreakpointTest(HardwareBreakpointTestBase):
    def does_not_support_hw_breakpoints(self):
        # FIXME: Use HardwareBreakpointTestBase.supports_hw_breakpoints
        if super().supports_hw_breakpoints() is None:
            return "Hardware breakpoints are unsupported"
        return None

    @skipTestIfFn(does_not_support_hw_breakpoints)
    def test(self):
        """Test SBBreakpoint::SetIsHardware"""
        self.build()

        # Set a breakpoint on main.
        target, process, _, main_bp = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.c")
        )

        break_on_me_bp = target.BreakpointCreateByLocation("main.c", 1)

        self.assertFalse(main_bp.IsHardware())
        self.assertFalse(break_on_me_bp.IsHardware())
        self.assertGreater(break_on_me_bp.GetNumResolvedLocations(), 0)

        error = break_on_me_bp.SetIsHardware(True)

        # Regardless of whether we succeeded in updating all the locations, the
        # breakpoint will be marked as a hardware breakpoint.
        self.assertTrue(break_on_me_bp.IsHardware())

        if super().supports_hw_breakpoints():
            self.assertSuccess(error)

            # Continue to our Hardware breakpoint and verify that's the reason
            # we're stopped.
            process.Continue()
            self.expect(
                "thread list",
                STOPPED_DUE_TO_BREAKPOINT,
                substrs=["stopped", "stop reason = breakpoint"],
            )
        else:
            self.assertFailure(error)
