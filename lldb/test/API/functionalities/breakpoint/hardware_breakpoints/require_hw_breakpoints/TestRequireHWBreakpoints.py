"""
Test require hardware breakpoints.

Some of these tests require a target that does not have hardware breakpoints.
So that we can check we fail when required to use them.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from functionalities.breakpoint.hardware_breakpoints.base import *


class BreakpointLocationsTestCase(HardwareBreakpointTestBase):
    def test_breakpoint(self):
        """Test regular breakpoints when hardware breakpoints are required."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        breakpoint = target.BreakpointCreateByLocation("main.c", 1)
        self.assertTrue(breakpoint.IsHardware())

    @expectedFailureIfFn(HardwareBreakpointTestBase.supports_hw_breakpoints)
    def test_step_range(self):
        """Test stepping when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 1
        )

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Ensure we fail in the interpreter.
        self.expect("thread step-in")
        self.expect("thread step-in", error=True)

        # Ensure we fail when stepping through the API.
        error = lldb.SBError()
        thread.StepInto("", 4, error)
        self.assertTrue(error.Fail())
        self.assertTrue(
            "Could not create hardware breakpoint for thread plan" in error.GetCString()
        )

    @expectedFailureIfFn(HardwareBreakpointTestBase.supports_hw_breakpoints)
    def test_step_out(self):
        """Test stepping out when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 1
        )

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Ensure this fails in the command interpreter.
        self.expect("thread step-out", error=True)

        # Ensure we fail when stepping through the API.
        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Fail())
        self.assertTrue(
            "Could not create hardware breakpoint for thread plan" in error.GetCString()
        )

    @expectedFailureIfFn(HardwareBreakpointTestBase.supports_hw_breakpoints)
    def test_step_over(self):
        """Test stepping over when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 7
        )

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Step over doesn't fail immediately but fails later on.
        self.expect(
            "thread step-over",
            error=True,
            substrs=["error: Could not create hardware breakpoint for thread plan."],
        )

    # Was reported to sometimes pass on certain hardware.
    @skipIf(oslist=["linux"], archs=["arm"])
    @expectedFailureIfFn(HardwareBreakpointTestBase.supports_hw_breakpoints)
    def test_step_until(self):
        """Test stepping until when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 7
        )

        self.runCmd("settings set target.require-hardware-breakpoint true")

        self.expect("thread until 5", error=True)

        # Ensure we fail when stepping through the API.
        error = thread.StepOverUntil(lldb.SBFrame(), lldb.SBFileSpec(), 5)
        self.assertTrue(error.Fail())
        self.assertTrue(
            "Could not create hardware breakpoint for thread plan" in error.GetCString()
        )
