"""
Test require hardware breakpoints.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointLocationsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    def supports_hw_breakpoints(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set -b main --hardware")
        self.runCmd("run")
        if 'stopped' in self.res.GetOutput():
            return 'Hardware breakpoints are supported'
        return None

    def test_breakpoint(self):
        """Test regular breakpoints when hardware breakpoints are required."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        breakpoint = target.BreakpointCreateByLocation("main.c", 1)
        self.assertTrue(breakpoint.IsHardware())

    @skipIfWindows
    @expectedFailure(supports_hw_breakpoints)
    def test_step_range(self):
        """Test stepping when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 1)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Ensure we fail in the interpreter.
        self.expect("thread step-in")
        self.expect("thread step-in", error=True)

        # Ensure we fail when stepping through the API.
        error = lldb.SBError()
        thread.StepInto('', 4, error)
        self.assertTrue(error.Fail())
        self.assertTrue("Could not create hardware breakpoint for thread plan"
                        in error.GetCString())

    @skipIfWindows
    @expectedFailure(supports_hw_breakpoints)
    def test_step_out(self):
        """Test stepping out when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 1)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Ensure this fails in the command interpreter.
        self.expect("thread step-out", error=True)

        # Ensure we fail when stepping through the API.
        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Fail())
        self.assertTrue("Could not create hardware breakpoint for thread plan"
                        in error.GetCString())

    @skipIfWindows
    @expectedFailure(supports_hw_breakpoints)
    def test_step_over(self):
        """Test stepping over when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 7)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        # Step over doesn't fail immediately but fails later on.
        self.expect(
            "thread step-over",
            error=True,
            substrs=[
                'error: Could not create hardware breakpoint for thread plan.'
            ])

    @skipIfWindows
    @expectedFailure(supports_hw_breakpoints)
    def test_step_until(self):
        """Test stepping until when hardware breakpoints are required."""
        self.build()

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), 7)

        self.runCmd("settings set target.require-hardware-breakpoint true")

        self.expect("thread until 5", error=True)

        # Ensure we fail when stepping through the API.
        error = thread.StepOverUntil(lldb.SBFrame(), lldb.SBFileSpec(), 5)
        self.assertTrue(error.Fail())
        self.assertTrue("Could not create hardware breakpoint for thread plan"
                        in error.GetCString())
