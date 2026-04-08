"""
Base class for hardware breakpoints tests.
"""

from lldbsuite.test.lldbtest import *


class HardwareBreakpointTestBase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def supports_hw_breakpoints(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        self.runCmd("breakpoint set -b main --hardware")
        self.runCmd("run")
        if "stopped" in self.res.GetOutput():
            return True
        return False

    def hw_breakpoints_supported(self):
        if self.supports_hw_breakpoints():
            return "Hardware breakpoints are supported"
        return None

    def hw_breakpoints_unsupported(self):
        if not self.supports_hw_breakpoints():
            return "Hardware breakpoints are unsupported"
        return None
