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
            return "Hardware breakpoints are supported"
        return None
