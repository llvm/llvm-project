
"""
Test that verbose_trap works on forward interop mode.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropVerboseTrap(TestBase):

    @swiftTest
    def test(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        target.BreakpointCreateByName("Break here", "a.out")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Make sure we stopped in the first user-level frame.
        self.assertTrue(self.frame().name.startswith("a.takes<"))
