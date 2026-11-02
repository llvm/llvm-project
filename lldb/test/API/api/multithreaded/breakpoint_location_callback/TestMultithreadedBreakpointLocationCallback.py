"""Test that the SBBreakpointLocation callback is invoked when a breakpoint is hit."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedBreakpointLocationCallbackTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/224303
    def test(self):
        self.buildAndRunDriver(
            "driver.cpp test_breakpoint_location_callback.cpp", "inferior.cpp"
        )
