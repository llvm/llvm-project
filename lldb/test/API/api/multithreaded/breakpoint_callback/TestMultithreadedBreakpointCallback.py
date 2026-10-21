"""Test that the SBBreakpoint callback is invoked when a breakpoint is hit."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedBreakpointCallbackTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.buildAndRunDriver(
            "driver.cpp test_breakpoint_callback.cpp", "inferior.cpp"
        )
