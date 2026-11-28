"""Test that a stop stays private to the thread handling it."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedStopReportingTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/225860
    # Occasional SIGABRT, see https://github.com/llvm/llvm-project/issues/225860.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test(self):
        self.buildAndRunDriver(
            "driver.cpp test_stop_reporting.cpp", "stop_reporting.cpp"
        )
