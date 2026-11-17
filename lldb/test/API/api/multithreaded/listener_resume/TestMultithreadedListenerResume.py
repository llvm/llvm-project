"""Test that a process can be resumed from a non-main thread."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedListenerResumeTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFlakeyFreeBSD
    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/231113
    @skipIf(oslist=["linux"])  # flakey
    def test(self):
        self.buildAndRunDriver(
            "driver.cpp listener_test.cpp test_listener_resume.cpp", "inferior.cpp"
        )
