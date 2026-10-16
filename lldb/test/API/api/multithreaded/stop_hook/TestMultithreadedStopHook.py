"""Test that you can run a python command in a stop-hook when stdin is File based."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedStopHookTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/225860
    # Occasional SIGABRT, see https://github.com/llvm/llvm-project/issues/225860.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test(self):
        self.buildAndRunDriver("driver.cpp test_stop-hook.cpp", "inferior.cpp")
