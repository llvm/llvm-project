"""Test that expressions for the same module can be evaluated and their results
inspected from several threads at the same time."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
# clang-cl does not support throw or catch (llvm.org/pr24538)
@skipIfWindows
@skipIfHostIncompatibleWithTarget
class MultithreadedConcurrentExpressionsTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.buildAndRunDriver(
            "driver.cpp test_concurrent_expressions.cpp", "global_struct.cpp"
        )
