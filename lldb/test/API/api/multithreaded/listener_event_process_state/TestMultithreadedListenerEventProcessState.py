"""Test that a registered SBListener receives events when a process changes state."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedListenerEventProcessStateTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFlakeyFreeBSD
    def test(self):
        self.buildAndRunDriver(
            "driver.cpp listener_test.cpp test_listener_event_process_state.cpp",
            "inferior.cpp",
        )
