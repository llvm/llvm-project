"""Test that the description of an SBListener breakpoint event is valid."""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireThreadSupport
@skipIfNoSBHeaders
@skipIfRemote
@skipIfHostIncompatibleWithTarget
class MultithreadedListenerEventDescriptionTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFlakeyFreeBSD
    def test(self):
        self.buildAndRunDriver(
            "driver.cpp listener_test.cpp test_listener_event_description.cpp",
            "inferior.cpp",
        )
