import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class TerminalDimensionsTest(PExpectTest):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfAsan
    def test(self):
        """Test that the lldb driver correctly reports the (PExpect) terminal dimension."""
        self.launch(dimensions=(40, 40))

        # Tests clear all the settings so we lose the launch values. Resize the
        # window to update the settings. These new values need to be different
        # to trigger a SIGWINCH.
        self.child.setwinsize(20, 60)

        self.expect("settings show term-height", ["term-height (unsigned) = 20"])
        self.expect("settings show term-width", ["term-width (unsigned) = 60"])
