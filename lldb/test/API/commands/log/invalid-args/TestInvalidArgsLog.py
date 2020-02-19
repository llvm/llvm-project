import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class InvalidArgsLogTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_enable_empty(self):
        self.expect("log enable", error=True,
                    substrs=["error: log enable takes a log channel and one or more log types."])

    @no_debug_info_test
    def test_disable_empty(self):
        self.expect("log disable", error=True,
                    substrs=["error: log disable takes a log channel and one or more log types."])

    @no_debug_info_test
    def test_timer_empty(self):
        self.expect("log timer", error=True,
                    substrs=["error: Missing subcommand"])
