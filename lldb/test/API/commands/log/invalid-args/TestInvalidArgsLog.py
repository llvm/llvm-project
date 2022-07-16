import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class InvalidArgsLogTestCase(TestBase):

    @no_debug_info_test
    def test_enable_empty(self):
        self.expect("log enable", error=True,
                    substrs=["error: log enable takes a log channel and one or more log types."])

    @no_debug_info_test
    def test_disable_empty(self):
        self.expect("log disable", error=True,
                    substrs=["error: log disable takes a log channel and one or more log types."])

    @no_debug_info_test
    def test_enable_empty(self):
        invalid_path = os.path.join("this", "is", "not", "a", "valid", "path")
        self.expect("log enable lldb all -f " + invalid_path, error=True,
                    substrs=["Unable to open log file '" + invalid_path + "': ", "\n"])
