import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class DeleteCommandTestCase(TestBase):

    @no_debug_info_test
    def test_delete_builtin(self):
        self.expect("command delete settings", error=True,
                    substrs=["'settings' is a permanent debugger command and cannot be removed."])

    @no_debug_info_test
    def test_delete_alias(self):
        self.expect("command delete bt", error=True,
                    substrs=["'bt' is not a known command."])
