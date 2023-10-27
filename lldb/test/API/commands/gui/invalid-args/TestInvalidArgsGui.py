import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class GuiTestCase(TestBase):
    @no_debug_info_test
    @skipIfCursesSupportMissing
    def test_reproducer_generate_invalid_invocation(self):
        self.expect(
            "gui blub", error=True, substrs=["'gui' doesn't take any arguments."]
        )
