import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_invalid_arg(self):
        self.expect(
            "target module-hook enable -1",
            error=True,
            startstr='error: invalid module hook id: "-1".',
        )
        self.expect(
            "target module-hook enable abcdfx",
            error=True,
            startstr='error: invalid module hook id: "abcdfx".',
        )
