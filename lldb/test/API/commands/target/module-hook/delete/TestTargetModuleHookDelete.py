import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_invalid_arg(self):
        self.expect(
            "target hook delete -1",
            error=True,
            startstr='error: invalid hook id: "-1"',
        )
        self.expect(
            "target hook delete abcdfx",
            error=True,
            startstr='error: invalid hook id: "abcdfx"',
        )
