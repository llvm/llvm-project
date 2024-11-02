import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @no_debug_info_test
    def test_invalid_arg(self):
        self.expect("target select -1", error=True,
                    startstr="error: invalid index string value '-1'")
        self.expect("target select abcdfx", error=True,
                    startstr="error: invalid index string value 'abcdfx'")
