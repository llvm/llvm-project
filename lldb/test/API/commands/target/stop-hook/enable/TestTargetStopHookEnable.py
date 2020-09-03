import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_invalid_arg(self):
        self.expect("target stop-hook enable -1", error=True,
                    startstr="error: invalid stop hook id: \"-1\".")
        self.expect("target stop-hook enable abcdfx", error=True,
                    startstr="error: invalid stop hook id: \"abcdfx\".")
