import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_file_close_invalid_arg(self):
        self.expect("platform file close y", error=True,
                    substrs=["'y' is not a valid file descriptor."])
        self.expect("platform file close -1", error=True,
                    substrs=["'-1' is not a valid file descriptor."])
