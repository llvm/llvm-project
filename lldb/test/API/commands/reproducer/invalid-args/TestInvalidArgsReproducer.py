import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class ReproducerTestCase(TestBase):

    @no_debug_info_test
    def test_reproducer_generate_invalid_invocation(self):
        self.expect("reproducer generate f", error=True,
                    substrs=["'reproducer generate' doesn't take any arguments"])

    @no_debug_info_test
    def test_reproducer_status_invalid_invocation(self):
        self.expect("reproducer status f", error=True,
                    substrs=["'reproducer status' doesn't take any arguments"])
