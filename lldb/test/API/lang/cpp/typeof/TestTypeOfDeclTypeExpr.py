import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test(self):
        self.expect_expr("int i; __typeof__(i) j = 1; j", result_type="typeof (i)", result_value="1")
        self.expect_expr("int i; typeof(i) j = 1; j", result_type="typeof (i)", result_value="1")
        self.expect_expr("int i; decltype(i) j = 1; j", result_type="decltype(i)", result_value="1")
