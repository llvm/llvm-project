import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_keyword(self):
        # Make sure that C++ keywords work in the expression parser.
        self.expect("expr -l c++ -- constexpr int i = 3 + 3; i", substrs=["= 6"])
