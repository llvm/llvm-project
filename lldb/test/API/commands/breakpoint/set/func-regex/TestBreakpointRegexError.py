import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_error(self):
        self.expect("breakpoint set --func-regex (", error=True,
                    substrs=["error: Function name regular expression could " +
                             "not be compiled: parentheses not balanced"])
