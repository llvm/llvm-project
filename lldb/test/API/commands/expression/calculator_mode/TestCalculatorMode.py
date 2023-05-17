"""
Test calling an expression without a target.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCalculatorMode(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test__calculator_mode(self):
        """Test calling expressions in the dummy target."""
        self.expect("expression 11 + 22", "11 + 22 didn't get the expected result", substrs=["33"])
        # Now try it with a specific language:
        self.expect("expression -l c -- 11 + 22", "11 + 22 didn't get the expected result", substrs=["33"])

