"""
Test that backticks without a target should work (not infinite looping).
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BackticksWithNoTargetTestCase(TestBase):
    @no_debug_info_test
    def test_backticks_no_target(self):
        """A simple test of backticks without a target."""
        self.expect("expression `1+2-3`", substrs=[" = 0"])
