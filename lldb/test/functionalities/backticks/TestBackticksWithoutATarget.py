"""
Test that backticks without a target should work (not infinite looping).
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class BackticksWithNoTargetTestCase(TestBase):

    mydir = "functionalities/backticks"

    def test_backticks_no_target(self):
        """A simple test of backticks without a target."""
        self.expect("print `1+2-3`",
            substrs = [' = 0'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
