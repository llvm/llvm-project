"""
Test that lldb persistent variables works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PersistentVariablesTestCase(TestBase):

    mydir = os.path.join("expression_command", "persistent_variables")

    def test_persistent_variables(self):
        """Test that lldb persistent variables works correctly."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --source-pattern-regexp break")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expression int $i = i")

        self.expect("expression $i == i",
            startstr = "(bool) $0 = true")

        self.expect("expression $i + 1",
            startstr = "(int) $1 = 6")

        self.expect("expression $i + 3",
            startstr = "(int) $2 = 8")

        self.expect("expression $2 + $1",
            startstr = "(int) $3 = 14")

        self.expect("expression $3",
            startstr = "(int) $3 = 14")

        self.expect("expression $2",
            startstr = "(int) $2 = 8")

        self.expect("expression (int)-2",
            startstr = "(int) $4 = -2")

        self.expect("expression $4 > (int)31",
            startstr = "(bool) $5 = false")

        self.expect("expression (long)$4",
            startstr = "(long) $6 = -2")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
