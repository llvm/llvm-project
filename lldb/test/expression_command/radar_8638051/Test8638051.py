"""
Test the robustness of lldb expression parser.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class Radar8638051TestCase(TestBase):

    mydir = os.path.join("expression_command", "radar_8638051")

    def test_expr_commands(self):
        """The following expression commands should not crash."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set -n c")

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expression val",
            startstr = "(int) $0 = 1")
        # (int) $0 = 1

        self.expect("expression *(&val)",
            startstr = "(int) $1 = 1")
        # (int) $1 = 1

        # rdar://problem/8638051
        # lldb expression command: Could this crash be avoided
        self.expect("expression &val",
            startstr = "(int *) $2 = ")
        # (int *) $2 = 0x....


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
