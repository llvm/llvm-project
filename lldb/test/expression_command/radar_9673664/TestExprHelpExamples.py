"""
Test example snippets from the lldb 'help expression' output.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class Radar9673644TestCase(TestBase):

    mydir = os.path.join("expression_command", "radar_9673664")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.c"
        self.line = line_number(self.main_source, '// Set breakpoint here.')

    # rdar://problem/9673664
    def test_expr_commands(self):
        """The following expression commands should just work."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, self.main_source, self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # rdar://problem/9673664 lldb expression evaluation problem

        self.expect('expr char c[] = "foo"; c[0]',
            substrs = ["'f'"])
        # runCmd: expr char c[] = "foo"; c[0]
        # output: (char) $0 = 'f'


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
