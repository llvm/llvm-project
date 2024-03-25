"""
Test example snippets from the lldb 'help expression' output.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Radar9673644TestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.c"
        self.line = line_number(self.main_source, "// Set breakpoint here.")

    def test_expr_commands(self):
        """The following expression commands should just work."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, self.main_source, self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # rdar://problem/9673664 lldb expression evaluation problem

        self.expect('expr char str[] = "foo"; str[0]', substrs=["'f'"])
        # runCmd: expr char c[] = "foo"; c[0]
        # output: (char) $0 = 'f'
