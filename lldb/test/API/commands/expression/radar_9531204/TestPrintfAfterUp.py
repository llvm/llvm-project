"""
The evaluating printf(...) after break stop and then up a stack frame.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Radar9531204TestCase(TestBase):
    # rdar://problem/9531204
    def test_expr_commands(self):
        """The evaluating printf(...) after break stop and then up a stack frame."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_symbol(
            self, "foo", sym_exact=True, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("frame variable")

        # This works fine.
        self.runCmd('expression (int)printf("value is: %d.\\n", value);')

        # rdar://problem/9531204
        # "Error dematerializing struct" error when evaluating expressions "up" on the stack
        self.runCmd("up")  # frame select -r 1

        self.runCmd("frame variable")

        # This does not currently.
        self.runCmd('expression (int)printf("argc is: %d.\\n", argc)')
