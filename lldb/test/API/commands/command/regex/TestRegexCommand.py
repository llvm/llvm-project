"""
This tests some simple examples of parsing regex commands
"""

import os
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestCommandRegexParsing(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_sample_rename_this(self):
        """Try out some simple regex commands, make sure they parse correctly."""
        self.runCmd("command regex one-substitution 's/(.+)/echo-cmd %1-first %1-second %1-third/'")
        self.expect("one-substitution ASTRING",
                    substrs = ["ASTRING-first", "ASTRING-second", "ASTRING-third"])

        self.runCmd("command regex two-substitution 's/([^ ]+) ([^ ]+)/echo-cmd %1-first %2-second %1-third %2-fourth/'")
        self.expect("two-substitution ASTRING BSTRING",
                    substrs = ["ASTRING-first", "BSTRING-second", "ASTRING-third", "BSTRING-fourth"])
        
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.runCmd("command script import " + os.path.join(self.getSourceDir(), "echo_command.py"))
        self.runCmd("command script add echo-cmd -f echo_command.echo_command")
