"""
Test multiword commands ('platform' in this case).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class MultiwordCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_ambiguous_subcommand(self):
        self.expect("platform s", error=True,
                    substrs=["ambiguous command 'platform s'. Possible completions:",
                             "\tselect\n",
                             "\tshell\n",
                             "\tsettings\n"])

    @no_debug_info_test
    def test_empty_subcommand(self):
        self.expect("platform \"\"", error=True, substrs=["Need to specify a non-empty subcommand."])

    @no_debug_info_test
    def test_help(self):
        # <multiword> help brings up help.
        self.expect("platform help",
                    substrs=["Commands to manage and create platforms.",
                             "Syntax: platform [",
                             "The following subcommands are supported:",
                             "connect",
                             "Select the current platform"])
