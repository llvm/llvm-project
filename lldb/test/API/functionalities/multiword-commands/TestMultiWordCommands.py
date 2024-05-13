"""
Test multiword commands ('platform' in this case).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MultiwordCommandsTestCase(TestBase):
    @no_debug_info_test
    def test_ambiguous_subcommand(self):
        self.expect(
            "platform s",
            error=True,
            substrs=[
                "ambiguous command 'platform s'. Possible completions:",
                "\tselect\n",
                "\tsettings\n",
                "\tshell\n",
            ],
        )

    @no_debug_info_test
    def test_empty_subcommand(self):
        self.expect(
            'platform ""',
            error=True,
            substrs=["Need to specify a non-empty subcommand."],
        )
