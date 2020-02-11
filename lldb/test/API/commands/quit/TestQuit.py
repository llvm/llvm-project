"""
Test lldb's quit command.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class QuitCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_quit_exit_code_disallow(self):
        self.ci.AllowExitCodeOnQuit(False)
        self.expect(
            "quit 20",
            substrs=[
                "error: The current driver doesn't allow custom exit codes for the quit command"],
            error=True)
        self.assertFalse(self.ci.HasCustomQuitExitCode())

    @no_debug_info_test
    def test_quit_exit_code_allow(self):
        self.ci.AllowExitCodeOnQuit(True)
        self.runCmd("quit 10", check=False)
        self.assertTrue(self.ci.HasCustomQuitExitCode())
        self.assertEqual(self.ci.GetQuitStatus(), 10)
