"""
Test lldb Python commands.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class CommandScriptAliasTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_pycmd(self):
        self.runCmd("command script import custom_command.py")
        self.runCmd("command script add -f custom_command.some_command_here bt")

        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("command script delete bt", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.expect("bt a", substrs=["Script command invoked"])
        self.runCmd("command script delete bt")
        # The built-in 'bt' must be invoked again now. The 'invalid target'
        # output is only created by the builtin 'bt' command.
        self.expect("bt", error=True, substrs=["invalid target"])
        self.assertNotIn("Script command invoked", self.res.GetOutput())
