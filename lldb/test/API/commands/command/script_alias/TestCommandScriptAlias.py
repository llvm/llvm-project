"""
Test lldb Python commands.
"""


import lldb
from lldbsuite.test.lldbtest import *


class CommandScriptAliasTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_pycmd(self):
        self.runCmd("command script import tcsacmd.py")
        self.runCmd("command script add -f tcsacmd.some_command_here attach")

        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("command script delete attach", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.expect("attach a", substrs=["Victory is mine"])
        self.runCmd("command script delete attach")
        # this can't crash but we don't care whether the actual attach works
        self.runCmd("attach noprocessexistswiththisname", check=False)
