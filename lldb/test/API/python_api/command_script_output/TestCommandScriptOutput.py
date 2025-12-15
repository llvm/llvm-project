"""
Test that HandleCommand captures stdout and stderr from script commands.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class CommandScriptOutputTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_script_command_stdout_stderr(self):
        """Test that HandleCommand captures stdout and stderr from script commands."""
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        res = lldb.SBCommandReturnObject()

        # Execute a script command that writes to stdout.
        ci.HandleCommand("script print('Hello stdout')", res)
        self.assertTrue(res.Succeeded())
        self.assertIn("Hello stdout", res.GetOutput())

        # Execute a script command that writes to stderr.
        ci.HandleCommand("script import sys; sys.stderr.write('Hello stderr\\n')", res)
        self.assertTrue(res.Succeeded())
        self.assertIn("Hello stderr", res.GetOutput())

        # Execute a script command that writes to both stdout and stderr.
        ci.HandleCommand(
            "script import sys; print('Output line'); sys.stderr.write('Error line\\n')",
            res,
        )
        self.assertTrue(res.Succeeded())
        self.assertIn("Output line", res.GetOutput())
        self.assertIn("Error line", res.GetOutput())

        # Test that multiple print statements are captured.
        ci.HandleCommand(
            "script print('Line 1'); print('Line 2'); print('Line 3')", res
        )
        self.assertTrue(res.Succeeded())
        output = res.GetOutput()
        self.assertIn("Line 1", output)
        self.assertIn("Line 2", output)
        self.assertIn("Line 3", output)
