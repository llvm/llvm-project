"""
Test how lldb reacts to ambiguous commands
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AmbiguousCommandTestCase(TestBase):
    @no_debug_info_test
    def test_ambiguous_command_with_alias(self):
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()

        command_interpreter.HandleCommand(
            "command alias corefile target create -c %0", result
        )
        self.assertTrue(result.Succeeded())

        command_interpreter.ResolveCommand("co", result)
        self.assertFalse(result.Succeeded())
        self.assertEqual(
            result.GetError(),
            "Ambiguous command 'co'. Possible matches:\n\tcommand\n\tcontinue\n\tcorefile\n",
        )

        command_interpreter.HandleCommand("command unalias continue", result)
        self.assertTrue(result.Succeeded())

        command_interpreter.ResolveCommand("co", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual(result.GetOutput(), "target create -c %0")
