"""
Test some lldb command abbreviations and aliases for proper resolution.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AbbreviationsTestCase(TestBase):
    @no_debug_info_test
    def test_command_abbreviations_and_aliases(self):
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()

        # Check that abbreviations are expanded to the full command.
        command_interpreter.ResolveCommand("ap script", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("apropos script", result.GetOutput())

        command_interpreter.ResolveCommand("h", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("help", result.GetOutput())

        # Check resolution of abbreviations for multi-word commands.
        command_interpreter.ResolveCommand("lo li", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("log list", result.GetOutput())

        command_interpreter.ResolveCommand("br s", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("breakpoint set", result.GetOutput())

        # Try an ambiguous abbreviation.
        # "pl" could be "platform" or "plugin".
        command_interpreter.ResolveCommand("pl", result)
        self.assertFalse(result.Succeeded())
        self.assertTrue(result.GetError().startswith("Ambiguous command"))

        # Make sure an unabbreviated command is not mangled.
        command_interpreter.ResolveCommand(
            "breakpoint set --name main --line 123", result
        )
        self.assertTrue(result.Succeeded())
        self.assertEqual("breakpoint set --name main --line 123", result.GetOutput())

        # Create some aliases.
        self.runCmd("com a alias com al")
        self.runCmd("alias gurp help")

        # Check that an alias is replaced with the actual command
        command_interpreter.ResolveCommand("gurp target create", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("help target create", result.GetOutput())

        # Delete the alias and make sure it no longer has an effect.
        self.runCmd("com u gurp")
        command_interpreter.ResolveCommand("gurp", result)
        self.assertFalse(result.Succeeded())

        # Check aliases with text replacement.
        self.runCmd("alias pltty process launch -s -o %1 -e %1")
        command_interpreter.ResolveCommand("pltty /dev/tty0", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual(
            "process launch -s -o /dev/tty0 -e /dev/tty0", result.GetOutput()
        )

        self.runCmd("alias xyzzy breakpoint set -n %1 -l %2")
        command_interpreter.ResolveCommand("xyzzy main 123", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("breakpoint set -n main -l 123", result.GetOutput().strip())

        # And again, without enough parameters.
        command_interpreter.ResolveCommand("xyzzy main", result)
        self.assertFalse(result.Succeeded())

        # Check a command that wants the raw input.
        command_interpreter.ResolveCommand(r"""sc print("\n\n\tHello!\n")""", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual(
            r"""scripting run print("\n\n\tHello!\n")""", result.GetOutput()
        )

        command_interpreter.ResolveCommand("script 1+1", result)
        self.assertTrue(result.Succeeded())
        self.assertEqual("scripting run 1+1", result.GetOutput())

        # Prompt changing stuff should be tested, but this doesn't seem like the
        # right test to do it in.  It has nothing to do with aliases or abbreviations.
        # self.runCmd("com sou ./change_prompt.lldb")
        # self.expect("settings show prompt",
        #            startstr = 'prompt (string) = "[with-three-trailing-spaces]   "')
        # self.runCmd("settings clear prompt")
        # self.expect("settings show prompt",
        #            startstr = 'prompt (string) = "(lldb) "')
        # self.runCmd("se se prompt 'Sycamore> '")
        # self.expect("se sh prompt",
        #            startstr = 'prompt (string) = "Sycamore> "')
        # self.runCmd("se cl prompt")
        # self.expect("set sh prompt",
        #            startstr = 'prompt (string) = "(lldb) "')
