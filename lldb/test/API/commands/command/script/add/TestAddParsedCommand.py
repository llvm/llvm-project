"""
Test option and argument definitions in parsed script commands
"""


import sys
import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class ParsedCommandTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.pycmd_tests()

    def check_help_options(self, cmd_name, opt_list, substrs=[]):
        """
        Pass the command name in cmd_name and a vector of the short option, type & long option.
        This will append the checks for all the options and test "help command".
        Any strings already in substrs will also be checked.
        Any element in opt list that begin with "+" will be added to the checked strings as is.
        """
        for elem in opt_list:
            if elem[0] == "+":
                substrs.append(elem[1:])
            else:
                (short_opt, type, long_opt) = elem
                substrs.append(f"-{short_opt} <{type}> ( --{long_opt} <{type}> )")
        print(f"Opt Vec\n{substrs}")
        self.expect("help " + cmd_name, substrs=substrs)

    def pycmd_tests(self):
        source_dir = self.getSourceDir()
        test_file_path = os.path.join(source_dir, "test_commands.py")
        self.runCmd("command script import " + test_file_path)
        self.expect("help", substrs=["no-args", "one-arg-no-opt", "two-args"])

        # Test that we did indeed add these commands as user commands:

        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd(
                "command script delete no-args one-arg-no-opt two-args", check=False
            )

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # First test the no arguments command.  Make sure the help is right:
        no_arg_opts = [
            ["b", "boolean", "bool-arg"],
            "+a boolean arg, defaults to True",
            ["d", "filename", "disk-file-name"],
            "+An on disk filename",
            ["e", "none", "enum-option"],
            "+An enum, doesn't actually do anything",
            "+Values: foo | bar | baz",
            ["l", "linenum", "line-num"],
            "+A line number",
            ["s", "shlib-name", "shlib-name"],
            "+A shared library name",
        ]
        substrs = [
            "Example command for use in debugging",
            "Syntax: no-args <cmd-options>",
        ]

        self.check_help_options("no-args", no_arg_opts, substrs)

        # Make sure the command doesn't accept arguments:
        self.expect(
            "no-args an-arg",
            substrs=["'no-args' doesn't take any arguments."],
            error=True,
        )

        # Try setting the bool with the wrong value:
        self.expect(
            "no-args -b Something",
            substrs=["Error setting option: bool-arg to Something"],
            error=True,
        )
        # Try setting the enum to an illegal value as well:
        self.expect(
            "no-args --enum-option Something",
            substrs=["error: Error setting option: enum-option to Something"],
            error=True,
        )

        # Check some of the command groups:
        self.expect(
            "no-args -b true -s Something -l 10",
            substrs=["error: invalid combination of options for the given command"],
            error=True,
        )

        # Now set the bool arg correctly, note only the first option was set:
        self.expect(
            "no-args -b true",
            substrs=[
                "bool-arg (set: True): True",
                "shlib-name (set: False):",
                "disk-file-name (set: False):",
                "line-num (set: False):",
                "enum-option (set: False):",
            ],
        )

        # Now set the enum arg correctly, note only the first option was set:
        self.expect(
            "no-args -e foo",
            substrs=[
                "bool-arg (set: False):",
                "shlib-name (set: False):",
                "disk-file-name (set: False):",
                "line-num (set: False):",
                "enum-option (set: True): foo",
            ],
        )
        # Try a pair together:
        self.expect(
            "no-args -b false -s Something",
            substrs=[
                "bool-arg (set: True): False",
                "shlib-name (set: True): Something",
                "disk-file-name (set: False):",
                "line-num (set: False):",
                "enum-option (set: False):",
            ],
        )

        # Next try some completion tests:

        interp = self.dbg.GetCommandInterpreter()
        matches = lldb.SBStringList()
        descriptions = lldb.SBStringList()

        # First try an enum completion:
        num_completions = interp.HandleCompletionWithDescriptions(
            "no-args -e f", 12, 0, 1000, matches, descriptions
        )
        self.assertEqual(num_completions, 1, "Only one completion for foo")
        self.assertEqual(
            matches.GetSize(), 2, "The first element is the complete additional text"
        )
        self.assertEqual(
            matches.GetStringAtIndex(0), "oo ", "And we got the right extra characters"
        )
        self.assertEqual(
            matches.GetStringAtIndex(1), "foo", "And we got the right match"
        )
        self.assertEqual(
            descriptions.GetSize(), 2, "descriptions matche the return length"
        )
        # FIXME: we don't return descriptions for enum elements
        # self.assertEqual(descriptions.GetStringAtIndex(1), "does foo things", "And we got the right description")

        # Now try an internal completer, the on disk file one is handy:
        partial_name = os.path.join(source_dir, "test_")
        cmd_str = f"no-args -d '{partial_name}'"

        matches.Clear()
        descriptions.Clear()
        num_completions = interp.HandleCompletionWithDescriptions(
            cmd_str, len(cmd_str) - 1, 0, 1000, matches, descriptions
        )
        print(
            f"First: {matches.GetStringAtIndex(0)}\nSecond: {matches.GetStringAtIndex(1)}\nThird: {matches.GetStringAtIndex(2)}"
        )
        self.assertEqual(num_completions, 1, "Only one completion for source file")
        self.assertEqual(matches.GetSize(), 2, "The first element is the complete line")
        self.assertEqual(
            matches.GetStringAtIndex(0),
            "commands.py' ",
            "And we got the right extra characters",
        )
        self.assertEqual(
            matches.GetStringAtIndex(1), test_file_path, "And we got the right match"
        )
        self.assertEqual(
            descriptions.GetSize(), 2, "descriptions match the return length"
        )
        # FIXME: we don't return descriptions for enum elements
        # self.assertEqual(descriptions.GetStringAtIndex(1), "does foo things", "And we got the right description")

        # Try a command with arguments.
        # FIXME: It should be enough to define an argument and it's type to get the completer
        # wired up for that argument type if it is a known type. But that isn't wired up in the
        # command parser yet, so I don't have any tests for that.  We also don't currently check
        # that the arguments passed match the argument specifications, so here I just pass a couple
        # sets of arguments and make sure we get back what we put in:
        self.expect(
            "two-args 'First Argument' 'Second Argument'",
            substrs=["0: First Argument", "1: Second Argument"],
        )
