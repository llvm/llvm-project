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

    def setUp(self):
        TestBase.setUp(self)
        self.stdin_path = self.getBuildArtifact("stdin.txt")
        self.stdout_path = self.getBuildArtifact("stdout.txt")

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
        self.expect("help " + cmd_name, substrs=substrs)

    def run_one_repeat(self, commands, expected_num_errors):
        with open(self.stdin_path, "w") as input_handle:
            input_handle.write(commands)

        in_fileH = open(self.stdin_path, "r")
        self.dbg.SetInputFileHandle(in_fileH, False)

        out_fileH = open(self.stdout_path, "w")
        self.dbg.SetOutputFileHandle(out_fileH, False)
        self.dbg.SetErrorFileHandle(out_fileH, False)

        options = lldb.SBCommandInterpreterRunOptions()
        options.SetEchoCommands(False)
        options.SetPrintResults(True)
        options.SetPrintErrors(True)
        options.SetAllowRepeats(True)

        n_errors, quit_requested, has_crashed = self.dbg.RunCommandInterpreter(
            True, False, options, 0, False, False
        )

        in_fileH.close()
        out_fileH.close()

        results = None
        with open(self.stdout_path, "r") as out_fileH:
            results = out_fileH.read()

        self.assertEqual(n_errors, expected_num_errors)

        return results

    def handle_completion(
        self,
        cmd_str,
        exp_num_completions,
        exp_matches,
        exp_descriptions,
        match_description,
    ):
        matches = lldb.SBStringList()
        descriptions = lldb.SBStringList()

        interp = self.dbg.GetCommandInterpreter()
        num_completions = interp.HandleCompletionWithDescriptions(
            cmd_str, len(cmd_str), 0, 1000, matches, descriptions
        )
        self.assertEqual(
            num_completions, exp_num_completions, "Number of completions is right."
        )
        num_matches = matches.GetSize()
        self.assertEqual(
            num_matches,
            exp_matches.GetSize(),
            "matches and expected matches of different lengths",
        )
        num_descriptions = descriptions.GetSize()
        if match_description:
            self.assertEqual(
                num_descriptions,
                exp_descriptions.GetSize(),
                "descriptions and expected of different lengths",
            )

        self.assertEqual(
            matches.GetSize(),
            num_completions + 1,
            "The first element is the complete additional text",
        )

        for idx in range(0, num_matches):
            match = matches.GetStringAtIndex(idx)
            exp_match = exp_matches.GetStringAtIndex(idx)
            self.assertEqual(
                match, exp_match, f"{match} did not match expectation: {exp_match}"
            )
        if match_description:
            desc = descriptions.GetStringAtIndex(idx)
            exp_desc = exp_descriptions.GetStringAtIndex(idx)
            self.assertEqual(
                desc, exp_desc, f"{desc} didn't match expectation: {exp_desc}"
            )

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
        # Note - this is an enum so all the values are returned:
        matches.AppendList(["oo ", "foo"], 2)

        self.handle_completion("no-args -e f", 1, matches, descriptions, False)

        # Now try an internal completer, the on disk file one is handy:
        partial_name = os.path.join(source_dir, "test_")
        cmd_str = f"no-args -d '{partial_name}'"

        matches.Clear()
        descriptions.Clear()
        matches.AppendList(["commands.py' ", test_file_path], 2)
        # We don't have descriptions for the file path completer:
        self.handle_completion(cmd_str, 1, matches, descriptions, False)

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

        # Now test custom completions - two-args has both option and arg completers.  In both
        # completers we return different values if the -p option is set, so we can test that too:
        matches.Clear()
        descriptions.Clear()
        cmd_str = "two-args -p something -c other_"
        matches.AppendString("something ")
        matches.AppendString("other_something")
        # This is a full match so no descriptions:
        self.handle_completion(cmd_str, 1, matches, descriptions, False)

        matches.Clear()
        descriptions.Clear()
        cmd_str = "two-args -c other_"
        matches.AppendList(["", "other_nice", "other_not_nice", "other_mediocre"], 4)
        # The option doesn't return descriptions either:
        self.handle_completion(cmd_str, 3, matches, descriptions, False)

        # Now try the argument - it says "no completions" if the proc_name was set:
        matches.Clear()
        descriptions.Clear()
        cmd_str = "two-args -p something arg"
        matches.AppendString("")
        self.handle_completion(cmd_str, 0, matches, descriptions, False)

        cmd_str = "two-args arg_"
        matches.Clear()
        descriptions.Clear()
        matches.AppendList(["", "arg_cool", "arg_yuck"], 3)
        descriptions.AppendList(["", "good idea", "bad idea"], 3)
        self.handle_completion(cmd_str, 2, matches, descriptions, True)

        # This one gets a single unique match:
        cmd_str = "two-args correct_"
        matches.Clear()
        descriptions.Clear()
        matches.AppendList(["answer ", "correct_answer"], 2)
        self.handle_completion(cmd_str, 1, matches, descriptions, False)

        # Now make sure get_repeat_command works properly:

        # no-args turns off auto-repeat
        results = self.run_one_repeat("no-args\n\n", 1)
        self.assertIn("no auto repeat", results, "Got auto-repeat error")

        # one-args does the normal repeat
        results = self.run_one_repeat("one-arg-no-opt ONE_ARG\n\n", 0)
        self.assertEqual(results.count("ONE_ARG"), 2, "We did a normal repeat")

        # two-args adds an argument:
        results = self.run_one_repeat("two-args FIRST_ARG SECOND_ARG\n\n", 0)
        self.assertEqual(
            results.count("FIRST_ARG"), 2, "Passed first arg to both commands"
        )
        self.assertEqual(
            results.count("SECOND_ARG"), 2, "Passed second arg to both commands"
        )
        self.assertEqual(results.count("THIRD_ARG"), 1, "Passed third arg in repeat")
