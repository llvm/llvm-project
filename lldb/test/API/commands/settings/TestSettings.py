"""
Test lldb settings command.
"""

import json
import os
import re

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class SettingsCommandTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_apropos_should_also_search_settings_description(self):
        """Test that 'apropos' command should also search descriptions for the settings variables."""

        self.expect(
            "apropos 'environment variable'",
            substrs=[
                "target.env-vars",
                "environment variables",
                "executable's environment",
            ],
        )

    def test_set_interpreter_repeat_prev_command(self):
        """Test the `interpreter.repeat-previous-command` setting."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        setting = "interpreter.repeat-previous-command"

        def cleanup(setting):
            self.runCmd("settings clear %s" % setting, check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup(setting))

        # First, check for the setting default value.
        self.expect(
            "setting show %s" % setting,
            substrs=["interpreter.repeat-previous-command (boolean) = true"],
        )

        # Then, invert the setting, and check that was set correctly
        self.runCmd("setting set %s false" % setting)
        self.expect(
            "setting show %s" % setting,
            substrs=["interpreter.repeat-previous-command (boolean) = false"],
        )

        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci.IsValid(), "Invalid command interpreter.")
        # Now, test the functionnality
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("breakpoint set -n main", res)
        self.assertTrue(res.Succeeded(), "Command failed.")
        ci.HandleCommand("", res)
        self.assertTrue(res.Succeeded(), "Empty command failed.")
        self.assertEqual(self.dbg.GetSelectedTarget().GetNumBreakpoints(), 1)

    def test_append_target_env_vars(self):
        """Test that 'append target.env-vars' works."""
        # Append the env-vars.
        self.runCmd("settings append target.env-vars MY_ENV_VAR=YES")
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd("settings clear target.env-vars"))

        # Check it immediately!
        self.expect("settings show target.env-vars", substrs=["MY_ENV_VAR=YES"])

    def test_insert_before_and_after_target_run_args(self):
        """Test that 'insert-before/after target.run-args' works."""
        # Set the run-args first.
        self.runCmd("settings set target.run-args a b c")
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd("settings clear target.run-args"))

        # Now insert-before the index-0 element with '__a__'.
        self.runCmd("settings insert-before target.run-args 0 __a__")
        # And insert-after the index-1 element with '__A__'.
        self.runCmd("settings insert-after target.run-args 1 __A__")
        # Check it immediately!
        self.expect(
            "settings show target.run-args",
            substrs=[
                "target.run-args",
                '[0]: "__a__"',
                '[1]: "a"',
                '[2]: "__A__"',
                '[3]: "b"',
                '[4]: "c"',
            ],
        )

    def test_replace_target_run_args(self):
        """Test that 'replace target.run-args' works."""
        # Set the run-args and then replace the index-0 element.
        self.runCmd("settings set target.run-args a b c")
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd("settings clear target.run-args"))

        # Now replace the index-0 element with 'A', instead.
        self.runCmd("settings replace target.run-args 0 A")
        # Check it immediately!
        self.expect(
            "settings show target.run-args",
            substrs=[
                "target.run-args (arguments) =",
                '[0]: "A"',
                '[1]: "b"',
                '[2]: "c"',
            ],
        )

    def test_set_prompt(self):
        """Test that 'set prompt' actually changes the prompt."""

        # Set prompt to 'lldb2'.
        self.runCmd("settings set prompt 'lldb2 '")

        # Immediately test the setting.
        self.expect(
            "settings show prompt",
            SETTING_MSG("prompt"),
            startstr='prompt (string) = "lldb2 "',
        )

        # The overall display should also reflect the new setting.
        self.expect(
            "settings show",
            SETTING_MSG("prompt"),
            substrs=['prompt (string) = "lldb2 "'],
        )

        # Use '-r' option to reset to the original default prompt.
        self.runCmd("settings clear prompt")

    def test_set_term_width(self):
        """Test that 'set term-width' actually changes the term-width."""

        self.runCmd("settings set term-width 70")

        # Immediately test the setting.
        self.expect(
            "settings show term-width",
            SETTING_MSG("term-width"),
            startstr="term-width (unsigned) = 70",
        )

        # The overall display should also reflect the new setting.
        self.expect(
            "settings show",
            SETTING_MSG("term-width"),
            substrs=["term-width (unsigned) = 70"],
        )

        self.dbg.SetTerminalWidth(60)

        self.expect(
            "settings show",
            SETTING_MSG("term-width"),
            substrs=["term-width (unsigned) = 60"],
        )

    def test_set_frame_format(self):
        """Test that 'set frame-format' with a backtick char in the format string works as well as fullpath."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        def cleanup():
            self.runCmd(
                "settings set frame-format %s" % self.format_string, check=False
            )

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("settings show frame-format")
        m = re.match(r'^frame-format \(format-string\) = "(.*)"$', self.res.GetOutput())
        self.assertTrue(m, "Bad settings string")
        self.format_string = m.group(1)

        # Change the default format to print function.name rather than
        # function.name-with-args
        format_string = "frame #${frame.index}: ${frame.pc}{ ${module.file.basename}\\`${function.name}{${function.pc-offset}}}{ at ${line.file.fullpath}:${line.number}}{, lang=${language}}\n"
        self.runCmd("settings set frame-format %s" % format_string)

        # Immediately test the setting.
        self.expect(
            "settings show frame-format",
            SETTING_MSG("frame-format"),
            substrs=[format_string],
        )

        self.runCmd("breakpoint set -n main")
        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )
        self.expect("thread backtrace", substrs=["`main", self.getSourceDir()])

    def test_set_auto_confirm(self):
        """Test that after 'set auto-confirm true', manual confirmation should not kick in."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("settings set auto-confirm true")

        # Immediately test the setting.
        self.expect(
            "settings show auto-confirm",
            SETTING_MSG("auto-confirm"),
            startstr="auto-confirm (boolean) = true",
        )

        # Now 'breakpoint delete' should just work fine without confirmation
        # prompt from the command interpreter.
        self.runCmd("breakpoint set -n main")
        self.expect("breakpoint delete", startstr="All breakpoints removed")

        # Restore the original setting of auto-confirm.
        self.runCmd("settings clear auto-confirm")
        self.expect(
            "settings show auto-confirm",
            SETTING_MSG("auto-confirm"),
            startstr="auto-confirm (boolean) = false",
        )

    @skipIf(archs=no_match(["x86_64", "i386", "i686"]))
    def test_disassembler_settings(self):
        """Test that user options for the disassembler take effect."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # AT&T syntax
        self.runCmd("settings set target.x86-disassembly-flavor att")
        self.runCmd("settings set target.use-hex-immediates false")
        self.expect("disassemble -n numberfn", substrs=["$90"])
        self.runCmd("settings set target.use-hex-immediates true")
        self.runCmd("settings set target.hex-immediate-style c")
        self.expect("disassemble -n numberfn", substrs=["$0x5a"])
        self.runCmd("settings set target.hex-immediate-style asm")
        self.expect("disassemble -n numberfn", substrs=["$5ah"])

        # Intel syntax
        self.runCmd("settings set target.x86-disassembly-flavor intel")
        self.runCmd("settings set target.use-hex-immediates false")
        self.expect("disassemble -n numberfn", substrs=["90"])
        self.runCmd("settings set target.use-hex-immediates true")
        self.runCmd("settings set target.hex-immediate-style c")
        self.expect("disassemble -n numberfn", substrs=["0x5a"])
        self.runCmd("settings set target.hex-immediate-style asm")
        self.expect("disassemble -n numberfn", substrs=["5ah"])

    @skipIfDarwinEmbedded  # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_run_args_and_env_vars(self):
        self.do_test_run_args_and_env_vars(use_launchsimple=False)

    @skipIfDarwinEmbedded  # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_launchsimple_args_and_env_vars(self):
        self.do_test_run_args_and_env_vars(use_launchsimple=True)

    def do_test_run_args_and_env_vars(self, use_launchsimple):
        """Test that run-args and env-vars are passed to the launched process."""
        self.build()

        # Set the run-args and the env-vars.
        # And add hooks to restore the settings during tearDown().
        self.runCmd("settings set target.run-args A B C")
        self.addTearDownHook(lambda: self.runCmd("settings clear target.run-args"))
        self.runCmd('settings set target.env-vars ["MY_ENV_VAR"]=YES')
        self.addTearDownHook(lambda: self.runCmd("settings clear target.env-vars"))

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.GetTargetAtIndex(0)
        launch_info = target.GetLaunchInfo()
        found_env_var = False
        for i in range(0, launch_info.GetNumEnvironmentEntries()):
            if launch_info.GetEnvironmentEntryAtIndex(i) == "MY_ENV_VAR=YES":
                found_env_var = True
                break
        self.assertTrue(found_env_var, "MY_ENV_VAR was not set in LaunchInfo object")

        self.assertEqual(launch_info.GetNumArguments(), 3)
        self.assertEqual(launch_info.GetArgumentAtIndex(0), "A")
        self.assertEqual(launch_info.GetArgumentAtIndex(1), "B")
        self.assertEqual(launch_info.GetArgumentAtIndex(2), "C")

        self.expect("target show-launch-environment", substrs=["MY_ENV_VAR=YES"])

        wd = self.get_process_working_directory()
        if use_launchsimple:
            process = target.LaunchSimple(None, None, wd)
            self.assertTrue(process)
        else:
            self.runCmd("process launch --working-dir '{0}'".format(wd), RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output2.txt")

        self.expect(
            output,
            exe=False,
            substrs=[
                "argv[1] matches",
                "argv[2] matches",
                "argv[3] matches",
                "Environment variable 'MY_ENV_VAR' successfully passed.",
            ],
        )

        # Check that env-vars overrides unset-env-vars.
        self.runCmd("settings set target.unset-env-vars MY_ENV_VAR")

        self.expect(
            "target show-launch-environment",
            "env-vars overrides unset-env-vars",
            substrs=["MY_ENV_VAR=YES"],
        )

        wd = self.get_process_working_directory()
        if use_launchsimple:
            process = target.LaunchSimple(None, None, wd)
            self.assertTrue(process)
        else:
            self.runCmd("process launch --working-dir '{0}'".format(wd), RUN_SUCCEEDED)

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output2.txt")

        self.expect(
            output,
            exe=False,
            substrs=["Environment variable 'MY_ENV_VAR' successfully passed."],
        )

    @skipIfRemote  # it doesn't make sense to send host env to remote target
    def test_pass_host_env_vars(self):
        """Test that the host env vars are passed to the launched process."""
        self.build()

        # Set some host environment variables now.
        os.environ["MY_HOST_ENV_VAR1"] = "VAR1"
        os.environ["MY_HOST_ENV_VAR2"] = "VAR2"

        # This is the function to unset the two env variables set above.
        def unset_env_variables():
            os.environ.pop("MY_HOST_ENV_VAR1")
            os.environ.pop("MY_HOST_ENV_VAR2")

        self.addTearDownHook(unset_env_variables)

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # By default, inherit-env is 'true'.
        self.expect(
            "settings show target.inherit-env",
            "Default inherit-env is 'true'",
            startstr="target.inherit-env (boolean) = true",
        )

        self.expect(
            "target show-launch-environment",
            "Host environment is passed correctly",
            substrs=["MY_HOST_ENV_VAR1=VAR1", "MY_HOST_ENV_VAR2=VAR2"],
        )
        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output1.txt")

        self.expect(
            output,
            exe=False,
            substrs=[
                "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.",
                "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed.",
            ],
        )

        # Now test that we can prevent the inferior from inheriting the
        # environment.
        self.runCmd("settings set target.inherit-env false")

        self.expect(
            "target show-launch-environment",
            "target.inherit-env affects `target show-launch-environment`",
            matching=False,
            substrs=["MY_HOST_ENV_VAR1=VAR1", "MY_HOST_ENV_VAR2=VAR2"],
        )

        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output1.txt")

        self.expect(
            output,
            exe=False,
            matching=False,
            substrs=[
                "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed.",
                "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed.",
            ],
        )

        # Now test that we can unset variables from the inherited environment.
        self.runCmd("settings set target.inherit-env true")
        self.runCmd("settings set target.unset-env-vars MY_HOST_ENV_VAR1")
        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )

        # Read the output file produced by running the program.
        output = lldbutil.read_file_from_process_wd(self, "output1.txt")

        self.expect(
            "target show-launch-environment",
            "MY_HOST_ENV_VAR1 is unset, it shouldn't be in `target show-launch-environment`",
            matching=False,
            substrs=["MY_HOST_ENV_VAR1=VAR1"],
        )
        self.expect(
            "target show-launch-environment",
            "MY_HOST_ENV_VAR2 shouldn be in `target show-launch-environment`",
            substrs=["MY_HOST_ENV_VAR2=VAR2"],
        )

        self.expect(
            output,
            exe=False,
            matching=False,
            substrs=[
                "The host environment variable 'MY_HOST_ENV_VAR1' successfully passed."
            ],
        )
        self.expect(
            output,
            exe=False,
            substrs=[
                "The host environment variable 'MY_HOST_ENV_VAR2' successfully passed."
            ],
        )

    @skipIfDarwinEmbedded  # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_set_error_output_path(self):
        """Test that setting target.error/output-path for the launched process works."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the error-path and output-path and verify both are set.
        self.runCmd(
            "settings set target.error-path '{0}'".format(
                lldbutil.append_to_process_working_directory(self, "stderr.txt")
            )
        )
        self.runCmd(
            "settings set target.output-path '{0}".format(
                lldbutil.append_to_process_working_directory(self, "stdout.txt")
            )
        )
        # And add hooks to restore the original settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd("settings clear target.output-path"))
        self.addTearDownHook(lambda: self.runCmd("settings clear target.error-path"))

        self.expect(
            "settings show target.error-path",
            SETTING_MSG("target.error-path"),
            substrs=["target.error-path (file)", 'stderr.txt"'],
        )

        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            substrs=["target.output-path (file)", 'stdout.txt"'],
        )

        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )

        output = lldbutil.read_file_from_process_wd(self, "stderr.txt")
        message = "This message should go to standard error."
        if lldbplatformutil.hasChattyStderr(self):
            self.expect(output, exe=False, substrs=[message])
        else:
            self.expect(output, exe=False, startstr=message)

        output = lldbutil.read_file_from_process_wd(self, "stdout.txt")
        self.expect(
            output, exe=False, startstr="This message should go to standard out."
        )

    @skipIfDarwinEmbedded  # <rdar://problem/34446098> debugserver on ios etc can't write files
    def test_same_error_output_path(self):
        """Test that setting target.error and output-path to the same file path for the launched process works."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set the error-path and output-path and verify both are set.
        self.runCmd(
            "settings set target.error-path '{0}'".format(
                lldbutil.append_to_process_working_directory(self, "output.txt")
            )
        )
        self.runCmd(
            "settings set target.output-path '{0}".format(
                lldbutil.append_to_process_working_directory(self, "output.txt")
            )
        )
        # And add hooks to restore the original settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd("settings clear target.output-path"))
        self.addTearDownHook(lambda: self.runCmd("settings clear target.error-path"))

        self.expect(
            "settings show target.error-path",
            SETTING_MSG("target.error-path"),
            substrs=["target.error-path (file)", 'output.txt"'],
        )

        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            substrs=["target.output-path (file)", 'output.txt"'],
        )

        self.runCmd(
            "process launch --working-dir '{0}'".format(
                self.get_process_working_directory()
            ),
            RUN_SUCCEEDED,
        )

        output = lldbutil.read_file_from_process_wd(self, "output.txt")
        err_message = "This message should go to standard error."
        out_message = "This message should go to standard out."
        # Error msg should get flushed by the output msg
        self.expect(output, exe=False, substrs=[out_message])
        self.assertNotIn(
            err_message,
            output,
            "Race condition when both stderr/stdout redirects to the same file",
        )

    def test_print_dictionary_setting(self):
        self.runCmd("settings clear target.env-vars")
        self.runCmd('settings set target.env-vars ["MY_VAR"]=some-value')
        self.expect("settings show target.env-vars", substrs=["MY_VAR=some-value"])
        self.runCmd("settings clear target.env-vars")

    def test_print_array_setting(self):
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings set target.run-args gobbledy-gook")
        self.expect("settings show target.run-args", substrs=['[0]: "gobbledy-gook"'])
        self.runCmd("settings clear target.run-args")

    def test_settings_with_quotes(self):
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings set target.run-args a b c")
        self.expect(
            "settings show target.run-args",
            substrs=['[0]: "a"', '[1]: "b"', '[2]: "c"'],
        )
        self.runCmd("settings set target.run-args 'a b c'")
        self.expect("settings show target.run-args", substrs=['[0]: "a b c"'])
        self.runCmd("settings clear target.run-args")
        self.runCmd("settings clear target.env-vars")
        self.runCmd(
            'settings set target.env-vars ["MY_FILE"]="this is a file name with spaces.txt"'
        )
        self.expect(
            "settings show target.env-vars",
            substrs=["MY_FILE=this is a file name with spaces.txt"],
        )
        self.runCmd("settings clear target.env-vars")
        # Test and make sure that setting "format-string" settings obeys quotes
        # if they are provided
        self.runCmd("settings set thread-format    'abc def'   ")
        self.expect(
            "settings show thread-format",
            startstr='thread-format (format-string) = "abc def"',
        )
        self.runCmd('settings set thread-format    "abc def"   ')
        self.expect(
            "settings show thread-format",
            startstr='thread-format (format-string) = "abc def"',
        )
        # Make sure when no quotes are provided that we maintain any trailing
        # spaces
        self.runCmd("settings set thread-format abc def   ")
        self.expect(
            "settings show thread-format",
            startstr='thread-format (format-string) = "abc def   "',
        )
        self.runCmd("settings clear thread-format")

    def test_settings_with_trailing_whitespace(self):
        # boolean
        # Set to known value
        self.runCmd("settings set target.skip-prologue true")
        # Set to new value with trailing whitespace
        self.runCmd("settings set target.skip-prologue false ")
        # Make sure the setting was correctly set to "false"
        self.expect(
            "settings show target.skip-prologue",
            SETTING_MSG("target.skip-prologue"),
            startstr="target.skip-prologue (boolean) = false",
        )
        self.runCmd("settings clear target.skip-prologue", check=False)
        # integer
        self.runCmd("settings set term-width 70")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set term-width 60 \t")
        self.expect(
            "settings show term-width",
            SETTING_MSG("term-width"),
            startstr="term-width (unsigned) = 60",
        )
        self.runCmd("settings clear term-width", check=False)
        # string
        self.runCmd("settings set target.arg0 abc")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.arg0 cde\t ")
        self.expect(
            "settings show target.arg0",
            SETTING_MSG("target.arg0"),
            startstr='target.arg0 (string) = "cde"',
        )
        self.runCmd("settings clear target.arg0", check=False)
        # file
        path1 = self.getBuildArtifact("path1.txt")
        path2 = self.getBuildArtifact("path2.txt")
        self.runCmd("settings set target.output-path %s" % path1)  # Set to known value
        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            startstr="target.output-path (file) = ",
            substrs=[path1],
        )
        self.runCmd(
            "settings set target.output-path %s " % path2
        )  # Set to new value with trailing whitespaces
        self.expect(
            "settings show target.output-path",
            SETTING_MSG("target.output-path"),
            startstr="target.output-path (file) = ",
            substrs=[path2],
        )
        self.runCmd("settings clear target.output-path", check=False)
        # enum
        # Set to known value
        self.runCmd("settings set stop-disassembly-display never")
        # Set to new value with trailing whitespaces
        self.runCmd("settings set stop-disassembly-display always ")
        self.expect(
            "settings show stop-disassembly-display",
            SETTING_MSG("stop-disassembly-display"),
            startstr="stop-disassembly-display (enum) = always",
        )
        self.runCmd("settings clear stop-disassembly-display", check=False)
        # language
        # Set to known value
        self.runCmd("settings set target.language c89")
        # Set to new value with trailing whitespace
        self.runCmd("settings set target.language c11 ")
        self.expect(
            "settings show target.language",
            SETTING_MSG("target.language"),
            startstr="target.language (language) = c11",
        )
        self.runCmd("settings clear target.language", check=False)
        # arguments
        self.runCmd("settings set target.run-args 1 2 3")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.run-args 3 4 5 ")
        self.expect(
            "settings show target.run-args",
            SETTING_MSG("target.run-args"),
            substrs=[
                "target.run-args (arguments) =",
                '[0]: "3"',
                '[1]: "4"',
                '[2]: "5"',
            ],
        )
        self.runCmd("settings set target.run-args 1 2 3")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd(r"settings set target.run-args 3 \  \ ")
        self.expect(
            "settings show target.run-args",
            SETTING_MSG("target.run-args"),
            substrs=[
                "target.run-args (arguments) =",
                '[0]: "3"',
                '[1]: " "',
                '[2]: " "',
            ],
        )
        self.runCmd("settings clear target.run-args", check=False)
        # dictionaries
        self.runCmd("settings clear target.env-vars")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.env-vars A=B C=D\t ")
        self.expect(
            "settings show target.env-vars",
            SETTING_MSG("target.env-vars"),
            substrs=["target.env-vars (dictionary of strings) =", "A=B", "C=D"],
        )
        self.runCmd("settings clear target.env-vars", check=False)
        # regex
        # Set to known value
        self.runCmd("settings clear target.process.thread.step-avoid-regexp")
        # Set to new value with trailing whitespaces
        self.runCmd("settings set target.process.thread.step-avoid-regexp foo\\ ")
        self.expect(
            "settings show target.process.thread.step-avoid-regexp",
            SETTING_MSG("target.process.thread.step-avoid-regexp"),
            substrs=["target.process.thread.step-avoid-regexp (regex) = foo\\ "],
        )
        self.runCmd(
            "settings clear target.process.thread.step-avoid-regexp", check=False
        )
        # format-string
        self.runCmd("settings clear disassembly-format")  # Set to known value
        # Set to new value with trailing whitespaces
        self.runCmd("settings set disassembly-format foo ")
        self.expect(
            "settings show disassembly-format",
            SETTING_MSG("disassembly-format"),
            substrs=['disassembly-format (format-string) = "foo "'],
        )
        self.runCmd("settings clear disassembly-format", check=False)

    def test_settings_list(self):
        # List settings (and optionally test the filter to only show 'target' settings).
        self.expect(
            "settings list target", substrs=["arg0", "detach-on-error", "language"]
        )
        self.expect("settings list target", matching=False, substrs=["packet-timeout"])
        self.expect(
            "settings list",
            substrs=["language", "arg0", "detach-on-error", "packet-timeout"],
        )

    def test_settings_remove_single(self):
        # Set some environment variables and use 'remove' to delete them.
        self.runCmd("settings set target.env-vars a=b c=d")
        self.expect("settings show target.env-vars", substrs=["a=b", "c=d"])
        self.runCmd("settings remove target.env-vars a")
        self.expect("settings show target.env-vars", matching=False, substrs=["a=b"])
        self.expect("settings show target.env-vars", substrs=["c=d"])
        self.runCmd("settings remove target.env-vars c")
        self.expect(
            "settings show target.env-vars", matching=False, substrs=["a=b", "c=d"]
        )

    def test_settings_remove_multiple(self):
        self.runCmd("settings set target.env-vars a=b c=d e=f")
        self.expect("settings show target.env-vars", substrs=["a=b", "c=d", "e=f"])
        self.runCmd("settings remove target.env-vars a e")
        self.expect(
            "settings show target.env-vars", matching=False, substrs=["a=b", "e=f"]
        )
        self.expect("settings show target.env-vars", substrs=["c=d"])

    def test_settings_remove_nonexistent_value(self):
        self.expect(
            "settings remove target.env-vars doesntexist",
            error=True,
            substrs=["no value found named 'doesntexist'"],
        )

    def test_settings_remove_nonexistent_settings(self):
        self.expect(
            "settings remove doesntexist alsodoesntexist",
            error=True,
            substrs=["error: invalid value path 'doesntexist'"],
        )

    def test_settings_remove_missing_arg(self):
        self.expect(
            "settings remove",
            error=True,
            substrs=["'settings remove' takes an array or dictionary item, or"],
        )

    def test_settings_remove_empty_arg(self):
        self.expect(
            "settings remove ''",
            error=True,
            substrs=["'settings remove' command requires a valid variable name"],
        )

    def test_settings_clear_all(self):
        # Change a dictionary.
        self.runCmd("settings set target.env-vars a=1 b=2 c=3")
        # Change an array.
        self.runCmd("settings set target.run-args a1 b2 c3")
        # Change a single boolean value.
        self.runCmd("settings set auto-confirm true")
        # Change a single integer value.
        self.runCmd("settings set tab-size 4")

        # Clear everything.
        self.runCmd("settings clear --all")

        # Check that settings have their default values after clearing.
        self.expect(
            "settings show target.env-vars",
            patterns=[r"^target.env-vars \(dictionary of strings\) =\s*$"],
        )
        self.expect(
            "settings show target.run-args",
            patterns=[r"^target.run-args \(arguments\) =\s*$"],
        )
        self.expect("settings show auto-confirm", substrs=["false"])
        self.expect("settings show tab-size", substrs=["2"])

        # Check that the command fails if we combine '--all' option with any arguments.
        self.expect(
            "settings clear --all auto-confirm",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["'settings clear --all' doesn't take any arguments"],
        )

    def test_all_settings_exist(self):
        self.expect(
            "settings show",
            substrs=[
                "auto-confirm",
                "frame-format",
                "notify-void",
                "prompt",
                "script-lang",
                "stop-disassembly-count",
                "stop-disassembly-display",
                "stop-line-count-after",
                "stop-line-count-before",
                "stop-show-column",
                "term-width",
                "thread-format",
                "use-external-editor",
                "target.breakpoints-use-platform-avoid-list",
                "target.default-arch",
                "target.disable-aslr",
                "target.disable-stdio",
                "target.x86-disassembly-flavor",
                "target.enable-synthetic-value",
                "target.env-vars",
                "target.error-path",
                "target.exec-search-paths",
                "target.expr-prefix",
                "target.hex-immediate-style",
                "target.inherit-env",
                "target.input-path",
                "target.language",
                "target.max-children-count",
                "target.max-string-summary-length",
                "target.move-to-nearest-code",
                "target.output-path",
                "target.prefer-dynamic-value",
                "target.run-args",
                "target.skip-prologue",
                "target.source-map",
                "target.use-hex-immediates",
                "target.process.disable-memory-cache",
                "target.process.extra-startup-command",
                "target.process.track-memory-cache-changes",
                "target.process.thread.trace-thread",
                "target.process.thread.step-avoid-regexp",
            ],
        )

    # settings under an ".experimental" domain should have two properties:
    #   1. If the name does not exist with "experimental" in the name path,
    #      the name lookup should try to find it without "experimental".  So
    #      a previously-experimental setting that has been promoted to a
    #      "real" setting will still be set by the original name.
    #   2. Changing a setting with .experimental., name, where the setting
    #      does not exist either with ".experimental." or without, should
    #      not generate an error.  So if an experimental setting is removed,
    #      people who may have that in their ~/.lldbinit files should not see
    #      any errors.
    def test_experimental_settings(self):
        cmdinterp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Set target.arg0 to a known value, check that we can retrieve it via
        # the actual name and via .experimental.
        self.expect("settings set target.arg0 first-value")
        self.expect("settings show target.arg0", substrs=["first-value"])
        self.expect(
            "settings show target.experimental.arg0",
            substrs=["first-value"],
            error=False,
        )

        # Set target.arg0 to a new value via a target.experimental.arg0 name,
        # verify that we can read it back via both .experimental., and not.
        self.expect("settings set target.experimental.arg0 second-value", error=False)
        self.expect("settings show target.arg0", substrs=["second-value"])
        self.expect(
            "settings show target.experimental.arg0",
            substrs=["second-value"],
            error=False,
        )

        # showing & setting an undefined .experimental. setting should generate no errors.
        self.expect(
            "settings show target.experimental.setting-which-does-not-exist",
            patterns=[r"^\s$"],
            error=False,
        )
        self.expect(
            "settings set target.experimental.setting-which-does-not-exist true",
            error=False,
        )

        # A domain component before .experimental. which does not exist should give an error
        # But the code does not yet do that.
        # self.expect('settings set target.setting-which-does-not-exist.experimental.arg0 true', error=True)

        # finally, confirm that trying to set a setting that does not exist still fails.
        # (SHOWING a setting that does not exist does not currently yield an error.)
        self.expect("settings set target.setting-which-does-not-exist true", error=True)

    def test_settings_set_exists(self):
        cmdinterp = self.dbg.GetCommandInterpreter()

        # An unknown option should succeed.
        self.expect("settings set -e foo bar")
        self.expect("settings set --exists foo bar")

        # A known option should fail if its argument is invalid.
        self.expect("settings set auto-confirm bogus", error=True)

    def test_settings_show_defaults(self):
        # boolean
        self.expect(
            "settings show --defaults auto-one-line-summaries",
            matching=False,
            substrs=["(default: true)"],
        )
        self.runCmd("settings set auto-one-line-summaries false")
        self.expect(
            "settings show --defaults auto-one-line-summaries",
            substrs=["= false (default: true)"],
        )
        # unsigned
        self.expect(
            "settings show --defaults stop-line-count-before",
            matching=False,
            patterns=[r"\(default: \d+\)"],
        )
        self.runCmd("settings set stop-line-count-before 99")
        self.expect(
            "settings show --defaults stop-line-count-before",
            patterns=[r"= 99 \(default: \d+\)"],
        )
        # string
        self.expect(
            "settings show --defaults prompt",
            matching=False,
            patterns=[r'\(default: ".+"\)'],
        )
        self.runCmd("settings set prompt '<LlDb> '")
        self.expect(
            "settings show --defaults prompt",
            patterns=[r'= "<LlDb> " \(default: ".+"\)'],
        )
        # enum
        self.expect(
            "settings show --defaults stop-disassembly-display",
            matching=False,
            patterns=[r"\(default: .+\)"],
        )
        self.runCmd("settings set stop-disassembly-display no-source")
        self.expect(
            "settings show --defaults stop-disassembly-display",
            patterns=[r"= no-source \(default: .+\)"],
        )
        # regex
        self.expect(
            "settings show --defaults target.process.thread.step-avoid-regexp",
            matching=False,
            patterns=[r"\(default: .+\)"],
        )
        self.runCmd("settings set target.process.thread.step-avoid-regexp dotstar")
        self.expect(
            "settings show --defaults target.process.thread.step-avoid-regexp",
            patterns=[r"= dotstar \(default: .+\)"],
        )
        # format-string
        self.expect(
            "settings show --defaults disassembly-format",
            matching=False,
            patterns=[r'\(default: ".+"\)'],
        )
        self.runCmd("settings set disassembly-format dollar")
        self.expect(
            "settings show --defaults disassembly-format",
            patterns=[r'= "dollar" \(default: ".+"\)'],
        )
        # arrays
        self.expect(
            "settings show --defaults target.unset-env-vars",
            matching=False,
            substrs=["(default: empty)"],
        )
        self.runCmd("settings set target.unset-env-vars PATH")
        self.expect(
            "settings show --defaults target.unset-env-vars",
            substrs=["(default: empty)", '[0]: "PATH"'],
        )
        # dictionaries
        self.runCmd("settings clear target.env-vars")
        self.expect(
            "settings show --defaults target.env-vars",
            matching=False,
            substrs=["(default: empty)"],
        )
        self.runCmd("settings set target.env-vars THING=value")
        self.expect(
            "settings show --defaults target.env-vars",
            substrs=["(default: empty)", "THING=value"],
        )
        pwd = os.getcwd()
        # file list
        self.expect(
            "settings show --defaults target.exec-search-paths",
            matching=False,
            substrs=["(default: empty)"],
        )
        self.runCmd(f"settings set target.exec-search-paths {pwd}")
        self.expect(
            "settings show --defaults target.exec-search-paths",
            substrs=["(default: empty)", f"[0]: {pwd}"],
        )
        # path map
        self.expect(
            "settings show --defaults target.source-map",
            matching=False,
            substrs=["(default: empty)"],
        )
        self.runCmd(f"settings set target.source-map /abc {pwd}")
        self.expect(
            "settings show --defaults target.source-map",
            patterns=[
                r"\(default: empty\)",
                rf'\[0\] "[/\\]abc" -> "{re.escape(pwd)}"',
            ],
        )

    def get_setting_json(self, setting_path=None):
        settings_data = self.dbg.GetSetting(setting_path)
        stream = lldb.SBStream()
        settings_data.GetAsJSON(stream)
        return json.loads(stream.GetData())

    def verify_setting_value_json(self, setting_path, setting_value):
        self.runCmd("settings set %s %s" % (setting_path, setting_value))
        settings_json = self.get_setting_json(setting_path)
        self.assertEqual(settings_json, setting_value)

    def test_settings_api(self):
        """
        Test that ensures SBDebugger::GetSetting() APIs
        can correctly fetch settings.
        """

        # Test basic values and embedding special JSON escaping characters.
        self.runCmd("settings set auto-confirm true")
        self.runCmd("settings set tab-size 4")
        arg_value = 'hello "world"'
        self.runCmd("settings set target.arg0 %s" % arg_value)

        settings_json = self.get_setting_json()
        self.assertEqual(settings_json["auto-confirm"], True)
        self.assertEqual(settings_json["tab-size"], 4)
        self.assertEqual(settings_json["target"]["arg0"], arg_value)

        settings_data = self.get_setting_json("target.arg0")
        self.assertEqual(settings_data, arg_value)

        # Test OptionValueFileSpec
        self.verify_setting_value_json(
            "platform.module-cache-directory", self.getBuildDir()
        )

        # Test OptionValueArray
        setting_path = "target.run-args"
        setting_value = ["value1", "value2", "value3"]
        self.runCmd("settings set %s %s" % (setting_path, " ".join(setting_value)))
        settings_json = self.get_setting_json(setting_path)
        self.assertEqual(settings_json, setting_value)

        # Test OptionValueFileSpec and OptionValueFileSpecList
        setting_path = "target.debug-file-search-paths"
        path1 = os.path.join(self.getSourceDir(), "tmp")
        path2 = os.path.join(self.getSourceDir(), "tmp2")
        self.runCmd("settings set %s '%s' '%s'" % (setting_path, path1, path2))
        settings_json = self.get_setting_json(setting_path)
        self.assertEqual(settings_json, [path1, path2])

        # Test OptionValueFormatEntity
        setting_value = """thread #${thread.index}{, name = \\'${thread.name}\\
        '}{, queue = ${ansi.fg.green}\\'${thread.queue}\\'${ansi.normal}}{,
        activity = ${ansi.fg.green}\\'${thread.info.activity.name}\\'${ansi.normal}}
        {, ${thread.info.trace_messages} messages}{, stop reason = ${ansi.fg.red}$
        {thread.stop-reason}${ansi.normal}}{\\\\nReturn value: ${thread.return-value}}
        {\\\\nCompleted expression: ${thread.completed-expression}}\\\\n"""
        self.verify_setting_value_json("thread-stop-format", setting_value)

        # Test OptionValueRegex
        self.verify_setting_value_json(
            "target.process.thread.step-avoid-regexp", "^std::"
        )

        # Test OptionValueLanguage
        self.verify_setting_value_json("repl-lang", "c++")

        # Test OptionValueEnumeration
        self.verify_setting_value_json("target.x86-disassembly-flavor", "intel")

        # Test OptionValueArch
        self.verify_setting_value_json("target.default-arch", "x86_64")
        self.runCmd("settings clear target.default-arch")

    def test_global_option(self):
        # This command used to crash the settings because -g was signaled by a
        # NULL execution context (not one with an empty Target...) and in the
        # special handling for load-script-from-symbol-file this wasn't checked.
        self.runCmd("settings set -g target.load-script-from-symbol-file true")
