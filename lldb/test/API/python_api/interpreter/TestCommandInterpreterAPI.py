"""Test the SBCommandInterpreter APIs."""

import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CommandInterpreterAPICase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number("main.c", "Hello world.")

    def buildAndCreateTarget(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)
        return ci

    def test_with_process_launch_api(self):
        """Test the SBCommandInterpreter APIs."""
        ci = self.buildAndCreateTarget()

        # Exercise some APIs....

        self.assertTrue(ci.HasCommands())
        self.assertTrue(ci.HasAliases())
        self.assertTrue(ci.HasAliasOptions())
        self.assertTrue(ci.CommandExists("breakpoint"))
        self.assertTrue(ci.CommandExists("target"))
        self.assertTrue(ci.CommandExists("platform"))
        self.assertTrue(ci.AliasExists("file"))
        self.assertTrue(ci.AliasExists("run"))
        self.assertTrue(ci.AliasExists("bt"))

        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("breakpoint set -f main.c -l %d" % self.line, res)
        self.assertTrue(res.Succeeded())
        ci.HandleCommand("process launch", res)
        self.assertTrue(res.Succeeded())

        # Boundary conditions should not crash lldb!
        self.assertFalse(ci.CommandExists(None))
        self.assertFalse(ci.AliasExists(None))
        ci.HandleCommand(None, res)
        self.assertFalse(res.Succeeded())
        res.AppendMessage("Just appended a message.")
        res.AppendMessage(None)
        if self.TraceOn():
            print(res)

        process = ci.GetProcess()
        self.assertTrue(process)

        import lldbsuite.test.lldbutil as lldbutil

        if process.GetState() != lldb.eStateStopped:
            self.fail(
                "Process should be in the 'stopped' state, "
                "instead the actual state is: '%s'"
                % lldbutil.state_type_to_str(process.GetState())
            )

        if self.TraceOn():
            lldbutil.print_stacktraces(process)

    def test_command_output(self):
        """Test command output handling."""
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Test that a command which produces no output returns "" instead of
        # None.
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("settings set use-color false", res)
        self.assertTrue(res.Succeeded())
        self.assertIsNotNone(res.GetOutput())
        self.assertEqual(res.GetOutput(), "")
        self.assertIsNotNone(res.GetError())
        self.assertEqual(res.GetError(), "")

    def getTranscriptAsPythonObject(self, ci):
        """Retrieve the transcript and convert it into a Python object"""
        structured_data = ci.GetTranscript()
        self.assertTrue(structured_data.IsValid())

        stream = lldb.SBStream()
        self.assertTrue(stream)

        error = structured_data.GetAsJSON(stream)
        self.assertSuccess(error)

        return json.loads(stream.GetData())

    def test_get_transcript(self):
        """Test structured transcript generation and retrieval."""
        ci = self.buildAndCreateTarget()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Make sure the "save-transcript" setting is on
        self.runCmd("settings set interpreter.save-transcript true")

        # Send a few commands through the command interpreter.
        #
        # Using `ci.HandleCommand` because some commands will fail so that we
        # can test the "error" field in the saved transcript.
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("version", res)
        ci.HandleCommand("an-unknown-command", res)
        ci.HandleCommand("br s -f main.c -l %d" % self.line, res)
        ci.HandleCommand("p a", res)
        ci.HandleCommand("statistics dump", res)
        total_number_of_commands = 6

        # Get transcript as python object
        transcript = self.getTranscriptAsPythonObject(ci)

        # All commands should have expected fields.
        for command in transcript:
            self.assertIn("command", command)
            # Unresolved commands don't have "commandName"/"commandArguments".
            # We will validate these fields below, instead of here.
            self.assertIn("output", command)
            self.assertIn("error", command)
            self.assertIn("durationInSeconds", command)
            self.assertIn("timestampInEpochSeconds", command)

        # The following validates individual commands in the transcript.
        #
        # Notes:
        # 1. Some of the asserts rely on the exact output format of the
        #    commands. Hopefully we are not changing them any time soon.
        # 2. We are removing the time-related fields from each command, so
        #    that some of the validations below can be easier / more readable.
        for command in transcript:
            del command["durationInSeconds"]
            del command["timestampInEpochSeconds"]

        # (lldb) version
        self.assertEqual(transcript[0]["command"], "version")
        self.assertEqual(transcript[0]["commandName"], "version")
        self.assertEqual(transcript[0]["commandArguments"], "")
        self.assertIn("lldb version", transcript[0]["output"])
        self.assertEqual(transcript[0]["error"], "")

        # (lldb) an-unknown-command
        self.assertEqual(transcript[1],
            {
                "command": "an-unknown-command",
                # Unresolved commands don't have "commandName"/"commandArguments"
                "output": "",
                "error": "error: 'an-unknown-command' is not a valid command.\n",
            })

        # (lldb) br s -f main.c -l <line>
        self.assertEqual(transcript[2]["command"], "br s -f main.c -l %d" % self.line)
        self.assertEqual(transcript[2]["commandName"], "breakpoint set")
        self.assertEqual(
            transcript[2]["commandArguments"], "-f main.c -l %d" % self.line
        )
        # Breakpoint 1: where = a.out`main + 29 at main.c:5:3, address = 0x0000000100000f7d
        self.assertIn("Breakpoint 1: where = a.out`main ", transcript[2]["output"])
        self.assertEqual(transcript[2]["error"], "")

        # (lldb) p a
        self.assertEqual(transcript[3],
            {
                "command": "p a",
                "commandName": "dwim-print",
                "commandArguments": "-- a",
                "output": "",
                "error": "error: <user expression 0>:1:1: use of undeclared identifier 'a'\n    1 | a\n      | ^\n",
            })

        # (lldb) statistics dump
        self.assertEqual(transcript[4]["command"], "statistics dump")
        self.assertEqual(transcript[4]["commandName"], "statistics dump")
        self.assertEqual(transcript[4]["commandArguments"], "")
        self.assertEqual(transcript[4]["error"], "")
        statistics_dump = json.loads(transcript[4]["output"])
        # Dump result should be valid JSON
        self.assertTrue(statistics_dump is not json.JSONDecodeError)
        # Dump result should contain expected fields
        self.assertIn("commands", statistics_dump)
        self.assertIn("memory", statistics_dump)
        self.assertIn("modules", statistics_dump)
        self.assertIn("targets", statistics_dump)

    def test_save_transcript_setting_default(self):
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # The setting's default value should be "false"
        self.runCmd("settings show interpreter.save-transcript", "interpreter.save-transcript (boolean) = false\n")

    def test_save_transcript_setting_off(self):
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Make sure the setting is off
        self.runCmd("settings set interpreter.save-transcript false")

        # The transcript should be empty after running a command
        self.runCmd("version")
        transcript = self.getTranscriptAsPythonObject(ci)
        self.assertEqual(transcript, [])

    def test_save_transcript_setting_on(self):
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Make sure the setting is on
        self.runCmd("settings set interpreter.save-transcript true")

        # The transcript should contain one item after running a command
        self.runCmd("version")
        transcript = self.getTranscriptAsPythonObject(ci)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["command"], "version")

    def test_get_transcript_returns_copy(self):
        """
        Test that the returned structured data is *at least* a shallow copy.

        We believe that a deep copy *is* performed in `SBCommandInterpreter::GetTranscript`.
        However, the deep copy cannot be tested and doesn't need to be tested,
        because there is no logic in the command interpreter to modify a
        transcript item (representing a command) after it has been returned.
        """
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Make sure the setting is on
        self.runCmd("settings set interpreter.save-transcript true")

        # Run commands and get the transcript as structured data
        self.runCmd("version")
        structured_data_1 = ci.GetTranscript()
        self.assertTrue(structured_data_1.IsValid())
        self.assertEqual(structured_data_1.GetSize(), 1)
        self.assertEqual(structured_data_1.GetItemAtIndex(0).GetValueForKey("command").GetStringValue(100), "version")

        # Run some more commands and get the transcript as structured data again
        self.runCmd("help")
        structured_data_2 = ci.GetTranscript()
        self.assertTrue(structured_data_2.IsValid())
        self.assertEqual(structured_data_2.GetSize(), 2)
        self.assertEqual(structured_data_2.GetItemAtIndex(0).GetValueForKey("command").GetStringValue(100), "version")
        self.assertEqual(structured_data_2.GetItemAtIndex(1).GetValueForKey("command").GetStringValue(100), "help")

        # Now, the first structured data should remain unchanged
        self.assertTrue(structured_data_1.IsValid())
        self.assertEqual(structured_data_1.GetSize(), 1)
        self.assertEqual(structured_data_1.GetItemAtIndex(0).GetValueForKey("command").GetStringValue(100), "version")
