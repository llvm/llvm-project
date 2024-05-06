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

    def test_with_process_launch_api(self):
        """Test the SBCommandInterpreter APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Retrieve the associated command interpreter from our debugger.
        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

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

    def test_structured_transcript(self):
        """Test structured transcript generation and retrieval."""
        # Get command interpreter and create a target
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        # Send a few commands through the command interpreter
        res = lldb.SBCommandReturnObject()
        ci.HandleCommand("version", res)
        ci.HandleCommand("an-unknown-command", res)
        ci.HandleCommand("breakpoint set -f main.c -l %d" % self.line, res)
        ci.HandleCommand("r", res)
        ci.HandleCommand("p a", res)
        total_number_of_commands = 5

        # Retrieve the transcript and convert it into a Python object
        transcript = ci.GetTranscript()
        self.assertTrue(transcript.IsValid())

        stream = lldb.SBStream()
        self.assertTrue(stream)

        error = transcript.GetAsJSON(stream)
        self.assertSuccess(error)

        transcript = json.loads(stream.GetData())

        print('TRANSCRIPT')
        print(transcript)

        # The transcript will contain a bunch of commands that are from
        # a general setup code. See `def setUpCommands(cls)` in
        # `lldb/packages/Python/lldbsuite/test/lldbtest.py`.
        # https://shorturl.at/bJKVW
        #
        # We only want to validate for the ones that are listed above, hence
        # trimming to the last parts.
        transcript = transcript[-total_number_of_commands:]

        # Validate the transcript.
        #
        # The following asserts rely on the exact output format of the
        # commands. Hopefully we are not changing them any time soon.

        # (lldb) version
        self.assertEqual(transcript[0]["command"], "version")
        self.assertTrue("lldb version" in transcript[0]["output"][0])
        self.assertEqual(transcript[0]["error"], [])

        # (lldb) an-unknown-command
        self.assertEqual(transcript[1],
            {
                "command": "an-unknown-command",
                "output": [],
                "error": [
                    "error: 'an-unknown-command' is not a valid command.",
                ],
            })

        # (lldb) breakpoint set -f main.c -l <line>
        self.assertEqual(transcript[2]["command"], "breakpoint set -f main.c -l %d" % self.line)
        # Breakpoint 1: where = a.out`main + 29 at main.c:5:3, address = 0x0000000100000f7d
        self.assertTrue("Breakpoint 1: where = a.out`main " in transcript[2]["output"][0])
        self.assertEqual(transcript[2]["error"], [])

        # (lldb) r
        self.assertEqual(transcript[3]["command"], "r")
        # Process 25494 launched: '<path>/TestCommandInterpreterAPI.test_structured_transcript/a.out' (x86_64)
        self.assertTrue("Process" in transcript[3]["output"][0])
        self.assertTrue("launched" in transcript[3]["output"][0])
        self.assertEqual(transcript[3]["error"], [])

        # (lldb) p a
        self.assertEqual(transcript[4],
            {
                "command": "p a",
                "output": [
                    "(int) 123",
                ],
                "error": [],
            })
