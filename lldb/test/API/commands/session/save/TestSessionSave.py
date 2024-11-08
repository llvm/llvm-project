"""
Test the session save feature
"""
import os
import tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SessionSaveTestCase(TestBase):
    def raw_transcript_builder(self, cmd, res):
        raw = "(lldb) " + cmd + "\n"
        if res.GetOutputSize():
            raw += res.GetOutput()
        if res.GetErrorSize():
            raw += res.GetError()
        return raw

    @no_debug_info_test
    def test_session_save(self):
        raw = ""
        interpreter = self.dbg.GetCommandInterpreter()

        # Make sure "save-transcript" is on, so that all the following setings
        # and commands are saved into the trasncript. Note that this cannot be
        # a part of the `settings`, because this command itself won't be saved
        # into the transcript.
        self.runCmd("settings set interpreter.save-transcript true")

        settings = [
            "settings set interpreter.echo-commands true",
            "settings set interpreter.echo-comment-commands true",
            "settings set interpreter.stop-command-source-on-error false",
            "settings set interpreter.open-transcript-in-editor false",
        ]

        for setting in settings:
            interpreter.HandleCommand(setting, lldb.SBCommandReturnObject())

        inputs = [
            "# This is a comment",  # Comment
            "help session",  # Valid command
            "Lorem ipsum",  # Invalid command
        ]

        for cmd in inputs:
            res = lldb.SBCommandReturnObject()
            interpreter.HandleCommand(cmd, res)
            raw += self.raw_transcript_builder(cmd, res)

        self.assertTrue(interpreter.HasCommands())
        self.assertNotEqual(len(raw), 0)

        # Check for error
        cmd = "session save /root/file"
        interpreter.HandleCommand(cmd, res)
        self.assertFalse(res.Succeeded())
        raw += self.raw_transcript_builder(cmd, res)

        output_file = self.getBuildArtifact('my-session')

        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("session save " + output_file, res)
        self.assertTrue(res.Succeeded())
        raw += self.raw_transcript_builder(cmd, res)

        with open(output_file, "r") as file:
            content = file.read()
            # Exclude last line, since session won't record it's own output
            lines = raw.splitlines()[:-1]
            for line in lines:
                self.assertIn(line, content)

        td = tempfile.TemporaryDirectory()
        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand(
            "settings set interpreter.save-session-directory " + td.name, res
        )
        self.assertTrue(res.Succeeded())

        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("session save", res)
        self.assertTrue(res.Succeeded())
        raw += self.raw_transcript_builder(cmd, res)
        # Also check that we don't print an error message about an empty transcript.
        self.assertNotIn("interpreter.save-transcript is set to false", res.GetError())

        with open(os.path.join(td.name, os.listdir(td.name)[0]), "r") as file:
            content = file.read()
            # Exclude last line, since session won't record it's own output
            lines = raw.splitlines()[:-1]
            for line in lines:
                self.assertIn(line, content)

    @no_debug_info_test
    def test_session_save_no_transcript_warning(self):
        interpreter = self.dbg.GetCommandInterpreter()

        self.runCmd("settings set interpreter.save-transcript false")

        # These commands won't be saved, so are arbitrary.
        commands = [
            "settings set interpreter.open-transcript-in-editor false",
            "p 1",
            "settings set interpreter.save-session-on-quit true",
            "fr v",
            "settings set interpreter.echo-comment-commands true",
        ]

        for command in commands:
            res = lldb.SBCommandReturnObject()
            interpreter.HandleCommand(command, res)

        output_file = self.getBuildArtifact("my-session")

        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("session save " + output_file, res)
        self.assertTrue(res.Succeeded())
        # We should warn about the setting being false.
        self.assertIn("interpreter.save-transcript is set to false", res.GetError())
        self.assertTrue(
            os.path.getsize(output_file) == 0,
            "Output file should be empty since we didn't save the transcript.",
        )

    @no_debug_info_test
    def test_session_save_on_quit(self):
        raw = ""
        interpreter = self.dbg.GetCommandInterpreter()

        # Make sure "save-transcript" is on, so that all the following setings
        # and commands are saved into the trasncript. Note that this cannot be
        # a part of the `settings`, because this command itself won't be saved
        # into the transcript.
        self.runCmd("settings set interpreter.save-transcript true")

        td = tempfile.TemporaryDirectory()

        settings = [
            "settings set interpreter.echo-commands true",
            "settings set interpreter.echo-comment-commands true",
            "settings set interpreter.stop-command-source-on-error false",
            "settings set interpreter.save-session-on-quit true",
            "settings set interpreter.save-session-directory " + td.name,
            "settings set interpreter.open-transcript-in-editor false",
        ]

        for setting in settings:
            res = lldb.SBCommandReturnObject()
            interpreter.HandleCommand(setting, res)
            raw += self.raw_transcript_builder(setting, res)

        self.dbg.Destroy(self.dbg)

        with open(os.path.join(td.name, os.listdir(td.name)[0]), "r") as file:
            content = file.read()
            # Exclude last line, since session won't record it's own output
            lines = raw.splitlines()[:-1]
            for line in lines:
                self.assertIn(line, content)
