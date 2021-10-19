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

    mydir = TestBase.compute_mydir(__file__)

    def raw_transcript_builder(self, cmd, res):
        raw = "(lldb) " + cmd + "\n"
        if res.GetOutputSize():
          raw += res.GetOutput()
        if res.GetErrorSize():
          raw += res.GetError()
        return raw


    @skipIfWindows
    @no_debug_info_test
    def test_session_save(self):
        raw = ""
        interpreter = self.dbg.GetCommandInterpreter()

        settings = [
          'settings set interpreter.echo-commands true',
          'settings set interpreter.echo-comment-commands true',
          'settings set interpreter.stop-command-source-on-error false'
        ]

        for setting in settings:
          interpreter.HandleCommand(setting, lldb.SBCommandReturnObject())

        inputs = [
          '# This is a comment',  # Comment
          'help session',         # Valid command
          'Lorem ipsum'           # Invalid command
        ]

        for cmd in inputs:
          res = lldb.SBCommandReturnObject()
          interpreter.HandleCommand(cmd, res)
          raw += self.raw_transcript_builder(cmd, res)

        self.assertTrue(interpreter.HasCommands())
        self.assertTrue(len(raw) != 0)

        # Check for error
        cmd = 'session save /root/file'
        interpreter.HandleCommand(cmd, res)
        self.assertFalse(res.Succeeded())
        raw += self.raw_transcript_builder(cmd, res)

        tf = tempfile.NamedTemporaryFile()
        output_file = tf.name

        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand('session save ' + output_file, res)
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
        interpreter.HandleCommand('settings set interpreter.save-session-directory ' + td.name, res)
        self.assertTrue(res.Succeeded())

        res = lldb.SBCommandReturnObject()
        interpreter.HandleCommand('session save', res)
        self.assertTrue(res.Succeeded())
        raw += self.raw_transcript_builder(cmd, res)

        with open(os.path.join(td.name, os.listdir(td.name)[0]), "r") as file:
          content = file.read()
          # Exclude last line, since session won't record it's own output
          lines = raw.splitlines()[:-1]
          for line in lines:
            self.assertIn(line, content)

    @skipIfWindows
    @no_debug_info_test
    def test_session_save_on_quit(self):
        raw = ""
        interpreter = self.dbg.GetCommandInterpreter()

        td = tempfile.TemporaryDirectory()

        settings = [
          'settings set interpreter.echo-commands true',
          'settings set interpreter.echo-comment-commands true',
          'settings set interpreter.stop-command-source-on-error false',
          'settings set interpreter.save-session-on-quit true',
          'settings set interpreter.save-session-directory ' + td.name,
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



