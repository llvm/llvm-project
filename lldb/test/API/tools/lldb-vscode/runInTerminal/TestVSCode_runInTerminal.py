"""
Test lldb-vscode runInTerminal reverse request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import time
import os
import subprocess
import shutil
import json
from threading import Thread


class TestVSCode_runInTerminal(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def readPidMessage(self, fifo_file):
        with open(fifo_file, "r") as file:
            self.assertIn("pid", file.readline())

    def sendDidAttachMessage(self, fifo_file):
        with open(fifo_file, "w") as file:
            file.write(json.dumps({"kind": "didAttach"}) + "\n")

    def readErrorMessage(self, fifo_file):
        with open(fifo_file, "r") as file:
            return file.readline()

    @skipIfWindows
    @skipIfRemote
    def test_runInTerminal(self):
        '''
            Tests the "runInTerminal" reverse request. It makes sure that the IDE can
            launch the inferior with the correct environment variables and arguments.
        '''
        if "debug" in str(os.environ["LLDBVSCODE_EXEC"]).lower():
            # We skip this test for debug builds because it takes too long parsing lldb's own
            # debug info. Release builds are fine.
            # Checking this environment variable seems to be a decent proxy for a quick
            # detection
            return
        program = self.getBuildArtifact("a.out")
        source = 'main.c'
        self.build_and_launch(
            program, stopOnEntry=True, runInTerminal=True, args=["foobar"],
            env=["FOO=bar"])

        breakpoint_line = line_number(source, '// breakpoint')

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # We verify we actually stopped inside the loop
        counter = int(self.vscode.get_local_variable_value('counter'))
        self.assertTrue(counter > 0)

        # We verify we were able to set the launch arguments
        argc = int(self.vscode.get_local_variable_value('argc'))
        self.assertEqual(argc, 2)

        argv1 = self.vscode.request_evaluate('argv[1]')['body']['result']
        self.assertIn('foobar', argv1)

        # We verify we were able to set the environment
        env = self.vscode.request_evaluate('foo')['body']['result']
        self.assertIn('bar', env)

    @skipIfWindows
    @skipIfRemote
    def test_runInTerminalInvalidTarget(self):
        self.build_and_create_debug_adaptor()
        response = self.launch(
            "INVALIDPROGRAM", stopOnEntry=True, runInTerminal=True, args=["foobar"], env=["FOO=bar"], expectFailure=True)
        self.assertFalse(response['success'])
        self.assertIn("Could not create a target for a program 'INVALIDPROGRAM': unable to find executable",
            response['message'])

    @skipIfWindows
    @skipIfRemote
    def test_missingArgInRunInTerminalLauncher(self):
        proc = subprocess.run([self.lldbVSCodeExec,  "--launch-target", "INVALIDPROGRAM"],
            capture_output=True, universal_newlines=True)
        self.assertTrue(proc.returncode != 0)
        self.assertIn('"--launch-target" requires "--comm-file" to be specified', proc.stderr)

    @skipIfWindows
    @skipIfRemote
    def test_FakeAttachedRunInTerminalLauncherWithInvalidProgram(self):
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbVSCodeExec, "--comm-file", comm_file, "--launch-target", "INVALIDPROGRAM"],
            universal_newlines=True, stderr=subprocess.PIPE)

        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)
        self.assertIn("No such file or directory", self.readErrorMessage(comm_file))
        
        _, stderr = proc.communicate()
        self.assertIn("No such file or directory", stderr)

    @skipIfWindows
    @skipIfRemote
    def test_FakeAttachedRunInTerminalLauncherWithValidProgram(self):
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbVSCodeExec, "--comm-file", comm_file, "--launch-target", "echo", "foo"],
            universal_newlines=True, stdout=subprocess.PIPE)

        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("foo", stdout)

    @skipIfWindows
    @skipIfRemote
    def test_FakeAttachedRunInTerminalLauncherAndCheckEnvironment(self):
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbVSCodeExec, "--comm-file", comm_file, "--launch-target", "env"],
            universal_newlines=True, stdout=subprocess.PIPE,
            env={**os.environ, "FOO": "BAR"})
        
        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("FOO=BAR", stdout)

    @skipIfWindows
    @skipIfRemote
    def test_NonAttachedRunInTerminalLauncher(self):
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbVSCodeExec, "--comm-file", comm_file, "--launch-target", "echo", "foo"],
            universal_newlines=True, stderr=subprocess.PIPE,
            env={**os.environ, "LLDB_VSCODE_RIT_TIMEOUT_IN_MS": "1000"})

        self.readPidMessage(comm_file)
        
        _, stderr = proc.communicate()
        self.assertIn("Timed out trying to get messages from the debug adaptor", stderr)
