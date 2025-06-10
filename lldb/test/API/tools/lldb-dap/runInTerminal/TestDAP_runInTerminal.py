"""
Test lldb-dap runInTerminal reverse request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import time
import os
import subprocess
import shutil
import json
from threading import Thread


class TestDAP_runInTerminal(lldbdap_testcase.DAPTestCaseBase):
    def readPidMessage(self, fifo_file):
        with open(fifo_file, "r") as file:
            self.assertIn("pid", file.readline())

    def sendDidAttachMessage(self, fifo_file):
        with open(fifo_file, "w") as file:
            file.write(json.dumps({"kind": "didAttach"}) + "\n")

    def readErrorMessage(self, fifo_file):
        with open(fifo_file, "r") as file:
            return file.readline()

    def isTestSupported(self):
        # For some strange reason, this test fails on python3.6
        if not (sys.version_info.major == 3 and sys.version_info.minor >= 7):
            return False
        try:
            # We skip this test for debug builds because it takes too long parsing lldb's own
            # debug info. Release builds are fine.
            # Checking the size of the lldb-dap binary seems to be a decent proxy for a quick
            # detection. It should be far less than 1 MB in Release builds.
            if os.path.getsize(os.environ["LLDBDAP_EXEC"]) < 1000000:
                return True
        except:
            return False

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_runInTerminal(self):
        if not self.isTestSupported():
            return
        """
            Tests the "runInTerminal" reverse request. It makes sure that the IDE can
            launch the inferior with the correct environment variables and arguments.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        self.build_and_launch(
            program, runInTerminal=True, args=["foobar"], env=["FOO=bar"]
        )

        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "make sure we got a reverse request",
        )

        request = self.dap_server.reverse_requests[0]
        self.assertIn(self.lldbDAPExec, request["arguments"]["args"])
        self.assertIn(program, request["arguments"]["args"])
        self.assertIn("foobar", request["arguments"]["args"])
        self.assertIn("FOO", request["arguments"]["env"])

        breakpoint_line = line_number(source, "// breakpoint")

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # We verify we actually stopped inside the loop
        counter = int(self.dap_server.get_local_variable_value("counter"))
        self.assertGreater(counter, 0)

        # We verify we were able to set the launch arguments
        argc = int(self.dap_server.get_local_variable_value("argc"))
        self.assertEqual(argc, 2)

        argv1 = self.dap_server.request_evaluate("argv[1]")["body"]["result"]
        self.assertIn("foobar", argv1)

        # We verify we were able to set the environment
        env = self.dap_server.request_evaluate("foo")["body"]["result"]
        self.assertIn("bar", env)

    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_runInTerminalWithObjectEnv(self):
        if not self.isTestSupported():
            return
        """
            Tests the "runInTerminal" reverse request. It makes sure that the IDE can
            launch the inferior with the correct environment variables using an object.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, runInTerminal=True, env={"FOO": "BAR"})

        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "make sure we got a reverse request",
        )

        request = self.dap_server.reverse_requests[0]
        request_envs = request["arguments"]["env"]

        self.assertIn("FOO", request_envs)
        self.assertEqual("BAR", request_envs["FOO"])

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_runInTerminalInvalidTarget(self):
        if not self.isTestSupported():
            return
        self.build_and_create_debug_adapter()
        response = self.launch(
            "INVALIDPROGRAM",
            runInTerminal=True,
            args=["foobar"],
            env=["FOO=bar"],
            expectFailure=True,
        )
        self.assertFalse(response["success"])
        self.assertIn(
            "'INVALIDPROGRAM' does not exist",
            response["body"]["error"]["format"],
        )

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_missingArgInRunInTerminalLauncher(self):
        if not self.isTestSupported():
            return
        proc = subprocess.run(
            [self.lldbDAPExec, "--launch-target", "INVALIDPROGRAM"],
            capture_output=True,
            universal_newlines=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            '"--launch-target" requires "--comm-file" to be specified', proc.stderr
        )

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_FakeAttachedRunInTerminalLauncherWithInvalidProgram(self):
        if not self.isTestSupported():
            return
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [
                self.lldbDAPExec,
                "--comm-file",
                comm_file,
                "--launch-target",
                "INVALIDPROGRAM",
            ],
            universal_newlines=True,
            stderr=subprocess.PIPE,
        )

        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)
        self.assertIn("No such file or directory", self.readErrorMessage(comm_file))

        _, stderr = proc.communicate()
        self.assertIn("No such file or directory", stderr)

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_FakeAttachedRunInTerminalLauncherWithValidProgram(self):
        if not self.isTestSupported():
            return
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [
                self.lldbDAPExec,
                "--comm-file",
                comm_file,
                "--launch-target",
                "echo",
                "foo",
            ],
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )

        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("foo", stdout)

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_FakeAttachedRunInTerminalLauncherAndCheckEnvironment(self):
        if not self.isTestSupported():
            return
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbDAPExec, "--comm-file", comm_file, "--launch-target", "env"],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            env={**os.environ, "FOO": "BAR"},
        )

        self.readPidMessage(comm_file)
        self.sendDidAttachMessage(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("FOO=BAR", stdout)

    @skipIfWindows
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    def test_NonAttachedRunInTerminalLauncher(self):
        if not self.isTestSupported():
            return
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [
                self.lldbDAPExec,
                "--comm-file",
                comm_file,
                "--launch-target",
                "echo",
                "foo",
            ],
            universal_newlines=True,
            stderr=subprocess.PIPE,
            env={**os.environ, "LLDB_DAP_RIT_TIMEOUT_IN_MS": "1000"},
        )

        self.readPidMessage(comm_file)

        _, stderr = proc.communicate()
        self.assertIn("Timed out trying to get messages from the debug adapter", stderr)
