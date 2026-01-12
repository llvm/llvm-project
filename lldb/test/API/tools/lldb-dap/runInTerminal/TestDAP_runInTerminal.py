"""
Test lldb-dap runInTerminal reverse request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase
import os
import subprocess
import json


@skipIfBuildType(["debug"])
class TestDAP_runInTerminal(lldbdap_testcase.DAPTestCaseBase):
    def read_pid_message(self, fifo_file):
        with open(fifo_file, "r") as file:
            self.assertIn("pid", file.readline())

    @staticmethod
    def send_did_attach_message(fifo_file):
        with open(fifo_file, "w") as file:
            file.write(json.dumps({"kind": "didAttach"}) + "\n")

    @staticmethod
    def read_error_message(fifo_file):
        with open(fifo_file, "r") as file:
            return file.readline()

    @skipIfAsan
    @skipIfWindows
    def test_runInTerminal(self):
        """
        Tests the "runInTerminal" reverse request. It makes sure that the IDE can
        launch the inferior with the correct environment variables and arguments.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        self.build_and_launch(
            program, console="integratedTerminal", args=["foobar"], env=["FOO=bar"]
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
        self.assertEqual(counter, 1)

        # We verify we were able to set the launch arguments
        argc = int(self.dap_server.get_local_variable_value("argc"))
        self.assertEqual(argc, 2)

        argv1 = self.dap_server.request_evaluate("argv[1]")["body"]["result"]
        self.assertIn("foobar", argv1)

        # We verify we were able to set the environment
        env = self.dap_server.request_evaluate("foo")["body"]["result"]
        self.assertIn("bar", env)

        self.continue_to_exit()

    @skipIfAsan
    @skipIfWindows
    def test_runInTerminalWithObjectEnv(self):
        """
        Tests the "runInTerminal" reverse request. It makes sure that the IDE can
        launch the inferior with the correct environment variables using an object.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, console="integratedTerminal", env={"FOO": "BAR"})

        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "make sure we got a reverse request",
        )

        request = self.dap_server.reverse_requests[0]
        request_envs = request["arguments"]["env"]

        self.assertIn("FOO", request_envs)
        self.assertEqual("BAR", request_envs["FOO"])

        self.continue_to_exit()

    @skipIfWindows
    def test_runInTerminalInvalidTarget(self):
        self.build_and_create_debug_adapter()
        response = self.launch(
            "INVALIDPROGRAM",
            console="integratedTerminal",
            args=["foobar"],
            env=["FOO=bar"],
            waitForResponse=True,
        )
        self.assertFalse(response["success"])
        self.assertIn(
            "'INVALIDPROGRAM' does not exist",
            response["body"]["error"]["format"],
        )

    @skipIfWindows
    def test_missingArgInRunInTerminalLauncher(self):
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
    def test_FakeAttachedRunInTerminalLauncherWithInvalidProgram(self):
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

        self.read_pid_message(comm_file)
        self.send_did_attach_message(comm_file)
        self.assertIn("No such file or directory", self.read_error_message(comm_file))

        _, stderr = proc.communicate()
        self.assertIn("No such file or directory", stderr)

    @skipIfWindows
    def test_FakeAttachedRunInTerminalLauncherWithValidProgram(self):
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

        self.read_pid_message(comm_file)
        self.send_did_attach_message(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("foo", stdout)

    @skipIfWindows
    def test_FakeAttachedRunInTerminalLauncherAndCheckEnvironment(self):
        comm_file = os.path.join(self.getBuildDir(), "comm-file")
        os.mkfifo(comm_file)

        proc = subprocess.Popen(
            [self.lldbDAPExec, "--comm-file", comm_file, "--launch-target", "env"],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            env={**os.environ, "FOO": "BAR"},
        )

        self.read_pid_message(comm_file)
        self.send_did_attach_message(comm_file)

        stdout, _ = proc.communicate()
        self.assertIn("FOO=BAR", stdout)

    @skipIfWindows
    def test_NonAttachedRunInTerminalLauncher(self):
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

        self.read_pid_message(comm_file)

        _, stderr = proc.communicate()
        self.assertIn("Timed out trying to get messages from the debug adapter", stderr)
