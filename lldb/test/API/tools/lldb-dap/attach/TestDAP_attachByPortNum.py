"""
Test lldb-dap "port" configuration to "attach" request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbplatformutil
from lldbgdbserverutils import Pipe
import lldbdap_testcase
import os
import shutil
import subprocess
import tempfile
import threading
import sys
import socket


class TestDAP_attachByPortNum(lldbdap_testcase.DAPTestCaseBase):
    default_timeout = 20

    def set_and_hit_breakpoint(self, continueToExit=True):
        source = "main.c"
        main_source_path = os.path.join(os.getcwd(), source)
        breakpoint1_line = line_number(main_source_path, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(main_source_path, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        if continueToExit:
            self.continue_to_exit()

    def get_debug_server_command_line_args(self):
        args = []
        if lldbplatformutil.getPlatform() == "linux":
            args = ["gdbserver"]
        elif lldbplatformutil.getPlatform() == "macosx":
            args = ["--listen"]
        if lldb.remote_platform:
            args += ["*:0"]
        else:
            args += ["localhost:0"]
        return args

    def get_debug_server_pipe(self):
        pipe = Pipe(self.getBuildDir())
        self.addTearDownHook(lambda: pipe.close())
        pipe.finish_connection(self.default_timeout)
        return pipe

    @skipIfWindows
    @skipIfNetBSD
    def test_by_port(self):
        """
        Tests attaching to a process by port.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        debug_server_tool = self.getBuiltinDebugServerTool()

        pipe = self.get_debug_server_pipe()
        args = self.get_debug_server_command_line_args()
        args += [program]
        args += ["--named-pipe", pipe.name]

        self.process = self.spawnSubprocess(
            debug_server_tool, args, install_remote=False
        )

        # Read the port number from the debug server pipe.
        port = pipe.read(10, self.default_timeout)
        # Trim null byte, convert to int
        port = int(port[:-1])
        self.assertIsNotNone(
            port, " Failed to read the port number from debug server pipe"
        )

        self.attach(program=program, gdbRemotePort=port, sourceInitFile=True)
        self.set_and_hit_breakpoint(continueToExit=True)
        self.process.terminate()

    @skipIfWindows
    @skipIfNetBSD
    def test_by_port_and_pid(self):
        """
        Tests attaching to a process by process ID and port number.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        # It is not necessary to launch "lldb-server" to obtain the actual port and pid for attaching.
        # However, when providing the port number and pid directly, "lldb-dap" throws an error message, which is expected.
        # So, used random pid and port numbers here.

        pid = 1354
        port = 1234

        response = self.attach(
            program=program,
            pid=pid,
            gdbRemotePort=port,
            sourceInitFile=True,
            expectFailure=True,
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"], "The user can't specify both pid and port"
            )

    @skipIfWindows
    @skipIfNetBSD
    def test_by_invalid_port(self):
        """
        Tests attaching to a process by invalid port number 0.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = 0
        response = self.attach(
            program=program, gdbRemotePort=port, sourceInitFile=True, expectFailure=True
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"],
                "The user can't attach with invalid port (%s)" % port,
            )

    @skipIfWindows
    @skipIfNetBSD
    def test_by_illegal_port(self):
        """
        Tests attaching to a process by illegal/greater port number 65536
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = 65536
        args = [program]
        debug_server_tool = self.getBuiltinDebugServerTool()
        self.process = self.spawnSubprocess(
            debug_server_tool, args, install_remote=False
        )

        response = self.attach(
            program=program, gdbRemotePort=port, sourceInitFile=True, expectFailure=True
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"],
                "The user can't attach with illegal port (%s)" % port,
            )
        self.process.terminate()
