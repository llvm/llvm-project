"""
Test lldb-dap "port" configuration to "attach" request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os
import shutil
import subprocess
import tempfile
import threading
import time
import sys

class TestDAP_attachByPortNum(lldbdap_testcase.DAPTestCaseBase):

    def runTargetProgramOnPort(self, port=None, program=None):
        server_tool = "lldb-server"
        server_path = self.getBuiltinServerTool(server_tool)
        if server_path:
            server_path += " g localhost:" +  port + " "

        self.process = subprocess.Popen(
            [server_path + program],
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return self.process

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

    @skipIfWindows
    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_port(self):
        """
        Tests attaching to a process by port.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = "2345"
        self.process = self.runTargetProgramOnPort(port=port, program=program)
        pid = self.process.pid
        response = self.attach_by_port(
            program=program, port=int(port), sourceInitFile=True
        )
        if not (response and response["success"]):
            self.assertTrue(
                response["success"], "attach failed (%s)" % (response["message"])
            )
        self.set_and_hit_breakpoint(continueToExit=True)
        self.process.kill()

    @skipIfWindows
    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_port_and_pid(self):
        """
        Tests attaching to a process by process ID and port number.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = "2345"
        self.process = self.runTargetProgramOnPort(port=port, program=program)
        response = self.attach_by_port(
            program=program, pid=1234, port=int(port), sourceInitFile=True
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"], "The user can't specify both pid and port"
            )
        self.process.kill()

    @skipIfWindows
    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_invalid_port(self):
        """
        Tests attaching to a process by invalid port number 0.
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = "0"
        self.process = self.runTargetProgramOnPort(port=port, program=program)
        response = self.attach_by_port(
            program=program, port=int(port), sourceInitFile=True
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"],
                "The user can't attach with invalid port (%s)" % port,
            )
        self.process.kill()

    @skipIfWindows
    @skipIfNetBSD  # Hangs on NetBSD as well
    @skipIfRemote
    def test_by_illegal_port(self):
        """
        Tests attaching to a process by illegal/greater port number 65536
        """
        self.build_and_create_debug_adaptor()
        program = self.getBuildArtifact("a.out")

        port = "65536"
        self.process = self.runTargetProgramOnPort(port=port, program=program)
        response = self.attach_by_port(
            program=program, port=int(port), sourceInitFile=True
        )
        if not (response and response["success"]):
            self.assertFalse(
                response["success"],
                "The user can't attach with illegal port (%s)" % port,
            )
        self.process.kill()
