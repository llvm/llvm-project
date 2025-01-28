"""
Test lldb-dap server integration.
"""

import os
import signal
import tempfile

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_server(lldbdap_testcase.DAPTestCaseBase):
    def start_server(self, connection):
        log_file_path = self.getBuildArtifact("dap.txt")
        server = dap_server.DebugAdaptorServer(
            executable=self.lldbDAPExec,
            connection=connection,
            init_commands=self.setUpCommands(),
            log_file=log_file_path,
        )

        def cleanup():
            server.terminate()

        self.addTearDownHook(cleanup)

        return server

    def run_debug_session(self, connection, name):
        self.dap_server = dap_server.DebugAdaptorServer(
            connection=connection,
        )
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        breakpoint_line = line_number(source, "// breakpoint")

        self.launch(
            program,
            args=[name],
            disconnectAutomatically=False,
        )
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()
        self.continue_to_exit()
        output = self.get_stdout()
        self.assertEqual(output, f"Hello {name}!\r\n")
        self.dap_server.request_disconnect()

    def test_server_port(self):
        """
        Test launching a binary with a lldb-dap in server mode on a specific port.
        """
        self.build()
        server = self.start_server(connection="tcp://localhost:0")
        self.run_debug_session(server.connection, "Alice")
        self.run_debug_session(server.connection, "Bob")

    @skipIfWindows
    def test_server_unix_socket(self):
        """
        Test launching a binary with a lldb-dap in server mode on a unix socket.
        """
        dir = tempfile.gettempdir()
        name = dir + "/dap-connection-" + str(os.getpid())

        def cleanup():
            os.unlink(name)

        self.addTearDownHook(cleanup)

        self.build()
        server = self.start_server(connection="unix://" + name)
        self.run_debug_session(server.connection, "Alice")
        self.run_debug_session(server.connection, "Bob")
