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
        (process, connection) = dap_server.DebugAdapterServer.launch(
            executable=self.lldbDAPExec,
            connection=connection,
            log_file=log_file_path,
        )

        def cleanup():
            process.terminate()

        self.addTearDownHook(cleanup)

        return (process, connection)

    def run_debug_session(self, connection, name):
        self.dap_server = dap_server.DebugAdapterServer(
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

    @skipIfWindows
    def test_server_port(self):
        """
        Test launching a binary with a lldb-dap in server mode on a specific port.
        """
        self.build()
        (_, connection) = self.start_server(connection="listen://localhost:0")
        self.run_debug_session(connection, "Alice")
        self.run_debug_session(connection, "Bob")

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
        (_, connection) = self.start_server(connection="accept://" + name)
        self.run_debug_session(connection, "Alice")
        self.run_debug_session(connection, "Bob")

    @skipIfWindows
    def test_server_interrupt(self):
        """
        Test launching a binary with lldb-dap in server mode and shutting down the server while the debug session is still active.
        """
        self.build()
        (process, connection) = self.start_server(connection="listen://localhost:0")
        self.dap_server = dap_server.DebugAdapterServer(
            connection=connection,
        )
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        breakpoint_line = line_number(source, "// breakpoint")

        self.launch(
            program,
            args=["Alice"],
            disconnectAutomatically=False,
        )
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # Interrupt the server which should disconnect all clients.
        process.send_signal(signal.SIGINT)

        # Wait for both events since they can happen in any order.
        self.dap_server.wait_for_event(["terminated", "exited"])
        self.dap_server.wait_for_event(["terminated", "exited"])
        self.assertIsNotNone(
            self.dap_server.exit_status,
            "Process exited before interrupting lldb-dap server",
        )
