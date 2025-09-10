"""
Test lldb-dap server integration.
"""

import os
import signal
import tempfile
import time

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_server(lldbdap_testcase.DAPTestCaseBase):
    def start_server(
        self, connection, connection_timeout=None, wait_seconds_for_termination=None
    ):
        log_file_path = self.getBuildArtifact("dap.txt")
        (process, connection) = dap_server.DebugAdapterServer.launch(
            executable=self.lldbDAPExec,
            connection=connection,
            connection_timeout=connection_timeout,
            log_file=log_file_path,
        )

        def cleanup():
            if wait_seconds_for_termination is not None:
                process.wait(wait_seconds_for_termination)
            else:
                process.terminate()

        self.addTearDownHook(cleanup)

        return (process, connection)

    def run_debug_session(self, connection, name, sleep_seconds_in_middle=None):
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
        if sleep_seconds_in_middle is not None:
            time.sleep(sleep_seconds_in_middle)
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

    @skipIfWindows
    def test_connection_timeout_at_server_start(self):
        """
        Test launching lldb-dap in server mode with connection timeout and waiting for it to terminate automatically when no client connects.
        """
        self.build()
        self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
            wait_seconds_for_termination=2,
        )

    @skipIfWindows
    def test_connection_timeout_long_debug_session(self):
        """
        Test launching lldb-dap in server mode with connection timeout and terminating the server after the a long debug session.
        """
        self.build()
        (_, connection) = self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
            wait_seconds_for_termination=2,
        )
        # The connection timeout should not cut off the debug session
        self.run_debug_session(connection, "Alice", 1.5)

    @skipIfWindows
    def test_connection_timeout_multiple_sessions(self):
        """
        Test launching lldb-dap in server mode with connection timeout and terminating the server after the last debug session.
        """
        self.build()
        (_, connection) = self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
            wait_seconds_for_termination=2,
        )
        time.sleep(0.5)
        # Should be able to connect to the server.
        self.run_debug_session(connection, "Alice")
        time.sleep(0.5)
        # Should be able to connect to the server, because it's still within the connection timeout.
        self.run_debug_session(connection, "Bob")
