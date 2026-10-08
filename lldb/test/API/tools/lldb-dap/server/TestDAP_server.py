"""
Test lldb-dap server integration.
"""

import os
import signal
import tempfile
import time

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.lldb_dap import *
from lldbsuite.test.tools.lldb_dap.types import *


@requireSocketPermission  # every test starts lldb-dap in listening server mode
class TestDAP_server(DAPTestCaseBase):
    USE_DEFAULT_DEBUG_ADAPTER = False

    @skipIfWindows
    def test_server_port(self):
        """
        Test launching a binary with a lldb-dap in server mode on a specific port.
        """
        self.build()
        adapter = self.start_server(connection="listen://localhost:0")
        # TODO: Ideally the sessions should run concurrently. But parsing the same module's
        #  debug info from multiple SBDebuggers simultaneously is currently not thread-safe.
        for name in ["Alice", "Bob"]:
            session = self.create_session(adapter, disconnect_automatically=False)
            self.run_debug_session(session, name)

    @requirePOSIX
    def test_server_unix_socket(self):
        """
        Test launching a binary with a lldb-dap in server mode on a unix socket.
        """
        self.build()
        # Unix socket paths are limited to about 104 bytes, and the default temp
        # directory on macOS can be long, so use /tmp when there is one.
        # The temp directory makes it unique "/tmp/tmp-dir9909ilskj32ewr/dap.sock"
        base_dir = "/tmp" if os.path.isdir("/tmp") else None
        temp_dir = tempfile.TemporaryDirectory(dir=base_dir)
        socket_path = os.path.join(temp_dir.name, "dap.sock")

        self.logger.info("adapter server socket path: %s", socket_path)
        self.addTearDownHook(temp_dir.cleanup)

        adapter = self.start_server(connection="accept://" + socket_path)
        # TODO: Ideally the sessions should run concurrently. But parsing the same module's
        #  debug info from multiple SBDebuggers simultaneously is currently not thread-safe.
        for name in ["Alice", "Bob"]:
            session = self.create_session(adapter, disconnect_automatically=False)
            self.run_debug_session(session, name)

    @skipIfWindows
    def test_server_interrupt(self):
        """
        Test launching a binary with lldb-dap in server mode and shutting down
        the server while the debug session is still active.
        """
        self.build()
        adapter = self.start_server(connection="listen://localhost:0")
        session = self.create_session(adapter, disconnect_automatically=False)
        stop_event = self.launch_to_breakpoint(session, "Alice")

        # Interrupt the server which should disconnect all clients.
        adapter.process.send_signal(signal.SIGINT)

        # Wait for both events since they can happen in any order.
        ending_events = (TerminatedEvent, ExitedEvent)
        first = session.wait_for_any_event(ending_events, after=stop_event)
        session.wait_for_any_event(ending_events, after=first)

        exit_code = int(signal.SIGKILL)
        session.verify_process_exited(exitCode=exit_code, after=stop_event)

        # The server shuts down cleanly once its clients are disconnected.
        self.assertEqual(adapter.process.wait(timeout=self.DEFAULT_TIMEOUT), 0)

    @skipIfWindows
    def test_connection_timeout_at_server_start(self):
        """
        Test launching lldb-dap in server mode with connection timeout and
        waiting for it to terminate automatically when no client connects.
        """
        self.build()
        adapter = self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
        )
        # The server must exit on its own, before the teardown hook kills it.
        self.assertEqual(adapter.process.wait(timeout=self.DEFAULT_TIMEOUT), 0)

    @skipIfWindows
    def test_connection_timeout_long_debug_session(self):
        """
        Test launching lldb-dap in server mode with connection timeout and
        terminating the server after the a long debug session.
        """
        self.build()
        adapter = self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
        )
        # The connection timeout should not cut off the debug session.
        session = self.create_session(adapter, disconnect_automatically=False)
        self.run_debug_session(session, "Alice", sleep_seconds_in_middle=1.2)
        self.assertTrue(adapter.is_alive, "expected the server to be running")

    @skipIfWindows
    def test_connection_timeout_multiple_sessions(self):
        """
        Test launching lldb-dap in server mode with connection timeout and
        terminating the server after the last debug session.
        """
        self.build()
        adapter = self.start_server(
            connection="listen://localhost:0",
            connection_timeout=1,
        )
        time.sleep(0.25)
        # Should be able to connect to the server.
        session1 = self.create_session(adapter, disconnect_automatically=False)
        self.run_debug_session(session1, "Alice")
        time.sleep(0.25)
        # Should be able to connect to the server, because it's still within the connection timeout.
        session2 = self.create_session(adapter, disconnect_automatically=False)
        self.run_debug_session(session2, "Bob")

        # The server shuts down once the connection timeout passes after the
        # last session ends.
        self.assertEqual(adapter.process.wait(timeout=self.DEFAULT_TIMEOUT), 0)

    @skipIfWindows
    def test_breakpoints_in_multiple_sessions(self):
        """
        Test in server mode setting a breakpoint in one session does not activate
        in another session.
        """
        self.build()
        program = self.getBuildArtifact("a.out")
        adapter = self.start_server(connection="listen://localhost:0")
        # With first breakpoint.
        session1 = self.create_session(adapter, disconnect_automatically=False)
        # With second breakpoint.
        session2 = self.create_session(adapter, disconnect_automatically=False)

        source = "main.c"
        bp1_line = line_number(source, "// breakpoint 1")
        launch_args = LaunchArgs(program)

        # Start the first session and stop at breakpoint 1.
        with session1.configure(launch_args) as ctx1:
            [breakpoint1] = session1.resolve_source_breakpoints(source, [bp1_line])

        session1.verify_stopped_on_breakpoint(breakpoint1, after=ctx1.process_event)

        # Start the second session and stop at breakpoint 2.
        bp2_line = line_number(source, "// breakpoint 2")
        with session2.configure(launch_args) as ctx2:
            [breakpoint2] = session2.resolve_source_breakpoints(source, [bp2_line])

        session2.verify_stopped_on_breakpoint(breakpoint2, after=ctx2.process_event)

        # Start and finish the third session with no breakpoint.
        session3 = self.create_session(adapter, disconnect_automatically=False)
        process_event3 = session3.launch(launch_args)
        session3.verify_process_exited(after=process_event3)

        # Finishing session1 and session2 should not hit any breakpoint.
        session1.continue_to_exit()
        session2.continue_to_exit()

    def start_server(self, connection: str, connection_timeout: int = 30):
        options = DebugAdapterOptions(
            connection=connection, connection_timeout=connection_timeout
        )
        return self.create_server_debug_adapter(options)

    def launch_to_breakpoint(self, session: DAPTestSession, name: str):
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        breakpoint_line = line_number(source, "// breakpoint 1")

        with session.configure(LaunchArgs(program, args=[name])) as ctx:
            session.resolve_source_breakpoints(source, [breakpoint_line])

        return session.verify_stopped_on_breakpoint(after=ctx.process_event)

    def continue_to_exit_and_disconnect(self, session: DAPTestSession, name: str):
        session.continue_to_exit()
        output = session.get_stdout()
        self.assertEqual(output, f"Hello {name}!\r\n")
        disconnect_resp = session.disconnect()

        with self.assertRaises(DAPError):
            # We should not receive a second terminated event. The check doesn't
            # stall, since the event history is closed.
            session.wait_for_terminated_event(after=disconnect_resp)

    def run_debug_session(
        self, session: DAPTestSession, name: str, *, sleep_seconds_in_middle: float = 0
    ):
        self.launch_to_breakpoint(session, name)
        if sleep_seconds_in_middle:
            time.sleep(sleep_seconds_in_middle)
        self.continue_to_exit_and_disconnect(session, name)
