"""
Test reuse lldb-dap adapter across debug sessions.
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import os
import time

import lldbdap_testcase
from lldbsuite.test import lldbutil


class TestDAP_reuseAdapter(lldbdap_testcase.DAPTestCaseBase):
    def start_server(self, connection):
        """Start lldb-dap in server mode.
        
        This helper method follows the same pattern as TestDAP_server.py.
        It launches lldb-dap in server mode, which allows multiple debug
        sessions to connect to the same server process sequentially.
        
        Args:
            connection: Connection string (e.g., "listen://localhost:0")
        
        Returns:
            Tuple of (process, connection) where connection may be updated
            with the actual listening address.
        """
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

    def test_session_id_update_when_reuse(self):
        """
        Test if vscode session id being updated when lldb-dap reused
        """
        self.build()
        program = self.getBuildArtifact("a.out")
        (_, connection) = self.start_server(connection="listen://localhost:0")

        # Add set up command to print out the session id, with a unique identifier as prefix
        postRunCommands = [
            "script print('Actual_Session_ID: ' + str(os.getenv('VSCODE_DEBUG_SESSION_ID')))"
        ]

        # First debug session
        self.dap_server = dap_server.DebugAdapterServer(connection=connection)
        self.launch(
            program,
            vscode_session_id="test_session_id",
            postRunCommands=postRunCommands,
            disconnectAutomatically=False,
        )

        # Validate the session id in the initial launch of lldb-dap
        output = self.get_console()
        lines = filter(lambda x: "Actual_Session_ID" in x, output.splitlines())
        self.assertTrue(
            any("test_session_id" in l for l in lines),
            "expect session id in console output",
        )
        self.dap_server.request_disconnect()

        # Second debug session by reusing lldb-dap server
        self.dap_server = dap_server.DebugAdapterServer(connection=connection)
        self.launch(
            program,
            vscode_session_id="NEW_session_id",
            postRunCommands=postRunCommands,
            disconnectAutomatically=False,
        )

        # Validate the updated session id in the reused lldb-dap
        output = self.get_console()
        lines = filter(lambda x: "Actual_Session_ID" in x, output.splitlines())
        self.assertTrue(
            any("NEW_session_id" in l for l in lines),
            "expect new session id in console output",
        )
        self.dap_server.request_disconnect()

    @skipIfWindows
    def test_exception_breakopints(self):
        """
        Test reuse lldb-dap works across debug sessions.
        """
        self.build()
        program = self.getBuildArtifact("a.out")
        (_, connection) = self.start_server(connection="listen://localhost:0")

        # First debug session
        self.dap_server = dap_server.DebugAdapterServer(connection=connection)
        self.launch(program, disconnectAutomatically=False)

        filters = ["cpp_throw", "cpp_catch"]
        response = self.dap_server.request_setExceptionBreakpoints(filters=filters)
        self.assertTrue(response)
        self.assertTrue(response["success"])
        self.continue_to_exception_breakpoint("C++ Throw")
        self.dap_server.request_disconnect()

        # Second debug session by reusing lldb-dap server
        self.dap_server = dap_server.DebugAdapterServer(connection=connection)
        self.launch(program, disconnectAutomatically=False)

        filters = ["cpp_throw", "cpp_catch"]
        response = self.dap_server.request_setExceptionBreakpoints(filters=filters)
        self.assertTrue(response)
        self.assertTrue(response["success"])
        self.continue_to_exception_breakpoint("C++ Throw")
        self.dap_server.request_disconnect()
