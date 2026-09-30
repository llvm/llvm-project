"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import requireSocketPermission
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import DAPError
from lldbsuite.test.tools.lldb_dap.utils import DebugAdapter


class TestDAP_launch_termination(DAPTestCaseBase):
    """
    Tests the correct termination of lldb-dap upon a 'disconnect' request.
    """

    USE_DEFAULT_DEBUG_ADAPTER = False

    @requireSocketPermission
    def test_termination_socket(self):
        adapter = self.create_server_debug_adapter(
            connection="listen://localhost:0",
            connection_timeout=1,
        )
        self.do_test_termination(adapter)

    def test_termination_stdio(self):
        adapter = self.create_stdio_debug_adapter()
        self.do_test_termination(adapter)

    def do_test_termination(self, adapter: DebugAdapter):
        # The underlying lldb-dap process must be alive.
        self.assertTrue(adapter.is_alive, f"adapter is dead: {adapter.process.args}")
        session = self.create_session(adapter, disconnect_automatically=False)

        session.initialize_sequence(session.initialize_args)
        # The lldb-dap process should finish even though
        # we didn't close the communication socket explicitly.
        disconnect_resp = session.disconnect()

        # No event comes after the 'disconnect' response, so the wait ends when
        # lldb-dap closes the connection and the session stops reading.
        with self.assertRaises(DAPError):
            session.wait_for_terminated_event(after=disconnect_resp)
        self.assertFalse(session.is_running(), f"expected ended session.")

        # Wait until the underlying lldb-dap process dies.
        adapter.process.wait(timeout=self.DEFAULT_TIMEOUT)

        # Check the return code.
        self.assertEqual(adapter.process.poll(), 0)
