"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.utils import DebugAdapter


class TestDAP_launch_termination(DAPTestCaseBase):
    """
    Tests the correct termination of lldb-dap upon a 'disconnect' request.
    """

    USE_DEFAULT_DEBUG_ADAPTER = False

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
        session.disconnect()

        # Wait until the underlying lldb-dap process dies.
        adapter.process.wait(timeout=self.DEFAULT_TIMEOUT)
        self.assertFalse(session.is_running(), f"expected ended session.")

        # Check the return code.
        self.assertEqual(adapter.process.poll(), 0)
