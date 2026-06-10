"""
Test lldb-dap launch request.
"""

import lldbdap_testcase


class TestDAP_launch_termination(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the correct termination of lldb-dap upon a 'disconnect' request.
    """

    def test(self):
        self.create_debug_adapter()
        # The underlying lldb-dap process must be alive
        self.assertEqual(self.dap_server.process.poll(), None)

        # The lldb-dap process should finish even though
        # we didn't close the communication socket explicitly
        self.dap_server.request_disconnect()

        # Wait until the underlying lldb-dap process dies.
        self.dap_server.process.wait(timeout=self.DEFAULT_TIMEOUT)

        # Check the return code
        self.assertEqual(self.dap_server.process.poll(), 0)
