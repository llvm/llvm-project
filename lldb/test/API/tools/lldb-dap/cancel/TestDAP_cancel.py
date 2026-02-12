"""
Test lldb-dap cancel request
"""

import os
import time
import lldbdap_testcase


class TestDAP_cancel(lldbdap_testcase.DAPTestCaseBase):
    def send_async_req(self, command: str, arguments: dict = {}) -> int:
        return self.dap_server.send_packet(
            {
                "seq": 0,
                "type": "request",
                "command": command,
                "arguments": arguments,
            }
        )

    def async_blocking_request(self, count: int) -> int:
        """
        Sends an evaluate request that will sleep for the specified count to
        block the request handling thread.
        """
        return self.send_async_req(
            command="evaluate",
            arguments={"expression": f"`busy-loop {count}", "context": "repl"},
        )

    def async_cancel(self, requestId: int) -> int:
        return self.send_async_req(command="cancel", arguments={"requestId": requestId})

    def test_pending_request(self):
        """
        Tests cancelling a pending request.
        """
        program = self.getBuildArtifact("a.out")
        busy_loop =  self.getSourcePath("busy_loop.py")
        self.build_and_launch(
            program,
            initCommands=[f"command script import {busy_loop}"],
            stopOnEntry=True,
        )
        self.verify_stop_on_entry()

        # Use a relatively short timeout since this is only to ensure the
        # following request is queued.
        blocking_seq = self.async_blocking_request(count=1)
        # Use a longer timeout to ensure we catch if the request was interrupted
        # properly.
        pending_seq = self.async_blocking_request(count=10)
        cancel_seq = self.async_cancel(requestId=pending_seq)

        blocking_resp = self.dap_server.receive_response(blocking_seq)
        self.assertEqual(blocking_resp["request_seq"], blocking_seq)
        self.assertEqual(blocking_resp["command"], "evaluate")
        self.assertEqual(blocking_resp["success"], True)

        pending_resp = self.dap_server.receive_response(pending_seq)
        self.assertEqual(pending_resp["request_seq"], pending_seq)
        self.assertEqual(pending_resp["command"], "evaluate")
        self.assertEqual(pending_resp["success"], False)
        self.assertEqual(pending_resp["message"], "cancelled")

        cancel_resp = self.dap_server.receive_response(cancel_seq)
        self.assertEqual(cancel_resp["request_seq"], cancel_seq)
        self.assertEqual(cancel_resp["command"], "cancel")
        self.assertEqual(cancel_resp["success"], True)
        self.continue_to_exit()

    def test_inflight_request(self):
        """
        Tests cancelling an inflight request.
        """
        program = self.getBuildArtifact("a.out")
        busy_loop = os.path.join(os.path.dirname(__file__), "busy_loop.py")
        self.build_and_launch(
            program,
            initCommands=[f"command script import {busy_loop}"],
            stopOnEntry=True,
        )
        self.verify_configuration_done()
        self.verify_stop_on_entry()

        blocking_seq = self.async_blocking_request(count=10)
        # Wait for the sleep to start to cancel the inflight request.
        time.sleep(0.5)
        cancel_seq = self.async_cancel(requestId=blocking_seq)

        blocking_resp = self.dap_server.receive_response(blocking_seq)
        self.assertEqual(blocking_resp["request_seq"], blocking_seq)
        self.assertEqual(blocking_resp["command"], "evaluate")
        self.assertEqual(blocking_resp["success"], False)
        self.assertEqual(blocking_resp["message"], "cancelled")

        cancel_resp = self.dap_server.receive_response(cancel_seq)
        self.assertEqual(cancel_resp["request_seq"], cancel_seq)
        self.assertEqual(cancel_resp["command"], "cancel")
        self.assertEqual(cancel_resp["success"], True)
        self.continue_to_exit()
