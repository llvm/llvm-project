"""
Test lldb-dap cancel request
"""

import time

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_launch(lldbdap_testcase.DAPTestCaseBase):
    def send_async_req(self, command: str, arguments={}) -> int:
        seq = self.dap_server.sequence
        self.dap_server.send_packet(
            {
                "type": "request",
                "command": command,
                "arguments": arguments,
            }
        )
        return seq

    def async_blocking_request(self, duration: float) -> int:
        """
        Sends an evaluate request that will sleep for the specified duration to
        block the request handling thread.
        """
        return self.send_async_req(
            command="evaluate",
            arguments={
                "expression": '`script import time; print("starting sleep", file=lldb.debugger.GetOutputFileHandle()); time.sleep({})'.format(
                    duration
                ),
                "context": "repl",
            },
        )

    def async_cancel(self, requestId: int) -> int:
        return self.send_async_req(command="cancel", arguments={"requestId": requestId})

    def test_pending_request(self):
        """
        Tests cancelling a pending request.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.continue_to_next_stop()

        # Use a relatively short timeout since this is only to ensure the
        # following request is queued.
        blocking_seq = self.async_blocking_request(duration=1.0)
        # Use a longer timeout to ensure we catch if the request was interrupted
        # properly.
        pending_seq = self.async_blocking_request(duration=self.timeoutval / 2)
        cancel_seq = self.async_cancel(requestId=pending_seq)

        blocking_resp = self.dap_server.recv_packet(filter_type=["response"])
        self.assertEqual(blocking_resp["request_seq"], blocking_seq)
        self.assertEqual(blocking_resp["command"], "evaluate")
        self.assertEqual(blocking_resp["success"], True)

        pending_resp = self.dap_server.recv_packet(filter_type=["response"])
        self.assertEqual(pending_resp["request_seq"], pending_seq)
        self.assertEqual(pending_resp["command"], "evaluate")
        self.assertEqual(pending_resp["success"], False)
        self.assertEqual(pending_resp["message"], "cancelled")

        cancel_resp = self.dap_server.recv_packet(filter_type=["response"])
        self.assertEqual(cancel_resp["request_seq"], cancel_seq)
        self.assertEqual(cancel_resp["command"], "cancel")
        self.assertEqual(cancel_resp["success"], True)
        self.continue_to_exit()

    def test_inflight_request(self):
        """
        Tests cancelling an inflight request.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.continue_to_next_stop()

        blocking_seq = self.async_blocking_request(duration=self.timeoutval / 2)
        # Wait for the sleep to start to cancel the inflight request.
        self.collect_stdout(
            timeout_secs=self.timeoutval,
            pattern="starting sleep",
        )
        cancel_seq = self.async_cancel(requestId=blocking_seq)

        blocking_resp = self.dap_server.recv_packet(filter_type=["response"])
        self.assertEqual(blocking_resp["request_seq"], blocking_seq)
        self.assertEqual(blocking_resp["command"], "evaluate")
        self.assertEqual(blocking_resp["success"], False)
        self.assertEqual(blocking_resp["message"], "cancelled")

        cancel_resp = self.dap_server.recv_packet(filter_type=["response"])
        self.assertEqual(cancel_resp["request_seq"], cancel_seq)
        self.assertEqual(cancel_resp["command"], "cancel")
        self.assertEqual(cancel_resp["success"], True)
        self.continue_to_exit()
