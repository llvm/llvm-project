"""
Test lldb-dap cancel request
"""

import time

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.session_helpers import DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import CancelArgs, EvaluateArgs, LaunchArgs


class TestDAP_cancel(DAPTestCaseBase):
    def async_blocking_request(self, session: DAPTestSession, count: int):
        """
        Sends an evaluate request that will sleep for the specified count to
        block the request handling thread.
        """
        return session.send_request(
            EvaluateArgs(expression=f"`busy-loop {count}", context="repl")
        )

    def async_cancel(self, session: DAPTestSession, requestId: int):
        return session.send_request(CancelArgs(requestId=requestId))

    def test_pending_request(self):
        """Tests cancelling a pending request."""
        program = self.getBuildArtifact("a.out")
        busy_loop = self.getSourcePath("busy_loop.py")
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(
                program,
                initCommands=[f"command script import {busy_loop}"],
                stopOnEntry=True,
            )
        )
        session.verify_stopped_on_entry(after=process_event)

        # Use a relatively short timeout since this is only to ensure the
        # following request is queued.
        blocking_handle = self.async_blocking_request(session, count=1)
        # Use a longer timeout to ensure we catch if the request was interrupted
        # properly.
        pending_handle = self.async_blocking_request(session, count=10)
        cancel_handle = self.async_cancel(session, requestId=pending_handle.seq)

        blocking_resp = blocking_handle.result()
        self.assertEqual(blocking_resp.request_seq, blocking_handle.seq)
        self.assertEqual(blocking_resp.command, "evaluate")
        self.assertEqual(blocking_resp.success, True)

        pending_resp = pending_handle.error()
        self.assertEqual(pending_resp.request_seq, pending_handle.seq)
        self.assertEqual(pending_resp.command, "evaluate")
        self.assertEqual(pending_resp.success, False)
        self.assertEqual(pending_resp.message, "cancelled")

        cancel_resp = cancel_handle.result()
        self.assertEqual(cancel_resp.request_seq, cancel_handle.seq)
        self.assertEqual(cancel_resp.command, "cancel")
        self.assertEqual(cancel_resp.success, True)
        session.continue_to_exit()

    def test_inflight_request(self):
        """Tests cancelling an inflight request."""
        program = self.getBuildArtifact("a.out")
        busy_loop = self.getSourcePath("busy_loop.py")
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(
                program,
                initCommands=[f"command script import {busy_loop}"],
                stopOnEntry=True,
            )
        )
        session.verify_stopped_on_entry(after=process_event)

        blocking_handle = self.async_blocking_request(session, count=10)
        # Wait for the sleep to start to cancel the inflight request.
        time.sleep(0.5)
        cancel_handle = self.async_cancel(session, requestId=blocking_handle.seq)

        blocking_resp = blocking_handle.error()
        self.assertEqual(blocking_resp.request_seq, blocking_handle.seq)
        self.assertEqual(blocking_resp.command, "evaluate")
        self.assertEqual(blocking_resp.success, False)
        self.assertEqual(blocking_resp.message, "cancelled")

        cancel_resp = cancel_handle.result()
        self.assertEqual(cancel_resp.request_seq, cancel_handle.seq)
        self.assertEqual(cancel_resp.command, "cancel")
        self.assertEqual(cancel_resp.success, True)
        session.continue_to_exit()
