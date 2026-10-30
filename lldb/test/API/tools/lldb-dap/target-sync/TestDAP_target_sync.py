"""Test DAP focus behavior for target-synchronization stops."""

import json

from lldbsuite.test.decorators import requireSocketPermission
from lldbsuite.test.gdbclientutils import (
    MockGDBServer,
    MockGDBServerResponder,
    TCPServerSocket,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import AttachArgs, StoppedReason


class TestDAPTargetSync(DAPTestCaseBase):
    def setUp(self):
        super().setUp()
        self.server = MockGDBServer(TCPServerSocket())
        self.server.start()

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            self.server.stop()

    def attach(self):
        program = self.getBuildArtifact("a.out")
        self.yaml2obj("a.yaml", program)
        port = int(self.server.get_connect_address().rsplit(":", 1)[1])
        session = self.create_session()
        with session.configure(AttachArgs(program=program, gdbRemotePort=port)) as ctx:
            pass
        return session, ctx.process_event

    @requireSocketPermission
    def test_target_sync_preserves_focus(self):
        class Responder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "m2"

            def qC(self):
                return "QC2"

            def haltReason(self):
                return "T13thread:2;reason:target sync;sync-plugin:6d6f636b;"

            def cont(self):
                return self.haltReason()

        self.server.responder = Responder()
        session, process_event = self.attach()
        stopped = session.wait_for_stopped_event(after=process_event)

        self.assertEqual(stopped.body.reason, StoppedReason.PAUSE)
        self.assertEqual(stopped.body.threadId, 2)
        self.assertTrue(stopped.body.preserveFocusHint)
        self.assertTrue(stopped.body.allThreadsStopped)

    @requireSocketPermission
    def test_real_stop_selected_over_target_sync(self):
        plugin_name = "mock;plugin:$}#"
        thread_info = [
            {"tid": 1, "signal": 2, "reason": "signal"},
            {
                "tid": 2,
                "signal": 19,
                "reason": "target sync",
                "sync-plugin": plugin_name,
            },
        ]
        jstopinfo = json.dumps(thread_info, separators=(",", ":")).encode().hex()
        plugin_name_hex = plugin_name.encode("ascii").hex()

        class Responder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "m1,2"

            def qC(self):
                return "QC2"

            def haltReason(self):
                return (
                    "T13thread:2;reason:target sync;sync-plugin:%s;"
                    "threads:1,2;jstopinfo:%s;" % (plugin_name_hex, jstopinfo)
                )

            def cont(self):
                return self.haltReason()

        self.server.responder = Responder()
        session, process_event = self.attach()
        first = session.wait_for_stopped_event(after=process_event)
        second = session.wait_for_stopped_event(after=first)

        events = {first.body.threadId: first, second.body.threadId: second}
        sync_event = events[2]
        real_event = events[1]

        self.assertEqual(sync_event.body.reason, StoppedReason.PAUSE)
        self.assertTrue(sync_event.body.preserveFocusHint)
        self.assertFalse(sync_event.body.allThreadsStopped)

        self.assertEqual(real_event.body.reason, StoppedReason.EXCEPTION)
        self.assertNotEqual(real_event.body.preserveFocusHint, True)
        self.assertTrue(real_event.body.allThreadsStopped)
        self.assertEqual(second.body.threadId, 1)
