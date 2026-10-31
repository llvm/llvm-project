import lldb
from lldbsuite.test.gdbclientutils import MockGDBServerResponder
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestTargetSyncStopReason(GDBRemoteTestBase):
    def test(self):
        """A target-sync thread is not selected over an independent stop."""

        # Include packet delimiters and escaped characters to verify that the
        # server's ASCII-hex field is decoded as one plugin name.
        plugin_name_hex = "mock;plugin:$}#".encode("ascii").hex()

        class MyResponder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "m1,2"

            def haltReason(self):
                return "T02thread:1;"

            def cont(self):
                # The server stopped thread 2 with SIGSTOP to keep the process
                # in sync with an accelerator plugin. Keep thread after the
                # encoded plugin field so this also checks packet framing.
                return (
                    "T13reason:target sync;sync-plugin:%s;thread:2;" % plugin_name_hex
                )

            def threadStopInfo(self, threadnum):
                # Meanwhile thread 1 stopped for SIGINT. An empty reply would
                # tell LLDB that the packet is not supported.
                if threadnum == 1:
                    return "T02thread:1;reason:signal;"
                return "T00thread:%x;" % threadnum

        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.assertSuccess(process.Continue())
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetThreadByID(2)
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonTargetSync)
        self.assertEqual(thread.GetStopDescription(256), "target sync")
        self.assertEqual(process.GetSelectedThread().GetThreadID(), 1)
