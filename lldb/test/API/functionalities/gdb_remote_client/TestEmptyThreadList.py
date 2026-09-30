"""
Test that LLDB survives a stop whose thread list comes back empty.

A stub that fails qfThreadInfo leaves the process with no threads while the
thread plan stack still remembers the threads from the previous stop. Reaping
those stale plans used to ask the thread list to refresh itself, re-entering
the update that was reaping them, without end.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestEmptyThreadList(GDBRemoteTestBase):
    def test_thread_list_goes_empty(self):
        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.threads_gone = False

            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+"

            def qfThreadInfo(self):
                # An error reply is what empties the list. An unsupported or
                # empty one instead falls back to assuming pid = tid = 1.
                if self.threads_gone:
                    return "E01"
                return "m401"

            def haltReason(self):
                # No "threads:" key, so the list has to come from qfThreadInfo.
                return "S13"

            def cont(self):
                return "S13"

            def other(self, packet):
                if packet == "vCont?":
                    return "vCont;c;C;s;S"
                if packet.startswith("vCont;"):
                    return "S13"
                return ""

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        # The first stop gives the plan stack a thread to remember.
        self.assertEqual(process.GetNumThreads(), 1)

        self.server.responder.threads_gone = True
        self.assertSuccess(process.Continue())
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertEqual(process.GetNumThreads(), 0)
