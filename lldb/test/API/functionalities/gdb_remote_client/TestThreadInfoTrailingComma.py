import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestThreadInfoTrailingComma(GDBRemoteTestBase):
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T02thread:1"

            def qfThreadInfo(self):
                return "m1,2,3,4,"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)
        self.assertEqual(process.GetThreadAtIndex(0).GetThreadID(), 1)
        self.assertEqual(process.GetThreadAtIndex(1).GetThreadID(), 2)
        self.assertEqual(process.GetThreadAtIndex(2).GetThreadID(), 3)
        self.assertEqual(process.GetThreadAtIndex(3).GetThreadID(), 4)
