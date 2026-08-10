import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestStopPCs(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                # lldb should treat the default halt reason, hwbreak and swbreak in the same way. Which is that it
                # expects the stub to have corrected the PC already, so lldb should not modify it further.
                return "T02thread:1ff0d;threads:1ff0d,2ff0d,3ff0d;thread-pcs:10001bc00,10002bc00,10003bc00;"

            def threadStopInfo(self, threadnum):
                if threadnum == 0x1FF0D:
                    return "T02thread:1ff0d;threads:1ff0d,2ff0d,3ff0d;thread-pcs:10001bc00,10002bc00,10003bc00;"
                if threadnum == 0x2FF0D:
                    return "T00swbreak:;thread:2ff0d;threads:1ff0d,2ff0d,3ff0d;thread-pcs:10001bc00,10002bc00,10003bc00;"
                if threadnum == 0x3FF0D:
                    return "T00hwbreak:;thread:3ff0d;threads:1ff0d,2ff0d,3ff0d;thread-pcs:10001bc00,10002bc00,10003bc00;"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                          </feature>
                        </target>""",
                        False,
                    )
                else:
                    return None, False

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)

        self.assertEqual(process.GetNumThreads(), 3)
        th0 = process.GetThreadAtIndex(0)
        th1 = process.GetThreadAtIndex(1)
        th2 = process.GetThreadAtIndex(2)
        self.assertEqual(th0.GetThreadID(), 0x1FF0D)
        self.assertEqual(th1.GetThreadID(), 0x2FF0D)
        self.assertEqual(th2.GetThreadID(), 0x3FF0D)
        self.assertEqual(th0.GetFrameAtIndex(0).GetPC(), 0x10001BC00)
        self.assertEqual(th1.GetFrameAtIndex(0).GetPC(), 0x10002BC00)
        self.assertEqual(th2.GetFrameAtIndex(0).GetPC(), 0x10003BC00)
