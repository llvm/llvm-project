import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class TestWriteMemory(GDBRemoteTestBase):

    def test(self):

        class MyResponder(MockGDBServerResponder):
            def setBreakpoint(self, packet):
                return "OK"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple('', 'x86_64-pc-linux')
        process = self.connect(target)

        bp = target.BreakpointCreateByAddress(0x1000)
        self.assertTrue(bp.IsValid())
        self.assertEqual(bp.GetNumLocations(), 1)
        bp.SetEnabled(True)
        self.assertTrue(bp.IsEnabled())

        err = lldb.SBError()
        data = str("\x01\x02\x03\x04")
        result = process.WriteMemory(0x1000, data, err)
        self.assertEqual(result, 4)
