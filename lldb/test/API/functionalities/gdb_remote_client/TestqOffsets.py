import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestqOffsets(GDBRemoteTestBase):

    class Responder(MockGDBServerResponder):
        def qOffsets(self):
            return 'Text=470000;Data=470000'

    def test(self):
        self.server.responder = TestqOffsets.Responder()
        target = self.createTarget("qOffsets.yaml")
        text = target.modules[0].FindSection(".text")
        self.assertEquals(text.GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)

        process = self.connect(target)
        self.assertEquals(text.GetLoadAddress(target), 0x471000)
