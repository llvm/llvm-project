import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestContinue(GDBRemoteTestBase):
    class BaseResponder(MockGDBServerResponder):
        def qSupported(self, client_supported):
            return "PacketSize=3fff;QStartNoAckMode+;multiprocess+"

        def qfThreadInfo(self):
            return "mp400.401"

        def haltReason(self):
            return "S13"

        def cont(self):
            return "W01"

        def other(self, packet):
            if packet == "vCont?":
                return "vCont;c;C;s;S"
            if packet.startswith("vCont;"):
                return "W00"
            return ""

    def test_continue_no_multiprocess(self):
        class MyResponder(self.BaseResponder):
            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+"

            def qfThreadInfo(self):
                return "m401"

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateExited]
        )
        self.assertPacketLogContains(["vCont;C13:401"])

    def test_continue_no_vCont(self):
        class MyResponder(self.BaseResponder):
            def qSupported(self, client_supported):
                return "PacketSize=3fff;QStartNoAckMode+"

            def qfThreadInfo(self):
                return "m401"

            def other(self, packet):
                return ""

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateExited]
        )
        self.assertPacketLogContains(["Hc401", "C13"])

    def test_continue_multiprocess(self):
        class MyResponder(self.BaseResponder):
            pass

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateExited]
        )
        self.assertPacketLogContains(["vCont;C13:p400.401"])
