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
        self.assertPacketLogContains(["Hc401", "C13"])

    def test_continue_multiprocess(self):
        class MyResponder(self.BaseResponder):
            pass

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.assertPacketLogContains(["vCont;C13:p400.401"])

    # uses 'S13' instead of 's' arm (the pc dance?), testing it on one
    # arch should be entirely sufficient
    @skipIf(archs=no_match(["x86_64"]))
    def test_step_multiprocess(self):
        class MyResponder(self.BaseResponder):
            def other(self, packet):
                if packet == "vCont?":
                    return "vCont;c;C;s;S"
                if packet.startswith("vCont;C"):
                    return "S13"
                if packet.startswith("vCont;s"):
                    return "W00"
                return ""

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        thread = process.GetSelectedThread()
        thread.StepInstruction(False)
        self.assertPacketLogContains(["vCont;s:p400.401"])
