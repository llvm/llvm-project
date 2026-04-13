import lldb
from lldbsuite.support import seven
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestReadMemory(GDBRemoteTestBase):
    def test_x_with_prefix(self):
        class MyResponder(MockGDBServerResponder):
            def qSupported(self, client_features):
                # binary-upload+ indicates we use the gdb style of x packets
                return super().qSupported(client_features) + ";binary-upload+"

            def x(self, addr, length):
                return "bfoobar" if addr == 0x1000 else "E01"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)

        error = lldb.SBError()
        self.assertEqual(b"foobar", process.ReadMemory(0x1000, 10, error))

    def test_x_bare(self):
        class MyResponder(MockGDBServerResponder):
            def x(self, addr, length):
                # The OK response indicates we use the old lldb style.
                if addr == 0 and length == 0:
                    return "OK"
                return "foobar" if addr == 0x1000 else "E01"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)

        error = lldb.SBError()
        self.assertEqual(b"foobar", process.ReadMemory(0x1000, 10, error))

    def test_m_fallback(self):
        class MyResponder(MockGDBServerResponder):
            def x(self, addr, length):
                # If `x` is unsupported, we should fall back to `m`.
                return ""

            def readMemory(self, addr, length):
                return seven.hexlify("foobar") if addr == 0x1000 else "E01"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)

        error = lldb.SBError()
        self.assertEqual(b"foobar", process.ReadMemory(0x1000, 10, error))
