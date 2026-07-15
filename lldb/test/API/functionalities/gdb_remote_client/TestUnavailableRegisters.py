import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

# Test that lldb will cache when a register is unavailable,
# and not request the value multiple times at a stop event.
#
# Also test that a stop reply packet which indicates that
# the register cannot be read currently will keep lldb from
# trying to read it again.


class TestUnavailableRegisters(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    @skipIfLLVMTargetMissing("AArch64")
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                super().__init__()
                self.initial_x2_read = True

            def haltReason(self):
                # Register 1 cannot be fetched.
                return (
                    "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;00:0100000000000000;"
                    "01:;"
                    "03:0500000000000000;04:00bc010001000000;"
                )

            def threadStopInfo(self, threadnum):
                return self.haltReason()

            def writeRegisters(self):
                return "E02"

            def qHostInfo(self):
                return "cputype:16777228;cpusubtype:2;endian:little;ptrsize:8;"

            def readRegisters(self):
                return "E01"

            def readRegister(self, regnum):
                # Register 1 was listed in the stop reply packet's expedited
                # registers as unavailable.  lldb should not send a "p1" to try
                # to read it.  Send back a value so we can detect if this happened.
                if regnum == 1:
                    return "5555555555555555"

                # Register 2 was not included in the stop reply packet.  Return an
                # error when we try to read it the first time, return a value the
                # second time we try to read it.  We should never see the value.
                if regnum == 2:
                    if self.initial_x2_read:
                        return "E20"
                        self.initial_x2_read = False
                    else:
                        return "8888888888888888"

                return "E03"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>aarch64</architecture>
                          <feature name="org.gnu.gdb.aarch64.core">
                            <reg name="x0" regnum="0" bitsize="64"/>
                            <reg name="x1" bitsize="64"/>
                            <reg name="x2" bitsize="64"/>
                            <reg name="x3" bitsize="64"/>
                            <reg name="x4" bitsize="64"/>
                            <reg name="pc" bitsize="64"/>
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

        thread = process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(0)

        self.assertEqual(frame.FindRegister("pc").GetValueAsUnsigned(), 0x10001BC00)

        # value 1 from the stop reply packet's expedited registers
        self.assertEqual(frame.FindRegister("x0").GetValueAsUnsigned(), 1)

        # Register is marked unavailable in the stop reply packet's
        # expedited register list.  Should not get a value by sending
        # 'p1'.
        self.assertFailure(frame.FindRegister("x1").GetError())

        # x2 is _not_ in the expedited registers, and will reply with
        # an error the first time it is requested.
        self.assertFailure(frame.FindRegister("x2").GetError())

        # x2 will reply with a value the second time it is requested.
        # lldb should cache the error from the first attempt and not
        # try again.
        self.assertFailure(frame.FindRegister("x2").GetError())
