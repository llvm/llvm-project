import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestNoWatchpointSupportInfo(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    @skipIfRemote
    def test(self):
        """
        Test lldb's parsing of the <architecture> tag in the target.xml register
        description packet.
        """

        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T02thread:1ff0d;thread-pcs:10001bc00;"

            def threadStopInfo(self, threadnum):
                if threadnum == 0x1FF0D:
                    return "T02thread:1ff0d;thread-pcs:10001bc00;"
                return ""

            def setBreakpoint(self, packet):
                if packet.startswith("Z2,"):
                    return "OK"

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
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))
        self.dbg.SetDefaultArchitecture("x86_64")
        target = self.dbg.CreateTargetWithFileAndArch(None, None)

        process = self.connect(target)

        if self.TraceOn():
            interp = self.dbg.GetCommandInterpreter()
            result = lldb.SBCommandReturnObject()
            interp.HandleCommand("target list", result)
            print(result.GetOutput())

        err = lldb.SBError()
        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)
        wp = target.WatchpointCreateByAddress(0x100, 8, wp_opts, err)
        if self.TraceOn() and (err.Fail() or not wp.IsValid):
            strm = lldb.SBStream()
            err.GetDescription(strm)
            print("watchpoint failed: %s" % strm.GetData())
        self.assertTrue(wp.IsValid())
