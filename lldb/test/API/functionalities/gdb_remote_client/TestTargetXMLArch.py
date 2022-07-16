from __future__ import print_function
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class MyResponder(MockGDBServerResponder):
    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            return """<?xml version="1.0"?>
                <target version="1.0">
                  <architecture>i386:x86-64</architecture>
                  <feature name="org.gnu.gdb.i386.core">

                 <flags id="i386_eflags" size="4">
                 <field name="CF" start="0" end="0"/>
                 <field name="" start="1" end="1"/>
                 <field name="PF" start="2" end="2"/>
                 <field name="AF" start="4" end="4"/>
                 <field name="ZF" start="6" end="6"/>
                 <field name="SF" start="7" end="7"/>
                 <field name="TF" start="8" end="8"/>
                 <field name="IF" start="9" end="9"/>
                 <field name="DF" start="10" end="10"/>
                 <field name="OF" start="11" end="11"/>
                 <field name="NT" start="14" end="14"/>
                 <field name="RF" start="16" end="16"/>
                 <field name="VM" start="17" end="17"/>
                 <field name="AC" start="18" end="18"/>
                 <field name="VIF" start="19" end="19"/>
                 <field name="VIP" start="20" end="20"/>
                 <field name="ID" start="21" end="21"/>
                 </flags>

                    <reg name="rax" bitsize="64" regnum="0" type="int" group="general"/>
                    <reg name="rbx" bitsize="64" regnum="1" type="int" group="general"/>
                    <reg name="rcx" bitsize="64" regnum="2" type="int" group="general"/>
                    <reg name="rdx" bitsize="64" regnum="3" type="int" group="general"/>
                    <reg name="rsi" bitsize="64" regnum="4" type="int" group="general"/>
                    <reg name="rdi" bitsize="64" regnum="5" type="int" group="general"/>
                    <reg name="rbp" bitsize="64" regnum="6" type="data_ptr" group="general"/>
                    <reg name="rsp" bitsize="64" regnum="7" type="data_ptr" group="general"/>
                    <reg name="r8" bitsize="64"  regnum="8" type="int" group="general"/>
                    <reg name="r9" bitsize="64"  regnum="9" type="int" group="general"/>
                    <reg name="r10" bitsize="64" regnum="10" type="int" group="general"/>
                    <reg name="r11" bitsize="64" regnum="11" type="int" group="general"/>
                    <reg name="r12" bitsize="64" regnum="12" type="int" group="general"/>
                    <reg name="r13" bitsize="64" regnum="13" type="int" group="general"/>
                    <reg name="r14" bitsize="64" regnum="14" type="int" group="general"/>
                    <reg name="r15" bitsize="64" regnum="15" type="int" group="general"/>
                    <reg name="rip" bitsize="64" regnum="16" type="code_ptr" group="general"/>
                    <reg name="eflags" bitsize="32" regnum="17" type="i386_eflags" group="general"/>

                    <reg name="cs" bitsize="32" regnum="18" type="int" group="general"/>
                    <reg name="ss" bitsize="32" regnum="19" type="int" group="general"/>
                    <reg name="ds" bitsize="32" regnum="20" type="int" group="general"/>
                    <reg name="es" bitsize="32" regnum="21" type="int" group="general"/>
                    <reg name="fs" bitsize="32" regnum="22" type="int" group="general"/>
                    <reg name="gs" bitsize="32" regnum="23" type="int" group="general"/>

                    <reg name="st0" bitsize="80" regnum="24" type="i387_ext" group="float"/>
                    <reg name="st1" bitsize="80" regnum="25" type="i387_ext" group="float"/>
                    <reg name="st2" bitsize="80" regnum="26" type="i387_ext" group="float"/>
                    <reg name="st3" bitsize="80" regnum="27" type="i387_ext" group="float"/>
                    <reg name="st4" bitsize="80" regnum="28" type="i387_ext" group="float"/>
                    <reg name="st5" bitsize="80" regnum="29" type="i387_ext" group="float"/>
                    <reg name="st6" bitsize="80" regnum="30" type="i387_ext" group="float"/>
                    <reg name="st7" bitsize="80" regnum="31" type="i387_ext" group="float"/>

                    <reg name="fctrl" bitsize="32" regnum="32" type="int" group="float"/>
                    <reg name="fstat" bitsize="32" regnum="33" type="int" group="float"/>
                    <reg name="ftag"  bitsize="32" regnum="34" type="int" group="float"/>
                    <reg name="fiseg" bitsize="32" regnum="35" type="int" group="float"/>
                    <reg name="fioff" bitsize="32" regnum="36" type="int" group="float"/>
                    <reg name="foseg" bitsize="32" regnum="37" type="int" group="float"/>
                    <reg name="fooff" bitsize="32" regnum="38" type="int" group="float"/>
                    <reg name="fop"   bitsize="32" regnum="39" type="int" group="float"/>
                  </feature>
                </target>""", False
        else:
            return None, False

    def qC(self):
        return "QC1"

    def haltReason(self):
        return "T05thread:00000001;06:9038d60f00700000;07:98b4062680ffffff;10:c0d7bf1b80ffffff;"

    def readRegister(self, register):
        regs = {0x0: "00b0060000610000",
                0xa: "68fe471c80ffffff",
                0xc: "60574a1c80ffffff",
                0xd: "18f3042680ffffff",
                0xe: "be8a4d7142000000",
                0xf: "50df471c80ffffff",
                0x10: "c0d7bf1b80ffffff" }
        if register in regs:
            return regs[register]
        else:
            return "0000000000000000"

class TestTargetXMLArch(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @expectedFailureAll(archs=["i386"])
    @skipIfRemote
    def test(self):
        """
        Test lldb's parsing of the <architecture> tag in the target.xml register
        description packet.
        """
        self.server.responder = MyResponder()
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(
                    lambda: self.runCmd("log disable gdb-remote packets"))

        target = self.dbg.CreateTarget('')
        self.assertEqual('', target.GetTriple())
        process = self.connect(target)
        if self.TraceOn():
            interp.HandleCommand("target list", result)
            print(result.GetOutput())
        self.assertTrue(target.GetTriple().startswith('x86_64-unknown-unknown'))

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_register_augmentation(self):
        """
        Test that we correctly associate the register info with the eh_frame
        register numbers.
        """

        target = self.createTarget("basic_eh_frame.yaml")
        self.server.responder = MyResponder()

        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                [lldb.eStateStopped])
        self.filecheck("image show-unwind -n foo", __file__,
            "--check-prefix=UNWIND")
# UNWIND: eh_frame UnwindPlan:
# UNWIND: row[0]:    0: CFA=rsp+128 => rip=[CFA-8]
