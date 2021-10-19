from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *
from textwrap import dedent

class MyResponder(MockGDBServerResponder):
    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            return dedent("""\
                <?xml version="1.0"?>
                  <target version="1.0">
                    <architecture>aarch64</architecture>
                    <feature name="org.gnu.gdb.aarch64.core">
                      <reg name="x0" bitsize="64"/>
                      <reg name="x1" bitsize="64"/>
                      <reg name="x2" bitsize="64"/>
                      <reg name="x3" bitsize="64"/>
                      <reg name="x4" bitsize="64"/>
                      <reg name="x5" bitsize="64"/>
                      <reg name="x6" bitsize="64"/>
                      <reg name="x7" bitsize="64"/>
                      <reg name="x8" bitsize="64"/>
                      <reg name="x9" bitsize="64"/>
                      <reg name="x10" bitsize="64"/>
                      <reg name="x11" bitsize="64"/>
                      <reg name="x12" bitsize="64"/>
                      <reg name="x13" bitsize="64"/>
                      <reg name="x14" bitsize="64"/>
                      <reg name="x15" bitsize="64"/>
                      <reg name="x16" bitsize="64"/>
                      <reg name="x17" bitsize="64"/>
                      <reg name="x18" bitsize="64"/>
                      <reg name="x19" bitsize="64"/>
                      <reg name="x20" bitsize="64"/>
                      <reg name="x21" bitsize="64"/>
                      <reg name="x22" bitsize="64"/>
                      <reg name="x23" bitsize="64"/>
                      <reg name="x24" bitsize="64"/>
                      <reg name="x25" bitsize="64"/>
                      <reg name="x26" bitsize="64"/>
                      <reg name="x27" bitsize="64"/>
                      <reg name="x28" bitsize="64"/>
                      <reg name="x29" bitsize="64"/>
                      <reg name="x30" bitsize="64"/>
                      <reg name="sp" bitsize="64"/>
                      <reg name="pc" bitsize="64"/>
                    </feature>
                  </target>
                """), False
        else:
            return None, False

class TestQemuAarch64TargetXml(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_register_augmentation(self):
        """
        Test that we correctly associate the register info with the eh_frame
        register numbers.
        """

        target = self.createTarget("basic_eh_frame-aarch64.yaml")
        self.server.responder = MyResponder()

        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                [lldb.eStateStopped])
        self.filecheck("image show-unwind -n foo", __file__,
            "--check-prefix=UNWIND")
# UNWIND: eh_frame UnwindPlan:
# UNWIND: row[0]:    0: CFA=x29+16 => x30=[CFA-8]
