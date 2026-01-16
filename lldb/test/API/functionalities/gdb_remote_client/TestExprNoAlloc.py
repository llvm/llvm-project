import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class MyResponder(MockGDBServerResponder):
    def _M(self, size, permissions) -> str:
        return "E04"

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        return "".join(
            [
                # x0
                "2000000000000000",
                # x1..x30, sp, pc
                32 * "0000000000000000",
                # cpsr
                "00000000",
            ]
        )


class TestExprNoAlloc(GDBRemoteTestBase):
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test(self):
        """
        Test that a simple expression can be evaluated when the server supports the '_M'
        packet, but memory cannot be allocated, and it returns an error code response.
        In this case, 'CanJIT' used to be set to 'eCanJITYes', so 'IRMemoryMap' tried to
        allocated memory in the inferior process and failed.
        """

        self.server.responder = MyResponder()
        # Note: DynamicLoaderStatic disables JIT by calling 'm_process->SetCanJIT(false)'
        # in LoadAllImagesAtFileAddresses(). Specifying a triple with "-linux" enables
        # DynamicLoaderPOSIXDYLD to be used instead.
        self.target = self.createTarget("basic_eh_frame-aarch64.yaml", "aarch64-linux")
        process = self.connect(self.target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        self.expect_expr("$x0", result_type="unsigned long", result_value="32")
