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
        return "20000000000000002000000000000000f0c154bfffff00005daa985a8fea0b48f0b954bfffff0000ad13cce570150b48380000000000000070456abfffff0000a700000000000000000000000000000001010101010101010000000000000000f0c154bfffff00000f2700000000000008e355bfffff0000080e55bfffff0000281041000000000010de61bfffff00005c05000000000000f0c154bfffff000090fcffffffff00008efcffffffff00008ffcffffffff00000000000000000000001000000000000090fcffffffff000000d06cbfffff0000f0c154bfffff00000100000000000000d0b954bfffff0000e407400000000000d0b954bfffff0000e40740000000000000100000"


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

        self.expect_expr("$x1", result_type="unsigned long", result_value="32")
