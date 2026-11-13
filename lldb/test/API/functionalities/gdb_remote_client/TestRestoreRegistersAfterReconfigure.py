"""
Test that restoring the register checkpoint taken before an expression does
not use a buffer saved under a different register layout. The stub changes
the SVE vector length while the expression runs, which makes lldb reconfigure
the register layout between the save and the restore.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

REGS = (
    [f"x{i}" for i in range(29)]
    + ["fp", "lr", "sp", "pc", "cpsr", "vg"]
    + [f"z{i}" for i in range(4)]
)
VG = REGS.index("vg")
PC = REGS.index("pc")


class Responder(MockGDBServerResponder):
    def __init__(self, restore_vector_length=False):
        super().__init__()
        self.restore_vector_length = restore_vector_length
        self.vg = 2
        self.pc = 0x30100
        self.breakpoint = None
        self.next_alloc = 0x20000000
        self.writes_after_resize = []
        self.restored_vectors = []
        self.resized = False

    def qSupported(self, client_supported):
        return "PacketSize=3fff;QStartNoAckMode+"

    def qHostInfo(self):
        return "cputype:16777228;cpusubtype:2;ostype:ios;vendor:apple;endian:little;ptrsize:8;"

    def qProcessInfo(self):
        return "pid:1;cputype:100000c;cpusubtype:2;ostype:ios;vendor:apple;endian:little;ptrsize:8;"

    def qfThreadInfo(self):
        return "m1"

    def haltReason(self):
        return f"T05thread:1;threads:1;thread-pcs:{self.pc:x};"

    def byte_size(self, regnum):
        name = REGS[regnum]
        if name[0] == "z":
            return self.vg * 8
        return 4 if name == "cpsr" else 8

    def qRegisterInfo(self, num):
        if num >= len(REGS):
            return "E45"
        offset = sum(self.byte_size(i) for i in range(num))
        return (
            f"name:{REGS[num]};bitsize:{self.byte_size(num) * 8};offset:{offset};"
            "encoding:uint;format:hex;set:General Purpose Registers;"
        )

    def readRegister(self, regnum):
        if regnum == VG:
            return self.vg.to_bytes(8, "little").hex()
        if regnum == PC:
            return self.pc.to_bytes(8, "little").hex()
        return "00" * self.byte_size(regnum)

    def writeRegister(self, register, value_hex):
        if self.vg != 2:
            self.writes_after_resize.append(REGS[register])
        if register == VG and self.restore_vector_length:
            self.vg = int.from_bytes(bytes.fromhex(value_hex), "little")
        if self.resized and REGS[register].startswith("z"):
            self.restored_vectors.append((REGS[register], len(value_hex) // 2))
        return "OK"

    def other(self, packet):
        if packet.startswith("_M"):
            self.next_alloc += 0x10000
            return f"{self.next_alloc:x}"
        if packet.startswith("_m"):
            return "OK"
        return ""

    def setBreakpoint(self, packet):
        self.breakpoint = int(packet.split(",")[1], 16)
        return "OK"

    def clearBreakpoint(self, packet):
        return "OK"

    def cont(self):
        # Grow the SVE registers while the expression runs.
        self.resized = True
        self.vg = 32
        self.pc = self.breakpoint
        return self.haltReason()


class TestRestoreRegistersAfterReconfigure(GDBRemoteTestBase):
    def evaluate_expression(self, responder):
        self.server.responder = responder
        target = self.createTarget("arm64-ios-main.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )
        target.SetModuleLoadAddress(target.GetModuleAtIndex(0), 0)

        self.expect("expression -- ((int(*)())0x30000)()", substrs=["= 0"])

    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test(self):
        responder = Responder()
        self.evaluate_expression(responder)
        self.assertEqual(responder.vg, 32)
        # Attempt the layout control, but do not replay the saved values if
        # the stub acknowledges it without restoring the saved vector length.
        self.assertEqual(responder.writes_after_resize, ["vg"])
        self.assertEqual(responder.restored_vectors, [])

    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_restores_vector_length_before_values(self):
        responder = Responder(restore_vector_length=True)
        self.evaluate_expression(responder)
        self.assertEqual(responder.vg, 2)
        self.assertEqual(responder.writes_after_resize, ["vg"])
        self.assertEqual(responder.restored_vectors, [(f"z{i}", 16) for i in range(4)])
