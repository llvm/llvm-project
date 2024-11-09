"""
Check that lldb falls back to default register layouts when the remote provides
no target XML or does not include registers in the target XML.

GPRS are passed to the responder to create register data to send back to lldb.
Registers in SUPPL are virtual registers based on those general ones. The tests
pass __file__ to FileCheck so any prefixes in this Python file will be checked.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

import binascii
from textwrap import dedent


class MyResponder(MockGDBServerResponder):
    @staticmethod
    def filecheck_to_blob(fc):
        for l in fc.strip().splitlines():
            val = l.split("0x")[1]
            yield binascii.b2a_hex(bytes(reversed(binascii.a2b_hex(val)))).decode()

    def __init__(self, architecture, has_target_xml, reg_data, halt_reason):
        super().__init__()
        self.architecture = architecture
        self.has_target_xml = has_target_xml
        self.reg_data = "".join(self.filecheck_to_blob(reg_data))
        self.halt_reason = halt_reason

    def readRegister(self, regnum):
        return ""

    def readRegisters(self):
        return self.reg_data

    def haltReason(self):
        return self.halt_reason

    def qXferRead(self, obj, annex, offset, length):
        if self.has_target_xml and annex == "target.xml":
            return (
                dedent(
                    f"""\
                    <?xml version="1.0"?>
                    <target version="1.0">
                      <architecture>{self.architecture}</architecture>
                    </target>"""
                ),
                False,
            )

        return None, False


class TestGDBServerTargetXML(GDBRemoteTestBase):
    def check_x86_64_regs(self, has_target_xml):
        GPRS = """
CHECK-AMD64-DAG: rax = 0x0807060504030201
CHECK-AMD64-DAG: rbx = 0x1817161514131211
CHECK-AMD64-DAG: rcx = 0x2827262524232221
CHECK-AMD64-DAG: rdx = 0x3837363534333231
CHECK-AMD64-DAG: rsi = 0x4847464544434241
CHECK-AMD64-DAG: rdi = 0x5857565554535251
CHECK-AMD64-DAG: rbp = 0x6867666564636261
CHECK-AMD64-DAG: rsp = 0x7877767574737271
CHECK-AMD64-DAG: r8 = 0x8887868584838281
CHECK-AMD64-DAG: r9 = 0x9897969594939291
CHECK-AMD64-DAG: r10 = 0xa8a7a6a5a4a3a2a1
CHECK-AMD64-DAG: r11 = 0xb8b7b6b5b4b3b2b1
CHECK-AMD64-DAG: r12 = 0xc8c7c6c5c4c3c2c1
CHECK-AMD64-DAG: r13 = 0xd8d7d6d5d4d3d2d1
CHECK-AMD64-DAG: r14 = 0xe8e7e6e5e4e3e2e1
CHECK-AMD64-DAG: r15 = 0xf8f7f6f5f4f3f2f1
CHECK-AMD64-DAG: rip = 0x100f0e0d0c0b0a09
CHECK-AMD64-DAG: eflags = 0x1c1b1a19
CHECK-AMD64-DAG: cs = 0x2c2b2a29
CHECK-AMD64-DAG: ss = 0x3c3b3a39
"""

        SUPPL = """
CHECK-AMD64-DAG: eax = 0x04030201
CHECK-AMD64-DAG: ebx = 0x14131211
CHECK-AMD64-DAG: ecx = 0x24232221
CHECK-AMD64-DAG: edx = 0x34333231
CHECK-AMD64-DAG: esi = 0x44434241
CHECK-AMD64-DAG: edi = 0x54535251
CHECK-AMD64-DAG: ebp = 0x64636261
CHECK-AMD64-DAG: esp = 0x74737271
CHECK-AMD64-DAG: r8d = 0x84838281
CHECK-AMD64-DAG: r9d = 0x94939291
CHECK-AMD64-DAG: r10d = 0xa4a3a2a1
CHECK-AMD64-DAG: r11d = 0xb4b3b2b1
CHECK-AMD64-DAG: r12d = 0xc4c3c2c1
CHECK-AMD64-DAG: r13d = 0xd4d3d2d1
CHECK-AMD64-DAG: r14d = 0xe4e3e2e1
CHECK-AMD64-DAG: r15d = 0xf4f3f2f1

CHECK-AMD64-DAG: ax = 0x0201
CHECK-AMD64-DAG: bx = 0x1211
CHECK-AMD64-DAG: cx = 0x2221
CHECK-AMD64-DAG: dx = 0x3231
CHECK-AMD64-DAG: si = 0x4241
CHECK-AMD64-DAG: di = 0x5251
CHECK-AMD64-DAG: bp = 0x6261
CHECK-AMD64-DAG: sp = 0x7271
CHECK-AMD64-DAG: r8w = 0x8281
CHECK-AMD64-DAG: r9w = 0x9291
CHECK-AMD64-DAG: r10w = 0xa2a1
CHECK-AMD64-DAG: r11w = 0xb2b1
CHECK-AMD64-DAG: r12w = 0xc2c1
CHECK-AMD64-DAG: r13w = 0xd2d1
CHECK-AMD64-DAG: r14w = 0xe2e1
CHECK-AMD64-DAG: r15w = 0xf2f1

CHECK-AMD64-DAG: ah = 0x02
CHECK-AMD64-DAG: bh = 0x12
CHECK-AMD64-DAG: ch = 0x22
CHECK-AMD64-DAG: dh = 0x32

CHECK-AMD64-DAG: al = 0x01
CHECK-AMD64-DAG: bl = 0x11
CHECK-AMD64-DAG: cl = 0x21
CHECK-AMD64-DAG: dl = 0x31
CHECK-AMD64-DAG: sil = 0x41
CHECK-AMD64-DAG: dil = 0x51
CHECK-AMD64-DAG: bpl = 0x61
CHECK-AMD64-DAG: spl = 0x71
CHECK-AMD64-DAG: r8l = 0x81
CHECK-AMD64-DAG: r9l = 0x91
CHECK-AMD64-DAG: r10l = 0xa1
CHECK-AMD64-DAG: r11l = 0xb1
CHECK-AMD64-DAG: r12l = 0xc1
CHECK-AMD64-DAG: r13l = 0xd1
CHECK-AMD64-DAG: r14l = 0xe1
CHECK-AMD64-DAG: r15l = 0xf1
"""

        self.server.responder = MyResponder(
            "x86-64",
            has_target_xml,
            GPRS,
            "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;",
        )

        target = self.createTarget("basic_eh_frame.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # test all registers
        self.filecheck(
            "register read --all",
            __file__,
            filecheck_options="--check-prefix=CHECK-AMD64",
        )

        # test generic aliases
        self.match("register read arg4", ["rcx = 0x2827262524232221"])
        self.match("register read arg3", ["rdx = 0x3837363534333231"])
        self.match("register read arg2", ["rsi = 0x4847464544434241"])
        self.match("register read arg1", ["rdi = 0x5857565554535251"])
        self.match("register read fp", ["rbp = 0x6867666564636261"])
        self.match("register read sp", ["rsp = 0x7877767574737271"])
        self.match("register read arg5", ["r8 = 0x8887868584838281"])
        self.match("register read arg6", ["r9 = 0x9897969594939291"])
        self.match("register read pc", ["rip = 0x100f0e0d0c0b0a09"])
        self.match("register read flags", ["eflags = 0x1c1b1a19"])

    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_x86_64_regs_no_target_xml(self):
        """Test grabbing various x86_64 registers from gdbserver when there
        is no target XML."""
        self.check_x86_64_regs(False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_x86_64_regs_no_register_info(self):
        """Test grabbing various x86_64 registers from gdbserver when there
        is target XML but it does not include register info."""
        self.check_x86_64_regs(True)

    def check_aarch64_regs(self, has_target_xml):
        GPRS = """
CHECK-AARCH64-DAG: x0 = 0x0001020304050607
CHECK-AARCH64-DAG: x1 = 0x0102030405060708
CHECK-AARCH64-DAG: x2 = 0x0203040506070809
CHECK-AARCH64-DAG: x3 = 0x030405060708090a
CHECK-AARCH64-DAG: x4 = 0x0405060708090a0b
CHECK-AARCH64-DAG: x5 = 0x05060708090a0b0c
CHECK-AARCH64-DAG: x6 = 0x060708090a0b0c0d
CHECK-AARCH64-DAG: x7 = 0x0708090a0b0c0d0e
CHECK-AARCH64-DAG: x8 = 0x08090a0b0c0d0e0f
CHECK-AARCH64-DAG: x9 = 0x090a0b0c0d0e0f10
CHECK-AARCH64-DAG: x10 = 0x0a0b0c0d0e0f1011
CHECK-AARCH64-DAG: x11 = 0x0b0c0d0e0f101112
CHECK-AARCH64-DAG: x12 = 0x0c0d0e0f10111213
CHECK-AARCH64-DAG: x13 = 0x0d0e0f1011121314
CHECK-AARCH64-DAG: x14 = 0x0e0f101112131415
CHECK-AARCH64-DAG: x15 = 0x0f10111213141516
CHECK-AARCH64-DAG: x16 = 0x1011121314151617
CHECK-AARCH64-DAG: x17 = 0x1112131415161718
CHECK-AARCH64-DAG: x18 = 0x1213141516171819
CHECK-AARCH64-DAG: x19 = 0x131415161718191a
CHECK-AARCH64-DAG: x20 = 0x1415161718191a1b
CHECK-AARCH64-DAG: x21 = 0x15161718191a1b1c
CHECK-AARCH64-DAG: x22 = 0x161718191a1b1c1d
CHECK-AARCH64-DAG: x23 = 0x1718191a1b1c1d1e
CHECK-AARCH64-DAG: x24 = 0x18191a1b1c1d1e1f
CHECK-AARCH64-DAG: x25 = 0x191a1b1c1d1e1f20
CHECK-AARCH64-DAG: x26 = 0x1a1b1c1d1e1f2021
CHECK-AARCH64-DAG: x27 = 0x1b1c1d1e1f202122
CHECK-AARCH64-DAG: x28 = 0x1c1d1e1f20212223
CHECK-AARCH64-DAG: x29 = 0x1d1e1f2021222324
CHECK-AARCH64-DAG: x30 = 0x1e1f202122232425
CHECK-AARCH64-DAG: sp = 0x1f20212223242526
CHECK-AARCH64-DAG: pc = 0x2021222324252627
CHECK-AARCH64-DAG: cpsr = 0x21222324
"""

        SUPPL = """
CHECK-AARCH64-DAG: w0 = 0x04050607
CHECK-AARCH64-DAG: w1 = 0x05060708
CHECK-AARCH64-DAG: w2 = 0x06070809
CHECK-AARCH64-DAG: w3 = 0x0708090a
CHECK-AARCH64-DAG: w4 = 0x08090a0b
CHECK-AARCH64-DAG: w5 = 0x090a0b0c
CHECK-AARCH64-DAG: w6 = 0x0a0b0c0d
CHECK-AARCH64-DAG: w7 = 0x0b0c0d0e
CHECK-AARCH64-DAG: w8 = 0x0c0d0e0f
CHECK-AARCH64-DAG: w9 = 0x0d0e0f10
CHECK-AARCH64-DAG: w10 = 0x0e0f1011
CHECK-AARCH64-DAG: w11 = 0x0f101112
CHECK-AARCH64-DAG: w12 = 0x10111213
CHECK-AARCH64-DAG: w13 = 0x11121314
CHECK-AARCH64-DAG: w14 = 0x12131415
CHECK-AARCH64-DAG: w15 = 0x13141516
CHECK-AARCH64-DAG: w16 = 0x14151617
CHECK-AARCH64-DAG: w17 = 0x15161718
CHECK-AARCH64-DAG: w18 = 0x16171819
CHECK-AARCH64-DAG: w19 = 0x1718191a
CHECK-AARCH64-DAG: w20 = 0x18191a1b
CHECK-AARCH64-DAG: w21 = 0x191a1b1c
CHECK-AARCH64-DAG: w22 = 0x1a1b1c1d
CHECK-AARCH64-DAG: w23 = 0x1b1c1d1e
CHECK-AARCH64-DAG: w24 = 0x1c1d1e1f
CHECK-AARCH64-DAG: w25 = 0x1d1e1f20
CHECK-AARCH64-DAG: w26 = 0x1e1f2021
CHECK-AARCH64-DAG: w27 = 0x1f202122
CHECK-AARCH64-DAG: w28 = 0x20212223
CHECK-AARCH64-DAG: w29 = 0x21222324
CHECK-AARCH64-DAG: w30 = 0x22232425
CHECK-AARCH64-DAG: w31 = 0x23242526
"""

        self.server.responder = MyResponder(
            "aarch64",
            has_target_xml,
            GPRS,
            "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;",
        )

        target = self.createTarget("basic_eh_frame-aarch64.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # test all registers
        self.filecheck(
            "register read --all",
            __file__,
            filecheck_options="--check-prefix=CHECK-AARCH64",
        )

        # test generic aliases
        self.match("register read arg1", ["x0 = 0x0001020304050607"])
        self.match("register read arg2", ["x1 = 0x0102030405060708"])
        self.match("register read fp", ["x29 = 0x1d1e1f2021222324"])
        self.match("register read lr", ["x30 = 0x1e1f202122232425"])
        self.match("register read ra", ["x30 = 0x1e1f202122232425"])
        self.match("register read flags", ["cpsr = 0x21222324"])

    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_regs_no_target_xml(self):
        """Test grabbing various aarch64 registers from gdbserver when there
        is no target XML."""
        self.check_aarch64_regs(False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_regs_no_register_info(self):
        """Test grabbing various aarch64 registers from gdbserver when there
        is target XML but it does not include register info."""
        self.check_aarch64_regs(True)

    def check_i386_regs(self, has_target_xml):
        GPRS = """
CHECK-I386-DAG: eax = 0x04030201
CHECK-I386-DAG: ecx = 0x14131211
CHECK-I386-DAG: edx = 0x24232221
CHECK-I386-DAG: ebx = 0x34333231
CHECK-I386-DAG: esp = 0x44434241
CHECK-I386-DAG: ebp = 0x54535251
CHECK-I386-DAG: esi = 0x64636261
CHECK-I386-DAG: edi = 0x74737271
CHECK-I386-DAG: eip = 0x84838281
CHECK-I386-DAG: eflags = 0x94939291
CHECK-I386-DAG: cs = 0xa4a3a2a1
CHECK-I386-DAG: ss = 0xb4b3b2b1
CHECK-I386-DAG: ds = 0xc4c3c2c1
CHECK-I386-DAG: es = 0xd4d3d2d1
CHECK-I386-DAG: fs = 0xe4e3e2e1
CHECK-I386-DAG: gs = 0xf4f3f2f1
"""

        SUPPL = """
CHECK-I386-DAG: ax = 0x0201
CHECK-I386-DAG: cx = 0x1211
CHECK-I386-DAG: dx = 0x2221
CHECK-I386-DAG: bx = 0x3231
CHECK-I386-DAG: sp = 0x4241
CHECK-I386-DAG: bp = 0x5251
CHECK-I386-DAG: si = 0x6261
CHECK-I386-DAG: di = 0x7271

CHECK-I386-DAG: ah = 0x02
CHECK-I386-DAG: ch = 0x12
CHECK-I386-DAG: dh = 0x22
CHECK-I386-DAG: bh = 0x32

CHECK-I386-DAG: al = 0x01
CHECK-I386-DAG: cl = 0x11
CHECK-I386-DAG: dl = 0x21
CHECK-I386-DAG: bl = 0x31
CHECK-I386-DAG: spl = 0x41
CHECK-I386-DAG: bpl = 0x51
CHECK-I386-DAG: sil = 0x61
CHECK-I386-DAG: dil = 0x71
"""

        self.server.responder = MyResponder(
            "i386",
            has_target_xml,
            GPRS,
            "T02thread:1ff0d;threads:1ff0d;thread-pcs:000000010001bc00;07:0102030405060708;10:1112131415161718;",
        )

        target = self.createTarget("basic_eh_frame-i386.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        # test all registers
        self.filecheck(
            "register read --all",
            __file__,
            filecheck_options="--check-prefix=CHECK-I386",
        )

        # test generic aliases
        self.match("register read fp", ["ebp = 0x54535251"])
        self.match("register read sp", ["esp = 0x44434241"])
        self.match("register read pc", ["eip = 0x84838281"])
        self.match("register read flags", ["eflags = 0x94939291"])

    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_i386_regs_no_target_xml(self):
        """Test grabbing various i386 registers from gdbserver when there is
        no target XML."""
        self.check_i386_regs(False)

    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_i386_regs_no_register_info(self):
        """Test grabbing various i386 registers from gdbserver when there is
        target XML but it does not include register info."""
        self.check_i386_regs(True)