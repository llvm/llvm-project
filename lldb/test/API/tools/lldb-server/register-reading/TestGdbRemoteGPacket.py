import gdbremote_testcase
import textwrap
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

def _extract_register_value(reg_info, reg_bank, byte_order, bytes_per_entry=8):
    reg_offset = int(reg_info["offset"])*2
    reg_byte_size = int(2 * int(reg_info["bitsize"]) / 8)
    # Create slice with the contents of the register.
    reg_slice = reg_bank[reg_offset:reg_offset+reg_byte_size]

    reg_value = []
    # Wrap slice according to bytes_per_entry.
    for entry in textwrap.wrap(reg_slice, 2 * bytes_per_entry):
        # Invert the bytes order if target uses little-endian.
        if byte_order == lldb.eByteOrderLittle:
            entry = "".join(reversed([entry[i:i+2] for i in range(0,
                                          len(entry),2)]))
        reg_value.append("0x" + entry)

    return reg_value


class TestGdbRemoteGPacket(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfOutOfTreeDebugserver
    @skipUnlessDarwin # G packet not supported
    def test_g_packet(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $g#67",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "register_bank"}}],
            True)
        context = self.expect_gdbremote_sequence()
        register_bank = context.get("register_bank")
        self.assertNotEqual(register_bank[0], 'E')

        self.test_sequence.add_log_lines(
            ["read packet: $G" + register_bank + "#00",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "G_reply"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertNotEqual(context.get("G_reply")[0], 'E')

    @skipIf(archs=no_match(["x86_64"]))
    def g_returns_correct_data(self, with_suffix):
        procs = self.prep_debug_monitor_and_inferior()

        self.add_register_info_collection_packets()
        if with_suffix:
            self.add_thread_suffix_request_packets()
        self.add_threadinfo_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)
        # Index register info entries by name.
        reg_infos = {info['name']: info for info in reg_infos}

        # Gather thread info.
        if with_suffix:
            threads = self.parse_threadinfo_packets(context)
            self.assertIsNotNone(threads)
            thread_id = threads[0]
            self.assertIsNotNone(thread_id)
        else:
            thread_id = None

        # Send vCont packet to resume the inferior.
        self.test_sequence.add_log_lines(["read packet: $vCont;c#a8",
                                          {"direction": "send",
                                           "regex": r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$",
                                           "capture": {1: "hex_exit_code"}},
                                          ],
                                         True)

        # Send g packet to retrieve the register bank
        if thread_id:
            g_request = "read packet: $g;thread:{:x}#00".format(thread_id)
        else:
            g_request = "read packet: $g#00"
        self.test_sequence.add_log_lines(
            [g_request,
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "register_bank"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        reg_bank = context.get("register_bank")
        self.assertNotEqual(reg_bank[0], 'E')

        byte_order = self.get_target_byte_order()
        get_reg_value = lambda reg_name : _extract_register_value(
            reg_infos[reg_name], reg_bank, byte_order)

        self.assertEqual(['0x0102030405060708'], get_reg_value('r8'))
        self.assertEqual(['0x1112131415161718'], get_reg_value('r9'))
        self.assertEqual(['0x2122232425262728'], get_reg_value('r10'))
        self.assertEqual(['0x3132333435363738'], get_reg_value('r11'))
        self.assertEqual(['0x4142434445464748'], get_reg_value('r12'))
        self.assertEqual(['0x5152535455565758'], get_reg_value('r13'))
        self.assertEqual(['0x6162636465666768'], get_reg_value('r14'))
        self.assertEqual(['0x7172737475767778'], get_reg_value('r15'))

        self.assertEqual(
            ['0x020406080a0c0e01', '0x030507090b0d0f00'], get_reg_value('xmm8'))
        self.assertEqual(
            ['0x121416181a1c1e11', '0x131517191b1d1f10'], get_reg_value('xmm9'))
        self.assertEqual(
            ['0x222426282a2c2e21', '0x232527292b2d2f20'], get_reg_value('xmm10'))
        self.assertEqual(
            ['0x323436383a3c3e31', '0x333537393b3d3f30'], get_reg_value('xmm11'))
        self.assertEqual(
            ['0x424446484a4c4e41', '0x434547494b4d4f40'], get_reg_value('xmm12'))
        self.assertEqual(
            ['0x525456585a5c5e51', '0x535557595b5d5f50'], get_reg_value('xmm13'))
        self.assertEqual(
            ['0x626466686a6c6e61', '0x636567696b6d6f60'], get_reg_value('xmm14'))
        self.assertEqual(
            ['0x727476787a7c7e71', '0x737577797b7d7f70'], get_reg_value('xmm15'))

    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48420")
    @expectedFailureNetBSD
    def test_g_returns_correct_data_with_suffix(self):
        self.build()
        self.set_inferior_startup_launch()
        self.g_returns_correct_data(True)

    @expectedFailureAll(oslist=["freebsd"], bugnumber="llvm.org/pr48420")
    @expectedFailureNetBSD
    def test_g_returns_correct_data_no_suffix(self):
        self.build()
        self.set_inferior_startup_launch()
        self.g_returns_correct_data(False)
