
import json

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

def invert_byte_order(a):
    return "".join(reversed([a[i:i+2] for i in range(0, len(a),2)]))

def decode_hex(a):
    return int(invert_byte_order(a), 16)

def encode_hex(a):
    return invert_byte_order("%016x" % a)

class TestGdbRemoteThreadsInfoMemory(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(archs=no_match(["x86_64"]))
    def threadsInfoStackCorrect(self):
        procs = self.prep_debug_monitor_and_inferior()

        self.add_register_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)
        # Index register info entries by name.
        reg_infos = {info['name']: info for info in reg_infos}

        # Send vCont packet to resume the inferior.
        self.test_sequence.add_log_lines(["read packet: $vCont;c#a8",
                                          {"direction": "send",
                                           "regex": r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$",
                                           "capture": {1: "hex_exit_code"}},
                                          ],
                                         True)

        # Send g packet to retrieve the register bank
        self.test_sequence.add_log_lines(
                [
                    "read packet: $jThreadsInfo#c1",
                    {
                        "direction": "send",
                        "regex": r"^\$(.*)#[0-9a-fA-F]{2}$",
                        "capture": {
                            1: "threads_info"}},
                ],
                True)

        context = self.expect_gdbremote_sequence()
        threads_info = context["threads_info"]
        threads_info = json.loads(self.decode_gdbremote_binary(threads_info))
        self.assertEqual(1, len(threads_info))
        thread = threads_info[0]

        # Read the stack pointer and the frame pointer from the jThreadsInfo
        # reply.
        rsp_id = reg_infos["rsp"]["lldb_register_index"]
        sp = decode_hex(thread["registers"][str(rsp_id)])
        rbp_id = reg_infos["rbp"]["lldb_register_index"]
        fp = decode_hex(thread["registers"][str(rbp_id)])

        # The top frame size is 3 words.
        self.assertEqual(sp + 3 * 8, fp)

        # Check the memory chunks.
        chunks = thread["memory"]
        self.assertEqual(3, len(chunks))
        # First memory chunk should contain everything between sp and fp.
        self.assertEqual(sp, chunks[0]["address"])
        self.assertEqual(encode_hex(6) + encode_hex(5) + encode_hex(4),
                         chunks[0]["bytes"])
        # Second chunk should be at |fp|, its return address should be 0xfeed,
        # and the next fp should 5 words away (3 values, ra and fp).
        self.assertEqual(fp, chunks[1]["address"])
        next_fp = fp + 5 * 8
        self.assertEqual(encode_hex(next_fp) + encode_hex(0xfeed),
                         chunks[1]["bytes"])
        # Third chunk at |next_fp|, the next fp is 0x1008 bytes away and
        # the ra is 0xf00d.
        self.assertEqual(next_fp, chunks[2]["address"])
        next_fp = next_fp + 0x1008
        self.assertEqual(encode_hex(next_fp) + encode_hex(0xf00d),
                         chunks[2]["bytes"])

    @expectedFailureAll(oslist=["windows"])
    @skipIfNetBSD
    @llgs_test
    def test_g_returns_correct_data_with_suffix_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.threadsInfoStackCorrect()
