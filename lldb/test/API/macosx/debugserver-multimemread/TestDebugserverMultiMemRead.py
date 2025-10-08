"""
Tests the exit code/description coming from the debugserver.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
@skipIfOutOfTreeDebugserver
class TestCase(TestBase):
    def send_process_packet(self, packet_str):
        self.runCmd(f"proc plugin packet send {packet_str}", check=False)
        # The output is of the form:
        #  packet: <packet_str>
        #  response: <response>
        reply = self.res.GetOutput().split("\n")
        reply[0] = reply[0].strip()
        reply[1] = reply[1].strip()

        self.assertTrue(reply[0].startswith("packet: "), reply[0])
        self.assertTrue(reply[1].startswith("response: "))
        return reply[1][len("response: ") :]

    def test_packets(self):
        self.build()
        source_file = lldb.SBFileSpec("main.c")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", source_file
        )

        reply = self.send_process_packet("qSupported")
        self.assertIn("MultiMemRead+", reply)

        mem_address_var = thread.frames[0].FindVariable("memory")
        self.assertTrue(mem_address_var)
        mem_address = int(mem_address_var.GetValue(), 16)

        reply = self.send_process_packet("MultiMemRead:")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:10")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:10,")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:10,2")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:10,2,")
        self.assertEqual(reply, "E03")
        reply = self.send_process_packet("MultiMemRead:ranges:10,2,;")
        self.assertEqual(reply, "0;")  # Debugserver is permissive with trailing commas.
        reply = self.send_process_packet("MultiMemRead:ranges:10,2;")
        self.assertEqual(reply, "0;")
        reply = self.send_process_packet(f"MultiMemRead:ranges:{mem_address:x},0;")
        self.assertEqual(reply, "0;")
        reply = self.send_process_packet(
            f"MultiMemRead:ranges:{mem_address:x},0;options:;"
        )
        self.assertEqual(reply, "0;")
        reply = self.send_process_packet(f"MultiMemRead:ranges:{mem_address:x},2;")
        self.assertEqual(reply, "2;ab")
        reply = self.send_process_packet(
            f"MultiMemRead:ranges:{mem_address:x},2,{mem_address+2:x},4;"
        )
        self.assertEqual(reply, "2,4;abcdef")
        reply = self.send_process_packet(
            f"MultiMemRead:ranges:{mem_address:x},2,{mem_address+2:x},4,{mem_address+6:x},8;"
        )
        self.assertEqual(reply, "2,4,8;abcdefghijklmn")
