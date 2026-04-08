"""
Tests debugserver support for MultiMemRead.
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
        packet = reply[0].strip()
        response = reply[1].strip()

        self.assertTrue(packet.startswith("packet: "))
        self.assertTrue(response.startswith("response: "))
        return response[len("response: ") :]

    def check_invalid_packet(self, packet_str):
        reply = self.send_process_packet("packet_str")
        self.assertEqual(reply, "E03")

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
        mem_address = mem_address_var.GetValueAsUnsigned() + 42

        # no ":"
        self.check_invalid_packet("MultiMemRead")
        # missing ranges
        self.check_invalid_packet("MultiMemRead:")
        # needs at least one range
        self.check_invalid_packet("MultiMemRead:ranges:")
        # needs at least one range
        self.check_invalid_packet("MultiMemRead:ranges:,")
        # a range is a pair of numbers
        self.check_invalid_packet("MultiMemRead:ranges:10")
        # a range is a pair of numbers
        self.check_invalid_packet("MultiMemRead:ranges:10,")
        # range list must end with ;
        self.check_invalid_packet("MultiMemRead:ranges:10,2")
        self.check_invalid_packet("MultiMemRead:ranges:10,2,")
        self.check_invalid_packet("MultiMemRead:ranges:10,2,3")
        # ranges are pairs of numbers.
        self.check_invalid_packet("MultiMemRead:ranges:10,2,3;")
        # unrecognized field
        self.check_invalid_packet("MultiMemRead:ranges:10,2;blah:;")
        # unrecognized field
        self.check_invalid_packet("MultiMemRead:blah:;ranges:10,2;")

        # Zero-length reads are ok.
        reply = self.send_process_packet("MultiMemRead:ranges:0,0;")
        self.assertEqual(reply, "0;")

        # Debugserver is permissive with trailing commas.
        reply = self.send_process_packet("MultiMemRead:ranges:10,2,;")
        self.assertEqual(reply, "0;")
        reply = self.send_process_packet(f"MultiMemRead:ranges:{mem_address:x},2,;")
        self.assertEqual(reply, "2;ab")

        reply = self.send_process_packet("MultiMemRead:ranges:10,2;")
        self.assertEqual(reply, "0;")
        reply = self.send_process_packet(f"MultiMemRead:ranges:{mem_address:x},0;")
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

        # Test zero length in the middle.
        reply = self.send_process_packet(
            f"MultiMemRead:ranges:{mem_address:x},2,{mem_address+2:x},0,{mem_address+6:x},8;"
        )
        self.assertEqual(reply, "2,0,8;abghijklmn")
        # Test zero length in the end.
        reply = self.send_process_packet(
            f"MultiMemRead:ranges:{mem_address:x},2,{mem_address+2:x},4,{mem_address+6:x},0;"
        )
        self.assertEqual(reply, "2,4,0;abcdef")
