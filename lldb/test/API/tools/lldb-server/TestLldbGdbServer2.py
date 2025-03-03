"""
Test case for testing the gdbremote protocol.

Tests run against debugserver and lldb-server (llgs).
lldb-server tests run where the lldb-server exe is
available.

The tests are split between the LldbGdbServerTestCase and
LldbGdbServerTestCase2 classes to avoid timeouts in the
test suite.
"""

import binascii
import itertools
import struct

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.support import seven
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbdwarf import *
from lldbsuite.test import lldbutil, lldbplatformutil


class LldbGdbServerTestCase2(
    gdbremote_testcase.GdbRemoteTestCaseBase, DwarfOpcodeParser
):
    @skipIfWindows  # No pty support to test any inferior output
    def test_m_packet_reads_memory(self):
        self.build()
        self.set_inferior_startup_launch()
        # This is the memory we will write into the inferior and then ensure we
        # can read back with $m.
        MEMORY_CONTENTS = "Test contents 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "set-message:%s" % MEMORY_CONTENTS,
                "get-data-address-hex:g_message",
                "sleep:5",
            ]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"data address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "message_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Grab contents from the inferior.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $m{0:x},{1:x}#00".format(
                    message_address, len(MEMORY_CONTENTS)
                ),
                {
                    "direction": "send",
                    "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
                    "capture": {1: "read_contents"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        self.assertIsNotNone(context.get("read_contents"))
        read_contents = seven.unhexlify(context.get("read_contents"))
        self.assertEqual(read_contents, MEMORY_CONTENTS)

    def test_qMemoryRegionInfo_is_supported(self):
        self.build()
        self.set_inferior_startup_launch()
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior()

        # Ask if it supports $qMemoryRegionInfo.
        self.test_sequence.add_log_lines(
            ["read packet: $qMemoryRegionInfo#00", "send packet: $OK#00"], True
        )
        self.expect_gdbremote_sequence()

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_code_address_as_executable(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-code-address-hex:hello", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"code address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "code_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the code address.
        self.assertIsNotNone(context.get("code_address"))
        code_address = int(context.get("code_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(code_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure code address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("x", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(code_address, mem_region_dict)

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_stack_address_as_rw(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-stack-address-hex:", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"stack address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "stack_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("stack_address"))
        stack_address = int(context.get("stack_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(stack_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("w", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(stack_address, mem_region_dict)

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_heap_address_as_rw(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-heap-address-hex:", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"heap address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "heap_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("heap_address"))
        heap_address = int(context.get("heap_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(heap_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("w", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(heap_address, mem_region_dict)

    def breakpoint_set_and_remove_work(self, want_hardware):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "get-code-address-hex:hello",
                "sleep:1",
                "call-function:hello",
            ]
        )

        # Run the process
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            [  # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the function call entry point.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"code address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "function_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info - we need endian of target to handle register
        # value conversions.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)
        (pc_lldb_reg_index, pc_reg_info) = self.find_pc_reg_info(reg_infos)
        self.assertIsNotNone(pc_lldb_reg_index)
        self.assertIsNotNone(pc_reg_info)

        # Grab the function address.
        self.assertIsNotNone(context.get("function_address"))
        function_address = int(context.get("function_address"), 16)

        # Get current target architecture
        target_arch = self.getArchitecture()

        # Set the breakpoint.
        if target_arch in ["arm", "arm64", "aarch64"]:
            # TODO: Handle case when setting breakpoint in thumb code
            BREAKPOINT_KIND = 4
        else:
            BREAKPOINT_KIND = 1

        # Set default packet type to Z0 (software breakpoint)
        z_packet_type = 0

        # If hardware breakpoint is requested set packet type to Z1
        if want_hardware:
            z_packet_type = 1

        self.reset_test_sequence()
        self.add_set_breakpoint_packets(
            function_address,
            z_packet_type,
            do_continue=True,
            breakpoint_kind=BREAKPOINT_KIND,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the stop signal reported was the breakpoint signal number.
        stop_signo = context.get("stop_signo")
        self.assertIsNotNone(stop_signo)
        self.assertEqual(int(stop_signo, 16), lldbutil.get_signal_number("SIGTRAP"))

        # Ensure we did not receive any output.  If the breakpoint was not set, we would
        # see output (from a launched process with captured stdio) printing a hello, world message.
        # That would indicate the breakpoint didn't take.
        self.assertEqual(len(context["O_content"]), 0)

        # Verify that the PC for the main thread is where we expect it - right at the breakpoint address.
        # This acts as a another validation on the register reading code.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                # Print the PC.  This should match the breakpoint address.
                "read packet: $p{0:x}#00".format(pc_lldb_reg_index),
                # Capture $p results.
                {
                    "direction": "send",
                    "regex": r"^\$([0-9a-fA-F]+)#",
                    "capture": {1: "p_response"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the PC is where we expect.  Note response is in endianness of
        # the inferior.
        p_response = context.get("p_response")
        self.assertIsNotNone(p_response)

        # Convert from target endian to int.
        returned_pc = lldbgdbserverutils.unpack_register_hex_unsigned(
            endian, p_response
        )
        self.assertEqual(returned_pc, function_address)

        # Verify that a breakpoint remove and continue gets us the expected
        # output.
        self.reset_test_sequence()

        # Add breakpoint remove packets
        self.add_remove_breakpoint_packets(
            function_address, z_packet_type, breakpoint_kind=BREAKPOINT_KIND
        )

        self.test_sequence.add_log_lines(
            [
                # Continue running.
                "read packet: $c#63",
                # We should now receive the output from the call.
                {"type": "output_match", "regex": r"^hello, world\r\n$"},
                # And wait for program completion.
                {"direction": "send", "regex": r"^\$W00(.*)#[0-9a-fA-F]{2}$"},
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipIfWindows  # No pty support to test any inferior output
    def test_software_breakpoint_set_and_remove_work(self):
        if self.getArchitecture() == "arm":
            # TODO: Handle case when setting breakpoint in thumb code
            self.build(dictionary={"CFLAGS_EXTRAS": "-marm"})
        else:
            self.build()
        self.set_inferior_startup_launch()
        self.breakpoint_set_and_remove_work(want_hardware=False)

    @skipUnlessPlatform(oslist=["linux"])
    @skipIf(archs=no_match(["arm", "aarch64"]))
    def test_hardware_breakpoint_set_and_remove_work(self):
        if self.getArchitecture() == "arm":
            # TODO: Handle case when setting breakpoint in thumb code
            self.build(dictionary={"CFLAGS_EXTRAS": "-marm"})
        else:
            self.build()
        self.set_inferior_startup_launch()
        self.breakpoint_set_and_remove_work(want_hardware=True)

    def get_qSupported_dict(self, features=[]):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets(features)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Retrieve the qSupported features.
        return self.parse_qSupported_response(context)

    def test_qSupported_returns_known_stub_features(self):
        supported_dict = self.get_qSupported_dict()
        self.assertIsNotNone(supported_dict)
        self.assertGreater(len(supported_dict), 0)

    def test_qSupported_auvx(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:auxv:read", "-"), expected)

    def test_qSupported_libraries_svr4(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:libraries-svr4:read", "-"), expected)

    def test_qSupported_siginfo_read(self):
        expected = (
            "+" if lldbplatformutil.getPlatform() in ["freebsd", "linux"] else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:siginfo:read", "-"), expected)

    def test_qSupported_QPassSignals(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("QPassSignals", "-"), expected)

    @add_test_categories(["fork"])
    def test_qSupported_fork_events(self):
        supported_dict = self.get_qSupported_dict(["multiprocess+", "fork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "+")
        self.assertEqual(supported_dict.get("fork-events", "-"), "+")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    @add_test_categories(["fork"])
    def test_qSupported_fork_events_without_multiprocess(self):
        supported_dict = self.get_qSupported_dict(["fork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "-")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    @add_test_categories(["fork"])
    def test_qSupported_vfork_events(self):
        supported_dict = self.get_qSupported_dict(["multiprocess+", "vfork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "+")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "+")

    @add_test_categories(["fork"])
    def test_qSupported_vfork_events_without_multiprocess(self):
        supported_dict = self.get_qSupported_dict(["vfork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "-")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    # We need to be able to self.runCmd to get cpuinfo,
    # which is not possible when using a remote platform.
    @skipIfRemote
    def test_qSupported_memory_tagging(self):
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(
            supported_dict.get("memory-tagging", "-"),
            "+" if self.isAArch64MTE() else "-",
        )

    @skipIfWindows  # No pty support to test any inferior output
    def test_written_M_content_reads_back_correctly(self):
        self.build()
        self.set_inferior_startup_launch()

        TEST_MESSAGE = "Hello, memory"

        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "set-message:xxxxxxxxxxxxxX",
                "get-data-address-hex:g_message",
                "sleep:1",
                "print-message:",
            ]
        )
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"data address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "message_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Hex-encode the test message, adding null termination.
        hex_encoded_message = seven.hexlify(TEST_MESSAGE)

        # Write the message to the inferior. Verify that we can read it with the hex-encoded (m)
        # and binary (x) memory read packets.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $M{0:x},{1:x}:{2}#00".format(
                    message_address, len(TEST_MESSAGE), hex_encoded_message
                ),
                "send packet: $OK#00",
                "read packet: $m{0:x},{1:x}#00".format(
                    message_address, len(TEST_MESSAGE)
                ),
                "send packet: ${0}#00".format(hex_encoded_message),
                "read packet: $x{0:x},{1:x}#00".format(
                    message_address, len(TEST_MESSAGE)
                ),
                "send packet: ${0}#00".format(TEST_MESSAGE),
                "read packet: $m{0:x},4#00".format(message_address),
                "send packet: ${0}#00".format(hex_encoded_message[0:8]),
                "read packet: $x{0:x},4#00".format(message_address),
                "send packet: ${0}#00".format(TEST_MESSAGE[0:4]),
                "read packet: $c#63",
                {
                    "type": "output_match",
                    "regex": r"^message: (.+)\r\n$",
                    "capture": {1: "printed_message"},
                },
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        printed_message = context.get("printed_message")
        self.assertIsNotNone(printed_message)
        self.assertEqual(printed_message, TEST_MESSAGE + "X")

    # Note: as of this moment, a hefty number of the GPR writes are failing with E32 (everything except rax-rdx, rdi, rsi, rbp).
    # Come back to this.  I have the test rigged to verify that at least some
    # of the bit-flip writes work.
    def test_P_writes_all_gpr_registers(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start inferior debug session, grab all register info.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:2"])
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Process register infos.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        # Process endian.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Pull out the register infos that we think we can bit flip
        # successfully,.
        gpr_reg_infos = [
            reg_info
            for reg_info in reg_infos
            if self.is_bit_flippable_register(reg_info)
        ]
        self.assertGreater(len(gpr_reg_infos), 0)

        # Write flipped bit pattern of existing value to each register.
        (successful_writes, failed_writes) = self.flip_all_bits_in_each_register_value(
            gpr_reg_infos, endian
        )
        self.trace(
            "successful writes: {}, failed writes: {}".format(
                successful_writes, failed_writes
            )
        )
        self.assertGreater(successful_writes, 0)

    # Note: as of this moment, a hefty number of the GPR writes are failing
    # with E32 (everything except rax-rdx, rdi, rsi, rbp).
    @skipIfWindows
    def test_P_and_p_thread_suffix_work(self):
        self.build()
        self.set_inferior_startup_launch()

        # Startup the inferior with three threads.
        _, threads = self.launch_with_threads(3)

        self.reset_test_sequence()
        self.add_thread_suffix_request_packets()
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        reg_index = self.select_modifiable_register(reg_infos)
        self.assertIsNotNone(reg_index)
        reg_byte_size = int(reg_infos[reg_index]["bitsize"]) // 8
        self.assertGreater(reg_byte_size, 0)

        expected_reg_values = []
        register_increment = 1
        next_value = None

        # Set the same register in each of 3 threads to a different value.
        # Verify each one has the unique value.
        for thread in threads:
            # If we don't have a next value yet, start it with the initial read
            # value + 1
            if not next_value:
                # Read pre-existing register value.
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    [
                        "read packet: $p{0:x};thread:{1:x}#00".format(
                            reg_index, thread
                        ),
                        {
                            "direction": "send",
                            "regex": r"^\$([0-9a-fA-F]+)#",
                            "capture": {1: "p_response"},
                        },
                    ],
                    True,
                )
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)

                # Set the next value to use for writing as the increment plus
                # current value.
                p_response = context.get("p_response")
                self.assertIsNotNone(p_response)
                next_value = lldbgdbserverutils.unpack_register_hex_unsigned(
                    endian, p_response
                )

            # Set new value using P and thread suffix.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $P{0:x}={1};thread:{2:x}#00".format(
                        reg_index,
                        lldbgdbserverutils.pack_register_hex(
                            endian, next_value, byte_size=reg_byte_size
                        ),
                        thread,
                    ),
                    "send packet: $OK#00",
                ],
                True,
            )
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Save the value we set.
            expected_reg_values.append(next_value)

            # Increment value for next thread to use (we want them all
            # different so we can verify they wrote to each thread correctly
            # next.)
            next_value += register_increment

        # Revisit each thread and verify they have the expected value set for
        # the register we wrote.
        thread_index = 0
        for thread in threads:
            # Read pre-existing register value.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $p{0:x};thread:{1:x}#00".format(reg_index, thread),
                    {
                        "direction": "send",
                        "regex": r"^\$([0-9a-fA-F]+)#",
                        "capture": {1: "p_response"},
                    },
                ],
                True,
            )
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Get the register value.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)
            read_value = lldbgdbserverutils.unpack_register_hex_unsigned(
                endian, p_response
            )

            # Make sure we read back what we wrote.
            self.assertEqual(read_value, expected_reg_values[thread_index])
            thread_index += 1

    @skipUnlessPlatform(oslist=["freebsd", "linux"])
    @add_test_categories(["llgs"])
    def test_qXfer_siginfo_read(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["thread:segfault", "thread:new", "sleep:10"]
        )
        self.test_sequence.add_log_lines(["read packet: $c#63"], True)
        self.expect_gdbremote_sequence()

        # Run until SIGSEGV comes in.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "signo", 2: "thread_id"},
                }
            ],
            True,
        )

        # Figure out which thread crashed.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        self.assertEqual(
            int(context["signo"], 16), lldbutil.get_signal_number("SIGSEGV")
        )
        crashing_thread = int(context["thread_id"], 16)

        # Grab siginfo for the crashing thread.
        self.reset_test_sequence()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            [
                "read packet: $Hg{:x}#00".format(crashing_thread),
                "send packet: $OK#00",
                "read packet: $qXfer:siginfo:read::0,80:#00",
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$([^E])(.*)#[0-9a-fA-F]{2}$", re.MULTILINE | re.DOTALL
                    ),
                    "capture": {1: "response_type", 2: "content_raw"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure we end up with all data in one packet.
        self.assertEqual(context.get("response_type"), "l")

        # Decode binary data.
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        content = self.decode_gdbremote_binary(content_raw).encode("latin1")

        # Decode siginfo_t.
        process_info = self.parse_process_info_response(context)
        pad = ""
        if process_info["ptrsize"] == "8":
            pad = "i"
        signo_idx = 0
        errno_idx = 1
        code_idx = 2
        addr_idx = -1
        SEGV_MAPERR = 1
        if process_info["ostype"] == "linux":
            # si_signo, si_errno, si_code, [pad], _sifields._sigfault.si_addr
            format_str = "iii{}P".format(pad)
        elif process_info["ostype"].startswith("freebsd"):
            # si_signo, si_errno, si_code, si_pid, si_uid, si_status, si_addr
            format_str = "iiiiiiP"
        elif process_info["ostype"].startswith("netbsd"):
            # _signo, _code, _errno, [pad], _reason._fault._addr
            format_str = "iii{}P".format(pad)
            errno_idx = 2
            code_idx = 1
        else:
            assert False, "unknown ostype"

        decoder = struct.Struct(format_str)
        decoded = decoder.unpack(content[: decoder.size])
        self.assertEqual(decoded[signo_idx], lldbutil.get_signal_number("SIGSEGV"))
        self.assertEqual(decoded[errno_idx], 0)  # si_errno
        self.assertEqual(decoded[code_idx], SEGV_MAPERR)  # si_code
        self.assertEqual(decoded[addr_idx], 0)  # si_addr
