import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbdwarf import *


class TestGdbRemote_qMemoryRegion(gdbremote_testcase.GdbRemoteTestCaseBase):
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
