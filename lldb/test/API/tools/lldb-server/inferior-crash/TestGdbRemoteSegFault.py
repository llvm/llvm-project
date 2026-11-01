import re
import struct

import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteSegFault(gdbremote_testcase.GdbRemoteTestCaseBase):
    GDB_REMOTE_STOP_CODE_BAD_ACCESS = 0x91

    def inferior_seg_fault_received(self, expected_signo):
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["segfault"])
        self.assertIsNotNone(procs)

        self.test_sequence.add_log_lines(
            [
                "read packet: $vCont;c#a8",
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2}).*#[0-9a-fA-F]{2}$",
                    "capture": {1: "hex_exit_code"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        hex_exit_code = context.get("hex_exit_code")
        self.assertIsNotNone(hex_exit_code)
        self.assertEqual(int(hex_exit_code, 16), expected_signo)

    @requireSignals
    def test_inferior_seg_fault_received(self):
        self.build()
        if self.platformIsDarwin():
            self.inferior_seg_fault_received(self.GDB_REMOTE_STOP_CODE_BAD_ACCESS)
        else:
            self.inferior_seg_fault_received(lldbutil.get_signal_number("SIGSEGV"))

    @skipUnlessPlatform(["linux"])
    @skipIf(archs=no_match(["x86_64", "i386"]))
    def test_siginfo_after_memory_allocation(self):
        """Test that allocating memory (which runs a syscall in the inferior)
        does not change the siginfo of the thread."""
        self.build()
        self.inferior_seg_fault_received(lldbutil.get_signal_number("SIGSEGV"))

        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $_M1000,rw#00",
                {"direction": "send", "regex": r"^\$[0-9a-f]+#[0-9a-fA-F]{2}$"},
                "read packet: $qXfer:siginfo:read::0,80:#00",
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$l(.*)#[0-9a-fA-F]{2}$", re.MULTILINE | re.DOTALL
                    ),
                    "capture": {1: "siginfo"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        siginfo = self.decode_gdbremote_binary(context["siginfo"]).encode("latin1")
        (signo,) = struct.unpack_from("=i", siginfo)
        self.assertEqual(signo, lldbutil.get_signal_number("SIGSEGV"))
