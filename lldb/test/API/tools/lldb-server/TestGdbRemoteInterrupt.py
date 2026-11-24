import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestGdbRemoteInterrupt(gdbremote_testcase.GdbRemoteTestCaseBase):
    def test_interrupt_stops_with_sigstop_or_sigint(self):
        """The stop reply to an interrupt carries SIGSTOP or SIGINT.

        When lldb interrupts a running inferior to send a packet, those are the
        only signals GDBRemoteClientBase::ShouldStop resumes from without
        reporting the stop to the user. They have to be numbered the way the
        client numbers them for the target.
        """
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior(inferior_args=["sleep:60"])
        self.test_sequence.add_log_lines(
            [
                "read packet: $c#63",
                "read packet: {}".format(chr(3)),
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})",
                    "capture": {1: "stop_signo"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()

        signals = self.dbg.GetSelectedPlatform().GetUnixSignals()
        self.assertIn(
            int(context.get("stop_signo"), 16),
            [
                signals.GetSignalNumberFromName("SIGSTOP"),
                signals.GetSignalNumberFromName("SIGINT"),
            ],
        )
