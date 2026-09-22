"""
Test the "process continue --reverse" and "--forward" options
when reverse-continue is not supported.
"""


import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestReverseContinueCommandNotSupported(TestBase):
    def test_reverse_continue_not_supported(self):
        target = self.connect()

        # Set breakpoint and reverse-continue
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        self.assertTrue(trigger_bkpt, VALID_BREAKPOINT)
        # `process continue --forward` should work.
        self.expect(
            "process continue --forward",
            substrs=["stop reason = breakpoint {0}.1".format(trigger_bkpt.GetID())],
        )
        self.expect(
            "process continue --reverse",
            error=True,
            # This error is "<plugin name> does not support...". The plugin name changes
            # between platforms.
            substrs=["does not support reverse execution of processes"],
        )

    def test_reverse_continue_forward_and_reverse(self):
        self.connect()

        self.expect(
            "process continue --forward --reverse",
            error=True,
            substrs=["invalid combination of options for the given command"],
        )

    def connect(self):
        self.build()
        target, _, _, _ = lldbutil.run_to_name_breakpoint(self, "main")
        return target
