"""
Confirm that lldb modify watchpoints only stop
when the value being watched changes.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


@skipIfWindows
class ModifyWatchpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(archs="^riscv.*")
    def test_modify_hardware_watchpoint(self):
        self.do_modify_watchpoint(WatchpointType.MODIFY, lldb.eWatchpointModeHardware)

    def test_modify_software_watchpoint(self):
        self.do_modify_watchpoint(WatchpointType.MODIFY, lldb.eWatchpointModeSoftware)

    def do_modify_watchpoint(self, wp_type, wp_mode):
        """Test that a modify watchpoint only stops when the value changes."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file
        )

        self.runCmd(
            f"{get_set_watchpoint_CLI_command(WatchpointCLICommandVariant.VARIABLE, wp_type, wp_mode)} value"
        )
        process.Continue()
        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(frame.locals["value"][0].GetValueAsUnsigned(), 10)

        process.Continue()
        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(frame.locals["value"][0].GetValueAsUnsigned(), 5)

        process.Continue()
        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(frame.locals["value"][0].GetValueAsUnsigned(), 7)

        process.Continue()
        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(frame.locals["value"][0].GetValueAsUnsigned(), 9)
