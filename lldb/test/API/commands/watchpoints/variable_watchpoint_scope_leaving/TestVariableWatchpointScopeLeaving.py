"""Test that variable watchpoints emit message leaving its scope."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbwatchpointutils import *


class TestVariableWatchpointScopeLeaving(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def do_test(self, wp_type, wp_mode):
        self.build()
        (target, process, cur_thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        # Set a watchpoint for 'local'
        self.expect(
            f"{get_set_watchpoint_CLI_command(WatchpointCLICommandVariant.VARIABLE, wp_type, wp_mode)} local",
            WATCHPOINT_CREATED,
            substrs=[
                "Watchpoint created",
                "size = 4",
                f"type = {wp_type.value[0]}",
            ],
        )

        # Resume process execution. We should return from the function and notify the user that the watchpoint
        # has left its scope.
        self.runCmd("process continue")
        self.assertIn(
            self.res.GetError(),
            "warning: Watchpoint 1 is leaving its scope! Disabling this watchpoint.",
        )

    @expectedFailureAll(archs="^riscv.*")
    def test_hardware_variable_watchpoint(self):
        self.do_test(WatchpointType.MODIFY, lldb.eWatchpointModeHardware)

    def test_software_variable_watchpoint(self):
        self.do_test(WatchpointType.MODIFY, lldb.eWatchpointModeSoftware)
