"""
Test stop hooks
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbvscode_testcase


class TestVSCode_stop_hooks(lldbvscode_testcase.VSCodeTestCaseBase):
    @skipIfRemote
    def test_stop_hooks_before_run(self):
        """
        Test that there is no race condition between lldb-vscode and
        stop hooks executor
        """
        program = self.getBuildArtifact("a.out")
        preRunCommands = ["target stop-hook add -o help"]
        self.build_and_launch(program, stopOnEntry=True, preRunCommands=preRunCommands)

        # The first stop is on entry.
        self.continue_to_next_stop()

        breakpoint_ids = self.set_function_breakpoints(["main"])
        # This request hangs if the race happens, because, in that case, the
        # command interpreter is in synchronous mode while lldb-vscode expects
        # it to be in asynchronous mode, so, the process doesn't send the stop
        # event to "lldb.Debugger" listener (which is monitored by lldb-vscode).
        self.continue_to_breakpoints(breakpoint_ids)

        self.continue_to_exit()
