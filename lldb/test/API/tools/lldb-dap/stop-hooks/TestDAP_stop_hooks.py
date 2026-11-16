"""
Test stop hooks
"""

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_stop_hooks(DAPTestCaseBase):
    def test_stop_hooks_before_run(self):
        """
        Test that there is no race condition between lldb-dap and
        stop hooks executor
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        launch_args = LaunchArgs(
            program, preRunCommands=["target stop-hook add -o help"]
        )
        with session.configure(launch_args) as ctx:
            breakpoint_ids = session.resolve_function_breakpoints(["main"])
        # This request hangs if the race happens, because, in that case, the
        # command interpreter is in synchronous mode while lldb-dap expects
        # it to be in asynchronous mode, so, the process doesn't send the stop
        # event to "lldb.Debugger" listener (which is monitored by lldb-dap).
        session.verify_stopped_on_breakpoint(breakpoint_ids, after=ctx.process_event)

        session.continue_to_exit()
