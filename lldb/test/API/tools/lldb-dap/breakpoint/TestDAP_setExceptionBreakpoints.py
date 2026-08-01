"""
Test lldb-dap setExceptionBreakpoints request
"""

from lldbsuite.test.decorators import (
    skipIfTargetDoesNotSupportSharedLibraries,
    skipIfWindows,
)
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


@skipIfTargetDoesNotSupportSharedLibraries()
class TestDAP_setExceptionBreakpoints(DAPTestCaseBase):
    @skipIfWindows
    def test_functionality(self):
        """Tests setting and clearing exception breakpoints.
        This packet is a bit tricky on the debug adapter side since there
        is no "clear exception breakpoints" packet. Exception breakpoints
        are set by sending a "setExceptionBreakpoints" packet with zero or
        more exception filters. If exception breakpoints have been set
        before, any existing breakpoints must remain set, and any new
        breakpoints must be created, and any breakpoints that were in
        previous requests and are not in the current request must be
        removed. This exception tests this setting and clearing and makes
        sure things happen correctly. It doesn't test hitting breakpoints
        and the functionality of each breakpoint, like 'conditions' and
        x'hitCondition' settings.
        """
        # Visual Studio Code Debug Adapters have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        with session.configure(LaunchArgs(program)) as ctx:
            response = session.set_exception_breakpoints(
                filters=["cpp_throw", "cpp_catch"]
            )
            breakpoints = self.expect_not_none(response.body.breakpoints)
            for bp in breakpoints:
                self.assertTrue(bp.verified, True)

        session.verify_stopped_on_exception(
            expected_description=r"breakpoint 1\.1",
            expected_text=r"C\+\+ Throw",
            after=ctx.process_event,
        )
        session.continue_to_exception_breakpoint(
            expected_description=r"breakpoint 2\.1", expected_text=r"C\+\+ Catch"
        )
        session.continue_to_exit()
