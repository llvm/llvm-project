"""
Test lldb-vscode setBreakpoints request
"""


import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_setExceptionBreakpoints(lldbvscode_testcase.VSCodeTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_functionality(self):
        """Tests setting and clearing exception breakpoints.
        This packet is a bit tricky on the debug adaptor side since there
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
        # Visual Studio Code Debug Adaptors have no way to specify the file
        # without launching or attaching to a process, so we must start a
        # process in order to be able to set breakpoints.
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        filters = ["cpp_throw", "cpp_catch"]
        response = self.vscode.request_setExceptionBreakpoints(filters)
        if response:
            self.assertTrue(response["success"])

        self.continue_to_exception_breakpoint("C++ Throw")
        self.continue_to_exception_breakpoint("C++ Catch")
