"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
from lldbsuite.test import lldbplatformutil
import lldbdap_testcase
import os


class TestDAP_launch_debuggerRoot(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the "debuggerRoot" will change the working directory of
    the lldb-dap debug adapter.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_parent_dir = os.path.realpath(os.path.dirname(os.path.dirname(program)))

        var = "%cd%" if lldbplatformutil.getHostPlatform() == "windows" else "$PWD"
        commands = [f"platform shell echo cwd = {var}"]

        self.build_and_launch(
            program, debuggerRoot=program_parent_dir, initCommands=commands
        )
        self.continue_to_exit()
        output = self.get_console()
        self.assertTrue(output and len(output) > 0, "expect console output")
        lines = output.splitlines()
        prefix = "cwd = "
        found = False
        for line in lines:
            if line.startswith(prefix):
                found = True
                self.assertEqual(
                    program_parent_dir,
                    line.strip()[len(prefix) :],
                    "lldb-dap working dir '%s' == '%s'"
                    % (program_parent_dir, line[len(prefix) :]),
                )
        self.assertTrue(found, "verified lldb-dap working directory")
