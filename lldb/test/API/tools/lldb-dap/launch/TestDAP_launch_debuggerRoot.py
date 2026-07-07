"""
Test lldb-dap launch request.
"""

import os

from lldbsuite.test import lldbplatformutil
from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_debuggerRoot(DAPTestCaseBase):
    """
    Tests the "debuggerRoot" will change the working directory of
    the lldb-dap debug adapter.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_parent_dir = os.path.realpath(os.path.dirname(os.path.dirname(program)))

        var = "%cd%" if lldbplatformutil.getHostPlatform() == "windows" else "$PWD"
        init_commands = [f"platform shell echo cwd = {var}"]

        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(
                program=program,
                debuggerRoot=program_parent_dir,
                initCommands=init_commands,
            )
        )
        session.verify_process_exited(after=process_event)

        output = session.get_console()
        self.assertTrue(output and len(output) > 0, "expect console output")

        prefix = "cwd = "
        cwd_lines = [line for line in output.splitlines() if line.startswith(prefix)]
        self.assertEqual(
            len(cwd_lines), 1, "expected exactly one cwd line in console output"
        )
        self.assertEqual(
            cwd_lines[0].strip()[len(prefix) :],
            program_parent_dir,
            f"lldb-dap working dir mismatch: expected '{program_parent_dir}', "
            f"got '{cwd_lines[0]}'",
        )
