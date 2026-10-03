"""
Test lldb-dap launch request.
"""

import os

from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_sourcePath(DAPTestCaseBase):
    """
    Tests the "sourcePath" will set the target.source-map.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_dir = os.path.dirname(program)
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(program=program, sourcePath=program_dir)
        )
        session.verify_process_exited(after=process_event)

        output = session.get_console()
        self.assertTrue(output and len(output) > 0, "expect console output")
        prefix = '(lldb) settings set target.source-map "." '
        found = False
        for line in output.splitlines():
            if line.startswith(prefix):
                found = True
                quoted_path = f'"{program_dir}"'
                self.assertEqual(
                    quoted_path,
                    line[len(prefix) :],
                    f"lldb-dap working dir {quoted_path} == {line[6:]}",
                )
        self.assertTrue(found, 'found "sourcePath" in console output')
