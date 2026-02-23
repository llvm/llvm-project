"""
Test lldb-dap launch request.
"""

import lldbdap_testcase
import tempfile


class TestDAP_launch_stdio_redirection(lldbdap_testcase.DAPTestCaseBase):
    """
    Test stdio redirection.
    """

    def test(self):
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        with tempfile.NamedTemporaryFile("rt") as f:
            self.launch_and_configurationDone(program, stdio=[None, f.name])
            self.verify_process_exited()
            lines = f.readlines()
            self.assertIn(
                program, lines[0], "make sure program path is in first argument"
            )
