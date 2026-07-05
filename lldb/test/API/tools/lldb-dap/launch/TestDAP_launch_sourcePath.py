"""
Test lldb-dap launch request.
"""

import lldbdap_testcase
import os


class TestDAP_launch_sourcePath(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the "sourcePath" will set the target.source-map.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_dir = os.path.dirname(program)
        self.build_and_launch(program, sourcePath=program_dir)
        self.continue_to_exit()
        output = self.get_console()
        self.assertTrue(output and len(output) > 0, "expect console output")
        lines = output.splitlines()
        prefix = '(lldb) settings set target.source-map "." '
        found = False
        for line in lines:
            if line.startswith(prefix):
                found = True
                quoted_path = '"%s"' % (program_dir)
                self.assertEqual(
                    quoted_path,
                    line[len(prefix) :],
                    "lldb-dap working dir %s == %s" % (quoted_path, line[6:]),
                )
        self.assertTrue(found, 'found "sourcePath" in console output')
