"""
Test lldb-dap output events
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import re


class TestDAP_output(lldbdap_testcase.DAPTestCaseBase):
    def test_output(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        main_source_path = self.getSourcePath(source)
        lines = [line_number(source, "// breakpoint 1")]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.continue_to_breakpoints(breakpoint_ids)
        
        output = self.collect_stdout(
            timeout_secs=1.0,
            pattern="abcdef"
        )
        self.assertTrue(output and len(output) > 0, "expect no program output")

        self.continue_to_exit()
        
        output += self.get_stdout(timeout=lldbdap_testcase.DAPTestCaseBase.timeoutval)
        self.assertTrue(output and len(output) > 0, "expect no program output")
        self.assertIn(
            "abcdefghi\r\nhello world\r\n",
            output,
            'full output not found in: ' + output
        )        
