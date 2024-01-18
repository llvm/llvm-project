"""
Test lldb-dap disassemble request
"""


import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_disassemble(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_disassemble(self):
        """
        Tests the 'disassemble' request.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"
        self.source_path = os.path.join(os.getcwd(), source)
        self.set_source_breakpoints(
            source,
            [
                line_number(source, "// breakpoint 1"),
            ],
        )
        self.continue_to_next_stop()

        pc_assembly = self.disassemble(frameIndex=0)
        self.assertTrue("location" in pc_assembly, "Source location missing.")
        self.assertTrue("instruction" in pc_assembly, "Assembly instruction missing.")

        # The calling frame (qsort) is coming from a system library, as a result
        # we should not have a source location.
        qsort_assembly = self.disassemble(frameIndex=1)
        self.assertFalse("location" in qsort_assembly, "Source location not expected.")
        self.assertTrue("instruction" in pc_assembly, "Assembly instruction missing.")
