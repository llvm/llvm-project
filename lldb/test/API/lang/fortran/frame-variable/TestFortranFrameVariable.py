"""
Tests that the frame variable command works 
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FortranTestFrameVariable(TestBase):

    def test_fortran_frame_variable(self):
        """Tests if frame variable outputs the expected results"""
        self.build()
        self.main_source_file = lldb.SBFileSpec("frame.f90")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "! Breakpoint here", self.main_source_file
        )

        self.expect("frame variable num_int", substrs=["(INTEGER) num_int = 152"])

        self.expect("frame variable num_real", substrs=["(REAL) num_real = 2.718"])

        self.expect("frame variable num_logical", substrs=["(LOGICAL) num_logical = true"])

        self.expect("frame variable num_complex", substrs=["(COMPLEX) num_complex = (1.3, 2.6)"])