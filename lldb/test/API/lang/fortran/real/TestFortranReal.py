"""
Tests that the real intrinsic type with different byte sizes works as expected 
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FortranTestReal(TestBase):

    def test_fortran_real(self):
        """Tests if Real type return the correct name, kind and value."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("real.f90")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "! Breakpoint here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)
        
        float_four = frame.FindVariable("float_four")
        self.assertSuccess(float_four.GetError(), "Failed to fetch float_four")
        self.assertEqual(float_four.GetTypeName(), "REAL")
        self.assertEqual(float_four.GetByteSize(), 4)
        self.assertTrue(float_four.GetValue().startswith("3.14"), "float_four value is correct")

        float_eight = frame.FindVariable("float_eight")
        self.assertSuccess(float_eight.GetError(), "Failed to fetch float_eight")
        self.assertEqual(float_eight.GetTypeName(), "REAL(KIND=8)")
        self.assertEqual(float_eight.GetByteSize(), 8)
        self.assertTrue(float_eight.GetValue().startswith("2.718"), "float_eight value is correct")