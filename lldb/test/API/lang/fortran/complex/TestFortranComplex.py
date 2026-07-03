"""
Tests that the complex intrinsic type with different byte sizes works as expected 
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FortranTestComplex(TestBase):

    def test_fortran_complex(self):
        """Tests if complex return the correct name, kind and value."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("complex.f90")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "! Breakpoint here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        complex_four = frame.FindVariable("complex_four")
        self.assertSuccess(complex_four.GetError(), "Failed to fetch complex_four.")
        self.assertEqual(complex_four.GetTypeName(), "COMPLEX")
        self.assertEqual(complex_four.GetByteSize(), 8)
        self.assertEqual(complex_four.GetValue(), "(2, 3)")

        complex_eight = frame.FindVariable("complex_eight")
        self.assertSuccess(complex_eight.GetError(), "Failed to fetch complex_eight.")
        self.assertEqual(complex_eight.GetTypeName(), "COMPLEX(KIND=8)")
        self.assertEqual(complex_eight.GetByteSize(), 16)
        self.assertEqual(complex_eight.GetValue(), "(1, 4)")