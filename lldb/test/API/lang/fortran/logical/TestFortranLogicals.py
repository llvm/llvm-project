"""
Tests that the logical intrinsic type with different byte sizes works as expected 
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FortranTestLogicals(TestBase):

    def test_fortran_logicals(self):
        """Tests if logicals return the correct name, kind and value."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("logical.f90")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "! Breakpoint here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        bool_one = frame.FindVariable("bool_one")
        self.assertSuccess(bool_one.GetError(), "Failed to fetch bool_one")
        self.assertEqual(bool_one.GetTypeName(), "LOGICAL(KIND=1)")
        self.assertEqual(bool_one.GetByteSize(), 1)
        self.assertEqual(bool_one.GetValue(), "true")

        bool_two = frame.FindVariable("bool_two")
        self.assertSuccess(bool_two.GetError(), "Failed to fetch bool_two")
        self.assertEqual(bool_two.GetTypeName(), "LOGICAL(KIND=2)")
        self.assertEqual(bool_two.GetByteSize(), 2)
        self.assertEqual(bool_two.GetValue(), "false")
        
        bool_four = frame.FindVariable("bool_four")
        self.assertSuccess(bool_four.GetError(), "Failed to fetch bool_four")
        self.assertEqual(bool_four.GetTypeName(), "LOGICAL")
        self.assertEqual(bool_four.GetByteSize(), 4)
        self.assertEqual(bool_four.GetValue(), "true")

        bool_eight = frame.FindVariable("bool_eight")
        self.assertSuccess(bool_eight.GetError(), "Failed to fetch bool_eight")
        self.assertEqual(bool_eight.GetTypeName(), "LOGICAL(KIND=8)")
        self.assertEqual(bool_eight.GetByteSize(), 8)
        self.assertEqual(bool_eight.GetValue(), "false")