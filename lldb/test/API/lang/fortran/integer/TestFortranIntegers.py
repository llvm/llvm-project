"""
Tests that the integer intrinsic type with different byte sizes works as expected 
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class FortranTestIntegers(TestBase):

    def test_fortran_integers(self):
        """Tests if integers return the correct name, kind and value."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("integers.f90")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "! Breakpoint here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        tiny_int = frame.FindVariable("tiny_int")
        self.assertSuccess(tiny_int.GetError(), "Failed to fetch tiny_int")
        self.assertEqual(tiny_int.GetTypeName(), "INTEGER(KIND=1)")
        self.assertEqual(tiny_int.GetByteSize(), 1)
        self.assertEqual(tiny_int.GetValueAsSigned(), 127)

        short_int = frame.FindVariable("short_int")
        self.assertSuccess(short_int.GetError(), "Failed to fetch short_int")
        self.assertEqual(short_int.GetTypeName(), "INTEGER(KIND=2)")
        self.assertEqual(short_int.GetByteSize(), 2)
        self.assertEqual(short_int.GetValueAsSigned(), 32767)
        
        normal_int = frame.FindVariable("normal_int")
        self.assertSuccess(normal_int.GetError(), "Failed to fetch normal_int")
        self.assertEqual(normal_int.GetTypeName(), "INTEGER")
        self.assertEqual(normal_int.GetByteSize(), 4)
        self.assertEqual(normal_int.GetValueAsSigned(), 2147483647)

        huge_int = frame.FindVariable("huge_int")
        self.assertSuccess(huge_int.GetError(), "Failed to fetch huge_int")
        self.assertEqual(huge_int.GetTypeName(), "INTEGER(KIND=8)")
        self.assertEqual(huge_int.GetByteSize(), 8)
        self.assertEqual(huge_int.GetValueAsSigned(), 9223372036854775807)