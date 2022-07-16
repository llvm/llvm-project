"""
Test that contiguous array prints correctly
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestContiguousArray(lldbtest.TestBase):

    @swiftTest
    def test_frame_contiguous_array(self):
        """Test that contiguous array prints correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect("frame variable",
                    startstr="""(ContiguousArray<a.Class>) array = 1 value {
  [0] = 0x""")
