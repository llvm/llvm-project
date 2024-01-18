"""
Test that formatter categories can work for multiple languages
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestMultilangFormatterCategories(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test_multilang_formatter_categories(self):
        """Test that formatter categories can work for multiple languages"""
        self.build()
        target, process, thread, a_breakpoint = \
            lldbutil.run_to_source_breakpoint(
                self, 'break here', lldb.SBFileSpec('main.swift'))
        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        dic = frame.FindVariable("dic")
        lldbutil.check_variable(
            self,
            dic,
            summary="2 key/value pairs",
            num_children=2)

        child0 = dic.GetChildAtIndex(0)
        lldbutil.check_variable(
            self,
            child0,
            num_children=2,
            typename="__lldb_autogen_nspair")

        id1 = child0.GetChildAtIndex(1)
        lldbutil.check_variable(self, id1, typename="__NSCFNumber *")

        id1child0 = dic.GetChildAtIndex(1).GetChildAtIndex(0)
        lldbutil.check_variable(
            self,
            id1child0,
            use_dynamic=True,
            typename="Swift.Optional<Foundation.NSURL>",
            summary='"http://www.google.com"')
