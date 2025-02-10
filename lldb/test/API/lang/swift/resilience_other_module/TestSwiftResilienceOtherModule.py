import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftResilienceOtherModule(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test_with_debug_info(self):
        self.impl('break here with debug info')

    @skipUnlessDarwin
    @swiftTest
    def test_without_debug_info(self):
        self.impl('break here without debug info')

    def impl(self, break_str):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, break_str, lldb.SBFileSpec('main.swift'))

        self.expect("expression s.a", substrs=["Int", "100"])
        self.expect("expression s.b", substrs=["Int", "200"])
