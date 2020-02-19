"""
Test more expression command sequences with objective-c.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationTestCaseNSArray(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_NSArray_expr_commands(self):
        """Test expression commands for NSArray."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
                self, '// Break here for NSArray tests',
                lldb.SBFileSpec('main.m', False))

        self.runCmd("thread backtrace")
        self.expect("expression (int)[nil_mutable_array count]",
                    patterns=["\(int\) \$.* = 0"])
        self.expect("expression (int)[array1 count]",
                    patterns=["\(int\) \$.* = 3"])
        self.expect("expression (int)[array2 count]",
                    patterns=["\(int\) \$.* = 3"])
        self.expect("expression (int)array1.count",
                    patterns=["\(int\) \$.* = 3"])
        self.expect("expression (int)array2.count",
                    patterns=["\(int\) \$.* = 3"])
        self.runCmd("process continue")
