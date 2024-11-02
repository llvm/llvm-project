"""
Test more expression command sequences with objective-c.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FoundationTestCaseString(TestBase):

    def test_NSString_expr_commands(self):
        """Test expression commands for NSString."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
                self, '// Break here for NSString tests',
                lldb.SBFileSpec('main.m', False))

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression (int)[str length]",
                    patterns=["\(int\) \$.* ="])
        self.expect("expression (int)[str_id length]",
                    patterns=["\(int\) \$.* ="])
        self.expect("expression (id)[str description]",
                    patterns=["\(id\) \$.* = 0x"])
        self.expect("expression (id)[str_id description]",
                    patterns=["\(id\) \$.* = 0x"])
        self.expect("expression str.length")
        self.expect('expression str = @"new"')
        self.runCmd("image lookup -t NSString")
        self.expect('expression str = (id)[NSString stringWithCString: "new"]')
        self.runCmd("process continue")

    @expectedFailureAll(archs=["i[3-6]86"], bugnumber="<rdar://problem/28814052>")
    def test_MyString_dump_with_runtime(self):
        """Test dump of a known Objective-C object by dereferencing it."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
                self, '// Set break point at this line',
                lldb.SBFileSpec('main.m', False))
        self.expect(
            "expression --show-types -- *my",
            patterns=[
                "\(MyString\) \$.* = ",
                "\(MyBase\)"])
        self.runCmd("process continue")
