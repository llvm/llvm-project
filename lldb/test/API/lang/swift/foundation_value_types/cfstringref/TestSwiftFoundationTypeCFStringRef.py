import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftFoundationTypeCFStringRef(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    @skipUnlessFoundation
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame variable short", substrs=["Hello"])
        self.expect("frame variable long", substrs=["longer"])
