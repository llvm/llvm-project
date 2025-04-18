import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftObjCBaseClassMemberLookup(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test accessing a static member from a member function"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift')
        )

        # dwim-print start by checking whether a member 'A' exists in 'self'.
        self.expect("dwim-print A.shared", substrs=["42"])
