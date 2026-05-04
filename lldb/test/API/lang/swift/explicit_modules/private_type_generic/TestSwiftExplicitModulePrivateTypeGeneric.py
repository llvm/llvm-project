import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModulePrivateTypeGeneric(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIfWindows
    def test(self):
        """Test frame variable of a generic struct specialized to a
        private type from another explicitly-built module."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=["Dylib"])
        self.expect("v s", substrs=["t = (value = 42)"])
        self.expect("expr -d run -- s", substrs=["t = (value = 42)"])
        self.expect("expression 1+1", substrs=["(Int)", "= 2"])
