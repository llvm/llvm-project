from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftLateDylib(TestBase):
    @skipUnlessDarwin
    @swiftTest
    @skipIfDarwinEmbedded 
    def test(self):
        """Test that a late loaded Swift dylib is debuggable"""
        self.build()
        target, process, _, _ = lldbutil.run_to_name_breakpoint(self, "main")
        # Initialize SwiftASTContext before loading the dylib.
        self.expect("expr -l Swift -- 0")
        bkpt = target.BreakpointCreateByLocation(lldb.SBFileSpec('dylib.swift'), 7)
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("v x", substrs=['Hello from the Dylib'])
        self.expect("expr x", substrs=['Hello from the Dylib'])
