import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftAsyncVariables(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @skipEmbeddedSwift  # rdar://183960945 (Fix async tests running in embedded mode)
    @swiftTest
    @skipIf(oslist=['windows'])
    def test(self):
        """Test local variables in async functions"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', src)

        while process.selected_thread.stop_reason == lldb.eStopReasonBreakpoint:
            self.expect("frame variable x", substrs=["23"])
            process.Continue()
