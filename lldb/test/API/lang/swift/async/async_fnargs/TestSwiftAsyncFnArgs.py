import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAsyncFnArgs(lldbtest.TestBase):

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test function arguments in async functions"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', src)

        while process.selected_thread.stop_reason == lldb.eStopReasonBreakpoint:
            self.expect("frame var -d run-target msg", patterns=['"(basic|generic|static|closure) world"'])
            process.Continue()
