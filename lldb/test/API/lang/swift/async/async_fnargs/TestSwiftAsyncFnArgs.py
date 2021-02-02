import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAsyncFnArgs(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test(self):
        """Test function arguments in async functions"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', src)

        self.expect("frame var -d run-target -- msg", substrs=['"world"'])

        # Continue into the second coroutine funclet.
        bkpt2 = target.BreakpointCreateBySourceRegex("And also here", src, None)
        self.assertGreater(bkpt2.GetNumLocations(), 0)
        process.Continue()
        self.assertEqual(
             len(lldbutil.get_threads_stopped_at_breakpoint(process, bkpt2)), 1)

        self.expect("frame var -d run-target -- msg", substrs=['"world"'])
