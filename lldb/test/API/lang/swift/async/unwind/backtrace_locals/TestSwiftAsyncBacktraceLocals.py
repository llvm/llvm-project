import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAsyncBacktraceLocals(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test async unwind"""
        self.build()
        src = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'main breakpoint', src)

        start_bkpt = target.BreakpointCreateBySourceRegex('function start', src, None)
        end_bkpt = target.BreakpointCreateBySourceRegex('end iteration', src, None)
        
        if self.TraceOn():
           self.runCmd("bt all")
           
        for n in range(10):
            lldbutil.continue_to_breakpoint(process, start_bkpt)
            for f in range(n+1):
               frame = thread.GetFrameAtIndex(f)
               self.assertIn("fibonacci", frame.GetFunctionName(), 
                             "Redundantly confirm that we're stopped in fibonacci()")
               if f == 0:
                   # Get arguments (arguments, locals, statics, in_scope_only)
                   args = frame.GetVariables(True, False, False, True)
                   self.assertEqual(len(args), 1, "Found one argument")
                   self.assertEqual(args[0].GetName(), "n", "Found n argument")
                   self.assertEqual(args[0].GetValue(), str(10-n), "n has correct value")
            self.assertIn("Main.main", thread.GetFrameAtIndex(n+1).GetFunctionName())

        lldbutil.continue_to_breakpoint(process, end_bkpt)
        frame = thread.GetFrameAtIndex(0)
        args = frame.GetVariables(True, False, False, True)
        self.assertEqual(len(args), 1, "Found one argument")
        self.assertEqual(args[0].GetName(), "n", "Found n argument")
        self.assertEqual(args[0].GetValue(), str(1), "n has correct value")

        lldbutil.continue_to_breakpoint(process, end_bkpt)
        frame = thread.GetFrameAtIndex(0)
        args = frame.GetVariables(True, False, False, True)
        self.assertEqual(len(args), 1, "Found one argument")
        self.assertEqual(args[0].GetName(), "n", "Found n argument")
        self.assertEqual(args[0].GetValue(), str(0), "n has correct value")
