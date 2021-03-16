import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAsyncBacktraceLocals(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIf(oslist=['windows', 'linux', 'macosx'])
    @skipIf(archs=no_match(["arm64", "arm64e", "arm64_32", "x86_64"]))
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

        # cfa[n] contains the CFA for the fibonacci(n) call.
        cfa = [ None for _ in range(11) ]

        # Continue 10 times, hitting all the initial calls to the fibonacci()
        # function with decreasing arguments.
        for n in range(10):
            lldbutil.continue_to_breakpoint(process, start_bkpt)
            # The top frame is fibonacci(10-n)
            for f in range(n+1):
                frame = thread.GetFrameAtIndex(f)
                # The selected frame is fibonacci(10-n+f)
                fibonacci_number = 10-n+f
                self.assertIn("fibonacci", frame.GetFunctionName(),
                              "Redundantly confirm that we're stopped in fibonacci()")
                if not cfa[fibonacci_number]:
                    cfa[fibonacci_number] = frame.GetCFA()
                else:
                    self.assertEqual(cfa[fibonacci_number], frame.GetCFA(),
                                     "Stable CFA for the first 10 recursions")
                # Get arguments (arguments, locals, statics, in_scope_only)
                args = frame.GetVariables(True, False, False, True)
                self.assertEqual(len(args), 1, "Found one argument")
                self.assertEqual(args[0].GetName(), "n", "Found n argument")
                self.assertEqual(args[0].GetValue(), str(fibonacci_number), "n has correct value")
                if f != 0:
                    # The PC of a logical frame is stored in its "callee"
                    # AsyncContext as the second pointer field.
                    error = lldb.SBError()
                    ret_addr = process.ReadPointerFromMemory(cfa[fibonacci_number-1] + target.addr_size, error)
                    self.assertTrue(error.Success(), "Managed to read context memory")
                    self.assertEqual(ret_addr, frame.GetPC())

            self.assertIn("Main.main", thread.GetFrameAtIndex(n+1).GetFunctionName())

        lldbutil.continue_to_breakpoint(process, end_bkpt)
        frame = thread.GetFrameAtIndex(0)
        args = frame.GetVariables(True, False, False, True)
        self.assertEqual(len(args), 1, "Found one argument")
        self.assertEqual(args[0].GetName(), "n", "Found n argument")
        self.assertEqual(args[0].GetValue(), str(1), "n has correct value")
        # Callers have n values from 2 to 10
        for n in range(1,9):
            frame = thread.GetFrameAtIndex(n)
            args = frame.GetVariables(True, False, False, True)
            self.assertEqual(len(args), 1, "Found one argument")
            self.assertEqual(args[0].GetName(), "n", "Found n argument")
            self.assertEqual(args[0].GetValue(), str(n+1), "n has correct value")

        lldbutil.continue_to_breakpoint(process, end_bkpt)
        frame = thread.GetFrameAtIndex(0)
        args = frame.GetVariables(True, False, False, True)
        self.assertEqual(len(args), 1, "Found one argument")
        self.assertEqual(args[0].GetName(), "n", "Found n argument")
        self.assertEqual(args[0].GetValue(), str(0), "n has correct value")
        # Callers have n values from 2 to 10
        for n in range(1,9):
            frame = thread.GetFrameAtIndex(n)
            args = frame.GetVariables(True, False, False, True)
            self.assertEqual(len(args), 1, "Found one argument")
            self.assertEqual(args[0].GetName(), "n", "Found n argument")
            self.assertEqual(args[0].GetValue(), str(n+1), "n has correct value")
