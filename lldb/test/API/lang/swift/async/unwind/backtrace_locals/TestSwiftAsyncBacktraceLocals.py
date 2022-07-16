import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAsyncBacktraceLocals(lldbtest.TestBase):

    def setUp(self):
        # Call super's setUp().
        lldbtest.TestBase.setUp(self)
        self.src = lldb.SBFileSpec('main.swift')
        self.fibo_naumbers = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    @swiftTest
    @skipIfWindows
    @skipIfLinux
    @skipIf(archs=no_match(["arm64", "arm64e", "arm64_32", "x86_64"]))
    def test(self):
        """Test async unwind"""
        self.build()
        target, process, thread, main_bkpt = lldbutil.run_to_source_breakpoint(
            self, 'main breakpoint', self.src)
        self.run_fibo_tests(target, process)

    @swiftTest
    @skipIfWindows
    @skipIfLinux
    @skipIf(archs=no_match(["arm64", "arm64e", "arm64_32", "x86_64"]))
    def test_actor(self):
        """Test async unwind"""
        self.build()
        target, process, thread, main_bkpt = lldbutil.run_to_source_breakpoint(
            self, 'main actor breakpoint', self.src)
        self.run_fibo_tests(target, process)

        
    def run_fibo_tests(self, target, process):
        self.start_bkpt = target.BreakpointCreateBySourceRegex('function start', self.src, None)
        self.end_bkpt = target.BreakpointCreateBySourceRegex('end recursion', self.src, None)
        self.recurse_bkpt = target.BreakpointCreateBySourceRegex('recurse', self.src, None)
        self.compute_bkpt = target.BreakpointCreateBySourceRegex('compute result', self.src, None)

        # cfa[n] contains the CFA for the fibonacci(n) call.
        cfa = [ None for _ in range(11) ]

        # Continue 10 times, hitting all the initial calls to the fibonacci()
        # function with decreasing arguments.
        for n in range(10):
            threads = lldbutil.continue_to_breakpoint(process, self.start_bkpt)
            self.assertEqual(len(threads), 1, "Found 1 thread stopped at breakpoint")
            thread = threads[0]
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
                n_var = frame.FindVariable("n")
                self.assertTrue(n_var, "Found 'n'")
                self.assertEqual(n_var.GetValueAsSigned(), fibonacci_number,
                                 "n has correct value (n=%d, f=%d)"%(n, f))
                if f != 0:
                    # The PC of a logical frame is stored in its "callee"
                    # AsyncContext as the second pointer field.
                    error = lldb.SBError()
                    ret_addr = process.ReadPointerFromMemory(
                        cfa[fibonacci_number-1] + target.addr_size, error)
                    self.assertSuccess(error, "Managed to read context memory")
                    self.assertEqual(ret_addr, frame.GetPC())

            self.assertIn("Main.main", thread.GetFrameAtIndex(n+1).GetFunctionName())

        # After having stopped at all the entry points, we stop at the end recursion
        # breakpoint.
        thread = lldbutil.continue_to_breakpoint(process, self.end_bkpt)
        frame = thread[0].GetFrameAtIndex(0)
        args = frame.GetVariables(True, False, False, True)
        n_var = frame.FindVariable("n")
        self.assertEqual(n_var.GetValueAsSigned(), 1, "n has correct value")
        # Callers have n values from 2 to 10
        for n in range(1,9):
            frame = thread[0].GetFrameAtIndex(n)
            n_var = frame.FindVariable("n")
            self.assertEqual(n_var.GetValueAsSigned(), n+1, "n has correct value")

        # Once we got our first result from the end recursion above, we continue
        # executing to the second recursion
        thread = lldbutil.continue_to_breakpoint(process, self.recurse_bkpt)
        self.assertNotEqual(len(thread), 0, "Hit the correct breakpoint")
        frame = thread[0].GetFrameAtIndex(0)
        n_var = frame.FindVariable("n")
        self.assertEqual(n_var.GetValueAsSigned(), 2, "n has correct value")
        # Callers have n values from 3 to 10
        for n in range(1,8):
            frame = thread[0].GetFrameAtIndex(n)
            n_var = frame.FindVariable("n")
            self.assertEqual(n_var.GetValueAsSigned(), n+2, "n has correct value")

        # The second recursion leads us to a new entry in the function with n == 0
        thread = lldbutil.continue_to_breakpoint(process, self.start_bkpt)
        self.assertNotEqual(len(thread), 0, "Hit the correct breakpoint")
        frame = thread[0].GetFrameAtIndex(0)
        n_var = frame.FindVariable("n")
        self.assertEqual(n_var.GetValueAsSigned(), 0, "n has correct value")
        # Callers have n values from 2 to 10
        for n in range(1,9):
            frame = thread[0].GetFrameAtIndex(n)
            n_var = frame.FindVariable("n")
            self.assertEqual(n_var.GetValueAsSigned(), n+1, "n has correct value")

        # Let's disable all intermediate breakpoints and verify that we can access
        # the locals in the last part of the function.
        self.start_bkpt.SetEnabled(False)
        self.end_bkpt.SetEnabled(False)
        self.recurse_bkpt.SetEnabled(False)

        last_result = None
        while True:
            thread = lldbutil.continue_to_breakpoint(process, self.compute_bkpt)
            if len(thread) == 0:
                self.assertEqual(last_result, 55, "Computed the right final value")
                break

            frame = thread[0].GetFrameAtIndex(0)
            n_var = frame.FindVariable("n")
            self.assertTrue(n_var, "Found n")
            n_1_var = frame.FindVariable("n_1")
            self.assertTrue(n_1_var, "Found n_1")
            n_2_var = frame.FindVariable("n_2")
            self.assertTrue(n_2_var, "Found n_2")

            n_val = n_var.GetValueAsSigned()
            self.assertEqual(n_1_var.GetValueAsSigned(), self.fibo_naumbers[n_val - 1],
                             "n_1 has correct value")
            self.assertEqual(n_2_var.GetValueAsSigned(), self.fibo_naumbers[n_val - 2],
                             "n_2 has correct value")
            last_result = self.fibo_naumbers[n_val]

