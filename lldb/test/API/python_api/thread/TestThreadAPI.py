"""
Test SBThread APIs.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_thread, get_caller_symbol


class ThreadAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_get_process(self):
        """Test Python SBThread.GetProcess() API."""
        self.build()
        self.get_process()

    def test_get_stop_description(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.build()
        self.get_stop_description()

    def test_run_to_address(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default build() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.run_to_address(self.exe_name)

    @skipIfAsan # The output looks different under ASAN.
    @expectedFailureAll(oslist=["linux"], archs=['arm'], bugnumber="llvm.org/pr45892")
    @expectedFailureAll(oslist=["windows"])
    def test_step_out_of_malloc_into_function_b(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        # We build a different executable than the default build() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_out_of_malloc_into_function_b(self.exe_name)

    def test_step_over_3_times(self):
        """Test Python SBThread.StepOver() API."""
        # We build a different executable than the default build() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_over_3_times(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number within main.cpp to break inside main().
        self.break_line = line_number(
            "main.cpp", "// Set break point at this line and check variable 'my_char'.")
        # Find the line numbers within main2.cpp for
        # step_out_of_malloc_into_function_b() and step_over_3_times().
        self.step_out_of_malloc = line_number(
            "main2.cpp", "// thread step-out of malloc into function b.")
        self.after_3_step_overs = line_number(
            "main2.cpp", "// we should reach here after 3 step-over's.")

        # We'll use the test method name as the exe_name for executable
        # compiled from main2.cpp.
        self.exe_name = self.testMethodName

    def get_process(self):
        """Test Python SBThread.GetProcess() API."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(
            "main.cpp", self.break_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")
        self.runCmd("process status")

        proc_of_thread = thread.GetProcess()
        self.trace("proc_of_thread:", proc_of_thread)
        self.assertEqual(proc_of_thread.GetProcessID(), process.GetProcessID())

    def get_stop_description(self):
        """Test Python SBThread.GetStopDescription() API."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(
            "main.cpp", self.break_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        #self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        # Get the stop reason. GetStopDescription expects that we pass in the size of the description
        # we expect plus an additional byte for the null terminator.

        # Test with a buffer that is exactly as large as the expected stop reason.
        self.assertEqual("breakpoint 1.1", thread.GetStopDescription(len('breakpoint 1.1') + 1))

        # Test some smaller buffer sizes.
        self.assertEqual("breakpoint", thread.GetStopDescription(len('breakpoint') + 1))
        self.assertEqual("break", thread.GetStopDescription(len('break') + 1))
        self.assertEqual("b", thread.GetStopDescription(len('b') + 1))

        # Test that we can pass in a much larger size and still get the right output.
        self.assertEqual("breakpoint 1.1", thread.GetStopDescription(len('breakpoint 1.1') + 100))

    def step_out_of_malloc_into_function_b(self, exe_name):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByName('malloc')
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        while True:
            thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
            self.assertTrue(
                thread.IsValid(),
                "There should be a thread stopped due to breakpoint")
            caller_symbol = get_caller_symbol(thread)
            if not caller_symbol:
                self.fail(
                    "Test failed: could not locate the caller symbol of malloc")

            # Our top frame may be an inlined function in malloc() (e.g., on
            # FreeBSD).  Apply a simple heuristic of stepping out until we find
            # a non-malloc caller
            while caller_symbol.startswith("malloc"):
                thread.StepOut()
                self.assertTrue(thread.IsValid(),
                                "Thread valid after stepping to outer malloc")
                caller_symbol = get_caller_symbol(thread)

            if caller_symbol == "b(int)":
                break
            process.Continue()

        # On Linux malloc calls itself in some case. Remove the breakpoint because we don't want
        # to hit it during step-out.
        target.BreakpointDelete(breakpoint.GetID())

        thread.StepOut()
        self.runCmd("thread backtrace")
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.step_out_of_malloc,
            "step out of malloc into function b is successful")

    def step_over_3_times(self, exe_name):
        """Test Python SBThread.StepOver() API."""
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(
            'main2.cpp', self.step_out_of_malloc)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.step_out_of_malloc.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertEqual(lineEntry.GetLine(), self.step_out_of_malloc)

        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        self.runCmd("thread backtrace")

        # Verify that we are stopped at the correct source line number in
        # main2.cpp.
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)
        # Expected failure with clang as the compiler.
        # rdar://problem/9223880
        #
        # Which has been fixed on the lldb by compensating for inaccurate line
        # table information with r140416.
        self.assertEqual(lineEntry.GetLine(), self.after_3_step_overs)

    def run_to_address(self, exe_name):
        """Test Python SBThread.RunToAddress() API."""
        exe = self.getBuildArtifact(exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(
            'main2.cpp', self.step_out_of_malloc)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.step_out_of_malloc.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertEqual(lineEntry.GetLine(), self.step_out_of_malloc)

        # Get the start/end addresses for this line entry.
        start_addr = lineEntry.GetStartAddress().GetLoadAddress(target)
        end_addr = lineEntry.GetEndAddress().GetLoadAddress(target)
        if self.TraceOn():
            print("start addr:", hex(start_addr))
            print("end addr:", hex(end_addr))

        # Disable the breakpoint.
        self.assertTrue(target.DisableAllBreakpoints())
        self.runCmd("breakpoint list")

        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        self.runCmd("thread backtrace")

        # Now ask SBThread to run to the address 'start_addr' we got earlier, which
        # corresponds to self.step_out_of_malloc line entry's start address.
        thread.RunToAddress(start_addr)
        self.runCmd("process status")
        #self.runCmd("thread backtrace")
