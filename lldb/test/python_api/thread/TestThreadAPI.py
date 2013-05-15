"""
Test SBThread APIs.
"""

import os, time
import unittest2
import lldb
from lldbutil import get_stopped_thread, get_caller_symbol
from lldbtest import *

class ThreadAPITestCase(TestBase):

    mydir = os.path.join("python_api", "thread")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_get_process_with_dsym(self):
        """Test Python SBThread.GetProcess() API."""
        self.buildDsym()
        self.get_process()

    @python_api_test
    @dwarf_test
    def test_get_process_with_dwarf(self):
        """Test Python SBThread.GetProcess() API."""
        self.buildDwarf()
        self.get_process()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_get_stop_description_with_dsym(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDsym()
        self.get_stop_description()

    @python_api_test
    @dwarf_test
    def test_get_stop_description_with_dwarf(self):
        """Test Python SBThread.GetStopDescription() API."""
        self.buildDwarf()
        self.get_stop_description()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_run_to_address_with_dsym(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.run_to_address(self.exe_name)

    @python_api_test
    @dwarf_test
    def test_run_to_address_with_dwarf(self):
        """Test Python SBThread.RunToAddress() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.run_to_address(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_step_out_of_malloc_into_function_b_with_dsym(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        # We build a different executable than the default buildDsym() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_out_of_malloc_into_function_b(self.exe_name)

    @expectedFailureLinux # llvm.org/pr14416
    @python_api_test
    @dwarf_test
    def test_step_out_of_malloc_into_function_b_with_dwarf(self):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_out_of_malloc_into_function_b(self.exe_name)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_step_over_3_times_with_dsym(self):
        """Test Python SBThread.StepOver() API."""
        # We build a different executable than the default buildDsym() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_over_3_times(self.exe_name)

    @python_api_test
    @dwarf_test
    def test_step_over_3_times_with_dwarf(self):
        """Test Python SBThread.StepOver() API."""
        # We build a different executable than the default buildDwarf() does.
        d = {'CXX_SOURCES': 'main2.cpp', 'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.step_over_3_times(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number within main.cpp to break inside main().
        self.break_line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")
        # Find the line numbers within main2.cpp for step_out_of_malloc_into_function_b() and step_over_3_times().
        self.step_out_of_malloc = line_number("main2.cpp", "// thread step-out of malloc into function b.")
        self.after_3_step_overs = line_number("main2.cpp", "// we should reach here after 3 step-over's.")

        # We'll use the test method name as the exe_name for executable comppiled from main2.cpp.
        self.exe_name = self.testMethodName

    def get_process(self):
        """Test Python SBThread.GetProcess() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.break_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        self.runCmd("process status")

        proc_of_thread = thread.GetProcess()
        #print "proc_of_thread:", proc_of_thread
        self.assertTrue(proc_of_thread.GetProcessID() == process.GetProcessID())

    def get_stop_description(self):
        """Test Python SBThread.GetStopDescription() API."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.break_line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        #self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
        #self.runCmd("process status")

        # Due to the typemap magic (see lldb.swig), we pass in an (int)length to GetStopDescription
        # and expect to get a Python string as the return object!
        # The 100 is just an arbitrary number specifying the buffer size.
        stop_description = thread.GetStopDescription(100)
        self.expect(stop_description, exe=False,
            startstr = 'breakpoint')

    def step_out_of_malloc_into_function_b(self, exe_name):
        """Test Python SBThread.StepOut() API to step out of a malloc call where the call site is at function b()."""
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByName('malloc')
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        while True:
            thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
            self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint")
            caller_symbol = get_caller_symbol(thread)
            #print "caller symbol of malloc:", caller_symbol
            if not caller_symbol:
                self.fail("Test failed: could not locate the caller symbol of malloc")
            if caller_symbol == "b(int)":
                break
            #self.runCmd("thread backtrace")
            #self.runCmd("process status")           
            process.Continue()

        thread.StepOut()
        self.runCmd("thread backtrace")
        #self.runCmd("process status")           
        self.assertTrue(thread.GetFrameAtIndex(0).GetLineEntry().GetLine() == self.step_out_of_malloc,
                        "step out of malloc into function b is successful")

    def step_over_3_times(self, exe_name):
        """Test Python SBThread.StepOver() API."""
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation('main2.cpp', self.step_out_of_malloc)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.step_out_of_malloc.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.step_out_of_malloc)

        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        self.runCmd("thread backtrace")

        # Verify that we are stopped at the correct source line number in main2.cpp.
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        # Expected failure with clang as the compiler.
        # rdar://problem/9223880
        #
        # Which has been fixed on the lldb by compensating for inaccurate line
        # table information with r140416.
        self.assertTrue(lineEntry.GetLine() == self.after_3_step_overs)

    def run_to_address(self, exe_name):
        """Test Python SBThread.RunToAddress() API."""
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation('main2.cpp', self.step_out_of_malloc)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.step_out_of_malloc.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "There should be a thread stopped due to breakpoint condition")
        self.runCmd("thread backtrace")
        frame0 = thread.GetFrameAtIndex(0)
        lineEntry = frame0.GetLineEntry()
        self.assertTrue(lineEntry.GetLine() == self.step_out_of_malloc)

        # Get the start/end addresses for this line entry.
        start_addr = lineEntry.GetStartAddress().GetLoadAddress(target)
        end_addr = lineEntry.GetEndAddress().GetLoadAddress(target)
        if self.TraceOn():
            print "start addr:", hex(start_addr)
            print "end addr:", hex(end_addr)

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


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
