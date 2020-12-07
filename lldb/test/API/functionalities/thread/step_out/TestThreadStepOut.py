"""
Test stepping out from a function in a multi-threaded program.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadStepOutTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Test occasionally times out on the Linux build bot
    @skipIfLinux
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr23477 Test occasionally times out on the Linux build bot")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18066 inferior does not exit")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureNetBSD
    def test_step_single_thread(self):
        """Test thread step out on one thread via command interpreter. """
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_single_thread_with_cmd)

    # Test occasionally times out on the Linux build bot
    @skipIfLinux
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr23477 Test occasionally times out on the Linux build bot")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr19347 2nd thread stops at breakpoint")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureAll(oslist=["watchos"], archs=['armv7k'], bugnumber="rdar://problem/34674488") # stop reason is trace when it should be step-out
    @expectedFailureNetBSD
    def test_step_all_threads(self):
        """Test thread step out on all threads via command interpreter. """
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_all_threads_with_cmd)

    # Test occasionally times out on the Linux build bot
    @skipIfLinux
    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr23477 Test occasionally times out on the Linux build bot")
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr19347 2nd thread stops at breakpoint")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24681")
    @expectedFailureNetBSD
    def test_python(self):
        """Test thread step out on one thread via Python API (dwarf)."""
        self.build(dictionary=self.getBuildFlags())
        self.step_out_test(self.step_out_with_python)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.bkpt_string = '// Set breakpoint here'
        self.breakpoint = line_number('main.cpp', self.bkpt_string)       

        if "gcc" in self.getCompiler() or self.isIntelCompiler():
            self.step_out_destination = line_number(
                'main.cpp', '// Expect to stop here after step-out (icc and gcc)')
        else:
            self.step_out_destination = line_number(
                'main.cpp', '// Expect to stop here after step-out (clang)')

    def step_out_single_thread_with_cmd(self):
        self.step_out_with_cmd("this-thread")
        self.expect(
            "thread backtrace all",
            "Thread location after step out is correct",
            substrs=[
                "main.cpp:%d" %
                self.step_out_destination,
                "main.cpp:%d" %
                self.breakpoint])

    def step_out_all_threads_with_cmd(self):
        self.step_out_with_cmd("all-threads")
        self.expect(
            "thread backtrace all",
            "Thread location after step out is correct",
            substrs=[
                "main.cpp:%d" %
                self.step_out_destination])

    def step_out_with_cmd(self, run_mode):
        self.runCmd("thread select %d" % self.step_out_thread.GetIndexID())
        self.runCmd("thread step-out -m %s" % run_mode)
        self.expect("process status", "Expected stop reason to be step-out",
                    substrs=["stop reason = step out"])

        self.expect(
            "thread list",
            "Selected thread did not change during step-out",
            substrs=[
                "* thread #%d" %
                self.step_out_thread.GetIndexID()])

    def step_out_with_python(self):
        self.step_out_thread.StepOut()

        reason = self.step_out_thread.GetStopReason()
        self.assertEqual(
            lldb.eStopReasonPlanComplete,
            reason,
            "Expected thread stop reason 'plancomplete', but got '%s'" %
            lldbutil.stop_reason_to_str(reason))

        # Verify location after stepping out
        frame = self.step_out_thread.GetFrameAtIndex(0)
        desc = lldbutil.get_description(frame.GetLineEntry())
        expect = "main.cpp:%d" % self.step_out_destination
        self.assertTrue(
            expect in desc, "Expected %s but thread stopped at %s" %
            (expect, desc))

    def step_out_test(self, step_out_func):
        """Test single thread step out of a function."""
        (self.inferior_target, self.inferior_process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, self.bkpt_string, lldb.SBFileSpec('main.cpp'), only_one_thread = False)

        # We hit the breakpoint on at least one thread.  If we hit it on both threads
        # simultaneously, we can try the step out.  Otherwise, suspend the thread
        # that hit the breakpoint, and continue till the second thread hits
        # the breakpoint:

        (breakpoint_threads, other_threads) = ([], [])
        lldbutil.sort_stopped_threads(self.inferior_process,
                                      breakpoint_threads=breakpoint_threads,
                                      other_threads=other_threads)
        if len(breakpoint_threads) == 1:
            success = thread.Suspend()
            self.assertTrue(success, "Couldn't suspend a thread")
            bkpt_threads = lldbutil.continue_to_breakpoint(bkpt)
            self.assertEqual(len(bkpt_threads), 1, "Second thread stopped")
            success = thread.Resume()
            self.assertTrue(success, "Couldn't resume a thread")

        self.step_out_thread = breakpoint_threads[0]

        # Step out of thread stopped at breakpoint
        step_out_func()
