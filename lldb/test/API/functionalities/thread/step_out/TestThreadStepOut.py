"""
Test stepping out from a function in a multi-threaded program.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadStepOutTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18066 inferior does not exit")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureNetBSD
    def test_step_single_thread(self):
        """Test thread step out on one thread via command interpreter. """
        self.build()
        self.step_out_test(self.step_out_single_thread_with_cmd)

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr19347 2nd thread stops at breakpoint")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"])
    @expectedFailureAll(oslist=["watchos"], archs=['armv7k'], bugnumber="rdar://problem/34674488") # stop reason is trace when it should be step-out
    @expectedFailureNetBSD
    def test_step_all_threads(self):
        """Test thread step out on all threads via command interpreter. """
        self.build()
        self.step_out_test(self.step_out_all_threads_with_cmd)

    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr19347 2nd thread stops at breakpoint")
    @skipIfWindows # This test will hang on windows llvm.org/pr21753
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24681")
    @expectedFailureNetBSD
    def test_python(self):
        """Test thread step out on one thread via Python API (dwarf)."""
        self.build()
        self.step_out_test(self.step_out_with_python)
        
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.bkpt_string = '// Set breakpoint here'
        self.breakpoint = line_number('main.cpp', self.bkpt_string)       
        self.step_in_line = line_number('main.cpp', '// But we might still be here')
        self.step_out_dest = line_number('main.cpp', '// Expect to stop here after step-out.')

    def check_stepping_thread(self):
        zeroth_frame = self.step_out_thread.frames[0]
        line_entry = zeroth_frame.line_entry
        self.assertTrue(line_entry.IsValid(), "Stopped at a valid line entry")
        self.assertEqual("main.cpp", line_entry.file.basename, "Still in main.cpp")
        # We can't really tell whether we stay on our line
        # or get to the next line, it depends on whether there are any
        # instructions between the call and the return.
        line = line_entry.line
        self.assertTrue(line == self.step_out_dest or line == self.step_in_line, "Stepped to the wrong line: {0}".format(line))
        
    def step_out_single_thread_with_cmd(self):
        other_threads = {}
        for thread in self.process.threads:
            if thread.GetIndexID() == self.step_out_thread.GetIndexID():
                continue
            other_threads[thread.GetIndexID()] = thread.frames[0].line_entry

        # There should be other threads...
        self.assertNotEqual(len(other_threads), 0)
        self.step_out_with_cmd("this-thread")
        # The other threads should not have made progress:
        for thread in self.process.threads:
            index_id = thread.GetIndexID()
            line_entry = other_threads.get(index_id)
            if line_entry:
                self.assertEqual(thread.frames[0].line_entry.file.basename, line_entry.file.basename, "Thread {0} moved by file".format(index_id))
                self.assertEqual(thread.frames[0].line_entry.line, line_entry.line, "Thread {0} moved by line".format(index_id))

    def step_out_all_threads_with_cmd(self):
        self.step_out_with_cmd("all-threads")
                                        
    def step_out_with_cmd(self, run_mode):
        self.runCmd("thread select %d" % self.step_out_thread.GetIndexID())
        self.runCmd("thread step-out -m %s" % run_mode)
        self.expect("process status", "Expected stop reason to be step-out",
                    substrs=["stop reason = step out"])

        selected_thread = self.process.GetSelectedThread()
        self.assertEqual(selected_thread.GetIndexID(), self.step_out_thread.GetIndexID(), "Step out changed selected thread.")
        self.check_stepping_thread()
                                        
    def step_out_with_python(self):
        self.step_out_thread.StepOut()

        reason = self.step_out_thread.GetStopReason()
        self.assertEqual(
            lldb.eStopReasonPlanComplete,
            reason,
            "Expected thread stop reason 'plancomplete', but got '%s'" %
            lldbutil.stop_reason_to_str(reason))
        self.check_stepping_thread()
                            

    def step_out_test(self, step_out_func):
        """Test single thread step out of a function."""
        (self.inferior_target, self.process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, self.bkpt_string, lldb.SBFileSpec('main.cpp'), only_one_thread = False)

        # We hit the breakpoint on at least one thread.  If we hit it on both threads
        # simultaneously, we can try the step out.  Otherwise, suspend the thread
        # that hit the breakpoint, and continue till the second thread hits
        # the breakpoint:

        (breakpoint_threads, other_threads) = ([], [])
        lldbutil.sort_stopped_threads(self.process,
                                      breakpoint_threads=breakpoint_threads,
                                      other_threads=other_threads)
        if len(breakpoint_threads) == 1:
            success = thread.Suspend()
            self.assertTrue(success, "Couldn't suspend a thread")
            breakpoint_threads = lldbutil.continue_to_breakpoint(self.process,
                                                           bkpt)
            self.assertEqual(len(breakpoint_threads), 2, "Second thread stopped")
            success = thread.Resume()
            self.assertTrue(success, "Couldn't resume a thread")

        self.step_out_thread = breakpoint_threads[0]

        # Step out of thread stopped at breakpoint
        step_out_func()
