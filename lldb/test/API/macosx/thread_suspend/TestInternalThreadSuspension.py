"""
Make sure that if threads are suspended outside of lldb, debugserver
won't make them run, even if we call an expression on the thread.
"""

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class TestSuspendedThreadHandling(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_suspended_threads(self):
        """Test that debugserver doesn't disturb the suspend count of a thread
           that has been suspended from within a program, when navigating breakpoints
           on other threads, or calling functions both on the suspended thread and
           on other threads."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.suspended_thread_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Set up your test case here. If your test doesn't need any set up then
        # remove this method from your TestCase class.

    def try_an_expression(self, thread, correct_value, test_bp):
        frame = thread.frames[0]

        value = frame.EvaluateExpression('function_to_call()')
        self.assertSuccess(value.GetError(), "Successfully called the function")
        self.assertEqual(value.GetValueAsSigned(), correct_value, "Got expected value for expression")

        # Again, make sure we didn't let the suspend thread breakpoint run:
        self.assertEqual(test_bp.GetHitCount(), 0, "First expression allowed the suspend thread to run")

        
    def make_bkpt(self, pattern):
        bp = self.target.BreakpointCreateBySourceRegex(pattern, self.main_source_file)
        self.assertEqual(bp.GetNumLocations(), 1, "Locations for %s"%(pattern))
        return bp
    
    def suspended_thread_test(self):
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Stop here to get things going", self.main_source_file)

        # Make in the running thread, so the we will have to stop a number of times
        # while handling breakpoints.  The first couple of times we hit it we will
        # run expressions as well.  Make sure we don't let the suspended thread run
        # during those operations.
        rt_bp = self.make_bkpt("Break here to show we can handle breakpoints")

        # Make a breakpoint that we will hit when the running thread exits:
        rt_exit_bp = self.make_bkpt("Break here after thread_join")

        # Make a breakpoint in the suspended thread.  We should not hit this till we
        # resume it after joining the running thread.
        st_bp = self.make_bkpt("We allowed the suspend thread to run")

        # Make a breakpoint after pthread_join of the suspend thread to ensure
        # that we didn't keep the thread from exiting normally
        st_exit_bp = self.make_bkpt(" Break here to make sure the thread exited normally")

        threads = lldbutil.continue_to_breakpoint(process, rt_bp)
        self.assertEqual(len(threads), 1, "Hit the running_func breakpoint")

        # Make sure we didn't hit the suspend thread breakpoint:
        self.assertEqual(st_bp.GetHitCount(), 0, "Continue allowed the suspend thread to run")

        # Now try an expression on the running thread:
        self.try_an_expression(threads[0], 0, st_bp)
        
        # Continue, and check the same things:
        threads = lldbutil.continue_to_breakpoint(process, rt_bp)
        self.assertEqual(len(threads), 1, "We didn't hit running breakpoint")

        # Try an expression on the suspended thread:
        thread = lldb.SBThread()
        for thread in process.threads:
            th_name = thread.GetName()
            if th_name == None:
                continue
            if "Look for me" in th_name:
                break
        self.assertTrue(thread.IsValid(), "We found the suspend thread.")
        self.try_an_expression(thread, 1, st_bp)
        
        # Now set the running thread breakpoint to auto-continue and let it
        # run a bit to make sure we still don't let the suspend thread run.
        rt_bp.SetAutoContinue(True)
        threads = lldbutil.continue_to_breakpoint(process, rt_exit_bp)
        self.assertEqual(len(threads), 1)
        self.assertEqual(st_bp.GetHitCount(), 0, "Continue again let suspended thread run")

        # Now continue and we SHOULD hit the suspend_func breakpoint:
        threads = lldbutil.continue_to_breakpoint(process, st_bp)
        self.assertEqual(len(threads), 1, "The thread resumed successfully")

        # Finally, continue again and we should get out of the last pthread_join
        # and the process should be about to exit
        threads = lldbutil.continue_to_breakpoint(process, st_exit_bp)
        self.assertEqual(len(threads), 1, "pthread_join exited successfully")
