"""
Test stop reasons after hitting and deleting a breakpoint and
stepping another thread. Scenario:
  - run a thread
  - stop the thread at a breakpoint
  - delete the breakpoint
  - single step on the main thread
The thread stopped at the deleted breakpoint should have stop reason
'none'.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadBreakStepOtherTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_hit_breakpoint_delete_step_other_thread(self):
        main_source_file = lldb.SBFileSpec("main.cpp")
        self.build()
        (target, process, main_thread, _) = lldbutil.run_to_source_breakpoint(
            self, "// main break here", main_source_file, only_one_thread = False)

        # Run until the breakpoint in the thread.
        thread_breakpoint = target.BreakpointCreateBySourceRegex(
            "// thread break here", main_source_file)
        self.assertGreater(
            thread_breakpoint.GetNumLocations(),
            0,
            "thread breakpoint has no locations associated with it.")
        process.Continue()
        stopped_threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, thread_breakpoint)
        self.assertEquals(
            1,
            len(stopped_threads),
            "only one thread expected stopped at the thread breakpoint")
        breakpoint_thread = stopped_threads[0]

        # Delete the breakpint in the thread and do a step in the main thread.
        target.BreakpointDelete(thread_breakpoint.GetID())
        main_thread.StepInstruction(False)

        # Check the stop reasons.
        reason = main_thread.GetStopReason()
        self.assertEqual(
            lldb.eStopReasonPlanComplete,
            reason,
            "Expected thread stop reason 'plancomplete', but got '%s'" %
            lldbutil.stop_reason_to_str(reason))

        reason = breakpoint_thread.GetStopReason()
        self.assertEqual(
            lldb.eStopReasonNone,
            reason,
            "Expected thread stop reason 'none', but got '%s'" %
            lldbutil.stop_reason_to_str(reason))
