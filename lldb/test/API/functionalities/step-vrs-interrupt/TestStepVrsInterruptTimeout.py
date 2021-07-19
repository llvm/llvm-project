"""
This is to make sure that the interrupt timer
doesn't influence synchronous user level stepping.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestStepVrsInterruptTimeout(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_step_vrs_interrupt(self):
        """This test is to make sure that the interrupt timeout
           doesn't cause use to flub events from a synchronous step."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.sample_test()

    def sample_test(self):
        """You might use the test implementation in several ways, say so here."""

        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)
        self.dbg.SetAsync(False)
        self.runCmd("settings set target.process.interrupt-timeout 1")
        thread.StepOver()
        self.assertEqual(process.GetState(), lldb.eStateStopped, "Stopped like we should")
