"""
Test that step over will let other threads run when necessary
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StepOverDoesntDeadlockTestCase(TestBase):

    def test_step_over(self):
        """Test that when step over steps over a function it lets other threads run."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "without running the first thread at least somewhat",
                                                                            lldb.SBFileSpec("locking.cpp"))
        # This is just testing that the step over actually completes.
        # If the test fails this step never return, so failure is really
        # signaled by the test timing out.
        
        thread.StepOver()
        state = process.GetState()
        self.assertState(state, lldb.eStateStopped)
