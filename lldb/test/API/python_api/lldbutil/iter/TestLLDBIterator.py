"""
Test the iteration protocol for some lldb container objects.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LLDBIteratorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.line1 = line_number(
            'main.cpp', '// Set break point at this line.')
        self.line2 = line_number('main.cpp', '// And that line.')

    @add_test_categories(['pyapi'])
    def test_lldb_iter_module(self):
        """Test module_iter works correctly for SBTarget -> SBModule."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        from lldbsuite.test.lldbutil import get_description
        yours = []
        for i in range(target.GetNumModules()):
            yours.append(target.GetModuleAtIndex(i))
        mine = []
        for m in target.module_iter():
            mine.append(m)

        self.assertTrue(len(yours) == len(mine))
        for i in range(len(yours)):
            if self.TraceOn():
                print("yours[%d]='%s'" % (i, get_description(yours[i])))
                print("mine[%d]='%s'" % (i, get_description(mine[i])))
            self.assertTrue(
                yours[i] == mine[i],
                "UUID+FileSpec of yours[{0}] and mine[{0}] matches".format(i))

    @add_test_categories(['pyapi'])
    def test_lldb_iter_breakpoint(self):
        """Test breakpoint_iter works correctly for SBTarget -> SBBreakpoint."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line2)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        self.assertTrue(target.GetNumBreakpoints() == 2)

        from lldbsuite.test.lldbutil import get_description
        yours = []
        for i in range(target.GetNumBreakpoints()):
            yours.append(target.GetBreakpointAtIndex(i))
        mine = []
        for b in target.breakpoint_iter():
            mine.append(b)

        self.assertTrue(len(yours) == len(mine))
        for i in range(len(yours)):
            if self.TraceOn():
                print("yours[%d]='%s'" % (i, get_description(yours[i])))
                print("mine[%d]='%s'" % (i, get_description(mine[i])))
            self.assertTrue(yours[i] == mine[i],
                            "ID of yours[{0}] and mine[{0}] matches".format(i))

    @add_test_categories(['pyapi'])
    def test_lldb_iter_frame(self):
        """Test iterator works correctly for SBProcess->SBThread->SBFrame."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        from lldbsuite.test.lldbutil import print_stacktrace
        stopped_due_to_breakpoint = False
        for thread in process:
            if self.TraceOn():
                print_stacktrace(thread)
            ID = thread.GetThreadID()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_due_to_breakpoint = True
            for frame in thread:
                self.assertTrue(frame.GetThread().GetThreadID() == ID)
                if self.TraceOn():
                    print(frame)

        self.assertTrue(stopped_due_to_breakpoint)
