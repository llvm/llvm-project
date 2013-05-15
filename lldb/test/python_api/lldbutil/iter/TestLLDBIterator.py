"""
Test the iteration protocol for some lldb container objects.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class LLDBIteratorTestCase(TestBase):

    mydir = os.path.join("python_api", "lldbutil", "iter")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.line1 = line_number('main.cpp', '// Set break point at this line.')
        self.line2 = line_number('main.cpp', '// And that line.')

    @python_api_test
    def test_lldb_iter_module(self):
        """Test module_iter works correctly for SBTarget -> SBModule."""
        self.buildDefault()
        self.lldb_iter_module()

    @python_api_test
    def test_lldb_iter_breakpoint(self):
        """Test breakpoint_iter works correctly for SBTarget -> SBBreakpoint."""
        self.buildDefault()
        self.lldb_iter_breakpoint()

    @python_api_test
    def test_lldb_iter_frame(self):
        """Test iterator works correctly for SBProcess->SBThread->SBFrame."""
        self.buildDefault()
        self.lldb_iter_frame()

    def lldb_iter_module(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        from lldbutil import get_description
        yours = []
        for i in range(target.GetNumModules()):
            yours.append(target.GetModuleAtIndex(i))
        mine = []
        for m in target.module_iter():
            mine.append(m)

        self.assertTrue(len(yours) == len(mine))
        for i in range(len(yours)):
            if self.TraceOn():
                print "yours[%d]='%s'" % (i, get_description(yours[i]))
                print "mine[%d]='%s'" % (i, get_description(mine[i]))
            self.assertTrue(yours[i] == mine[i],
                            "UUID+FileSpec of yours[{0}] and mine[{0}] matches".format(i))

    def lldb_iter_breakpoint(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line2)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        self.assertTrue(target.GetNumBreakpoints() == 2)

        from lldbutil import get_description
        yours = []
        for i in range(target.GetNumBreakpoints()):
            yours.append(target.GetBreakpointAtIndex(i))
        mine = []
        for b in target.breakpoint_iter():
            mine.append(b)

        self.assertTrue(len(yours) == len(mine))
        for i in range(len(yours)):
            if self.TraceOn():
                print "yours[%d]='%s'" % (i, get_description(yours[i]))
                print "mine[%d]='%s'" % (i, get_description(mine[i]))
            self.assertTrue(yours[i] == mine[i],
                            "ID of yours[{0}] and mine[{0}] matches".format(i))

    def lldb_iter_frame(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        from lldbutil import print_stacktrace
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
                    print frame

        self.assertTrue(stopped_due_to_breakpoint)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
