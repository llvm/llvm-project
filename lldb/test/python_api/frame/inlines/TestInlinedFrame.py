"""
Testlldb Python SBFrame APIs IsInlined() and GetFunctionName().
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class InlinedFrameAPITestCase(TestBase):

    mydir = os.path.join("python_api", "frame", "inlines")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_stop_at_outer_inline_with_dsym(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.buildDsym()
        self.do_stop_at_outer_inline()

    @python_api_test
    @dwarf_test
    def test_stop_at_outer_inline_with_dwarf(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        self.buildDwarf()
        self.do_stop_at_outer_inline()

    def setUp(self):
        
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.source = 'inlines.c'
        self.first_stop = line_number(self.source, '// This should correspond to the first break stop.')
        self.second_stop = line_number(self.source, '// This should correspond to the second break stop.')

    def do_stop_at_outer_inline(self):
        """Exercise SBFrame.IsInlined() and SBFrame.GetFunctionName()."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by the name of 'inner_inline'.
        breakpoint = target.BreakpointCreateByName('inner_inline', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() > 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        import lldbutil
        stack_traces1 = lldbutil.print_stacktraces(process, string_buffer=True)
        if self.TraceOn():
            print "Full stack traces when first stopped on the breakpoint 'inner_inline':"
            print stack_traces1

        # The first breakpoint should correspond to an inlined call frame.
        # If it's an inlined call frame, expect to find, in the stack trace,
        # that there is a frame which corresponds to the following call site:
        #
        #     outer_inline (argc);
        #
        frame0 = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        if frame0.IsInlined():
            filename = frame0.GetLineEntry().GetFileSpec().GetFilename()
            self.assertTrue(filename == self.source)
            self.expect(stack_traces1, "First stop at %s:%d" % (self.source, self.first_stop), exe=False,
                        substrs = ['%s:%d' % (self.source, self.first_stop)])

            # Expect to break again for the second time.
            process.Continue()
            self.assertTrue(process.GetState() == lldb.eStateStopped,
                            PROCESS_STOPPED)
            stack_traces2 = lldbutil.print_stacktraces(process, string_buffer=True)
            if self.TraceOn():
                print "Full stack traces when stopped on the breakpoint 'inner_inline' for the second time:"
                print stack_traces2
                self.expect(stack_traces2, "Second stop at %s:%d" % (self.source, self.second_stop), exe=False,
                            substrs = ['%s:%d' % (self.source, self.second_stop)])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
