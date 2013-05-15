"""
Test SBValue.GetObjectDescription() with the value from SBTarget.FindGlobalVariables().
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ObjectDescriptionAPITestCase(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")

    # rdar://problem/10857337
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_find_global_variables_then_object_description_with_dsym(self):
        """Exercise SBTaget.FindGlobalVariables() API."""
        d = {'EXE': 'a.out'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_global_variables_then_object_description('a.out')

    # rdar://problem/10857337
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_find_global_variables_then_object_description_with_dwarf(self):
        """Exercise SBTarget.FindGlobalVariables() API."""
        d = {'EXE': 'b.out'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.find_global_variables_then_object_description('b.out')

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.source = 'main.m'
        self.line = line_number(self.source, '// Set break point at this line.')

    def find_global_variables_then_object_description(self, exe_name):
        """Exercise SBTaget.FindGlobalVariables() followed by SBValue.GetObjectDescription()."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)
        # Make sure we hit our breakpoint:
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)

        thread = thread_list[0]
        frame0 = thread.GetFrameAtIndex(0)

        # Note my_global_str's object description prints fine here.
        value_list1 = frame0.GetVariables(True, True, True, True)
        for v in value_list1:
            self.DebugSBValue(v)
            if self.TraceOn():
                print "val:", v
                print "object description:", v.GetObjectDescription()
            if v.GetName() == 'my_global_str':
                self.assertTrue(v.GetObjectDescription() == 'This is a global string')

        # But not here!
        value_list2 = target.FindGlobalVariables('my_global_str', 3)
        for v in value_list2:
            self.DebugSBValue(v)
            if self.TraceOn():
                print "val:", v
                print "object description:", v.GetObjectDescription()
            if v.GetName() == 'my_global_str':
                self.assertTrue(v.GetObjectDescription() == 'This is a global string')

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
