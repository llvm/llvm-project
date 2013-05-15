"""Test calling methods on super."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestObjCSuperMethod(TestBase):

    mydir = os.path.join("lang", "objc", "objc-super")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailurei386
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test calling methods on super."""
        self.buildDsym()
        self.objc_super()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @expectedFailurei386
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test calling methods on super."""
        self.buildDwarf()
        self.objc_super()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "class.m"
        self.break_line = line_number(self.main_source, '// Set breakpoint here.')

    def objc_super(self):
        """Test calling methods on super."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bpt = target.BreakpointCreateByLocation(self.main_source, self.break_line)
        self.assertTrue(bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue (len(thread_list) != 0, "No thread stopped at our breakpoint.")
        self.assertTrue (len(thread_list) == 1, "More than one thread stopped at our breakpoint.")
            
        # Now make sure we can call a function in the class method we've stopped in.
        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue (frame, "Got a valid frame 0 frame.")

        cmd_value = frame.EvaluateExpression ("[self get]")
        self.assertTrue (cmd_value.IsValid())
        self.assertTrue (cmd_value.GetValueAsUnsigned() == 2)

        cmd_value = frame.EvaluateExpression ("[super get]")
        self.assertTrue (cmd_value.IsValid())
        self.assertTrue (cmd_value.GetValueAsUnsigned() == 1)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
