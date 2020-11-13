"""
Test that lldb can continue past a __builtin_debugtrap, but not a __builtin_trap
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class BuiltinDebugTrapTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    # Currently this depends on behavior in debugserver to
    # advance the pc past __builtin_trap instructions so that
    # continue works.  Everyone is in agreement that this
    # should be moved up into lldb instead of depending on the
    # remote stub rewriting the pc values.
    @skipUnlessDarwin

    def test(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set a breakpoint here", lldb.SBFileSpec("main.cpp"))

        # Continue to __builtin_debugtrap()
        process.Continue()
        if self.TraceOn():
            self.runCmd("f")
            self.runCmd("bt")
            self.runCmd("ta v global")

        self.assertEqual(process.GetSelectedThread().GetStopReason(), 
                         lldb.eStopReasonException)

        list = target.FindGlobalVariables("global", 1, lldb.eMatchTypeNormal)
        self.assertEqual(list.GetSize(), 1)
        global_value = list.GetValueAtIndex(0)

        self.assertEqual(global_value.GetValueAsUnsigned(), 5)

        # Continue to the __builtin_trap() -- we should be able to 
        # continue past __builtin_debugtrap.
        process.Continue()
        if self.TraceOn():
            self.runCmd("f")
            self.runCmd("bt")
            self.runCmd("ta v global")

        self.assertEqual(process.GetSelectedThread().GetStopReason(), 
                         lldb.eStopReasonException)

        # "global" is now 10.
        self.assertEqual(global_value.GetValueAsUnsigned(), 10)

        # We should be at the same point as before -- cannot advance
        # past a __builtin_trap().
        process.Continue()
        if self.TraceOn():
            self.runCmd("f")
            self.runCmd("bt")
            self.runCmd("ta v global")

        self.assertEqual(process.GetSelectedThread().GetStopReason(), 
                         lldb.eStopReasonException)

        # "global" is still 10.
        self.assertEqual(global_value.GetValueAsUnsigned(), 10)
