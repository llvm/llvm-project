"""
Test that lldb can continue past a __builtin_debugtrap, but not a __builtin_trap
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class BuiltinDebugTrapTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        platform_stop_reason = lldb.eStopReasonSignal
        platform = self.getPlatform()
        if platform == "darwin" or platform == "windows":
            platform_stop_reason = lldb.eStopReasonException

        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set a breakpoint here", lldb.SBFileSpec("main.c")
        )

        # Continue to __builtin_debugtrap()
        process.Continue()
        if self.TraceOn():
            self.runCmd("f")
            self.runCmd("bt")
            self.runCmd("ta v global")

        self.assertEqual(
            process.GetSelectedThread().GetStopReason(), platform_stop_reason
        )

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

        self.assertEqual(
            process.GetSelectedThread().GetStopReason(), platform_stop_reason
        )

        # "global" is now 10.
        self.assertEqual(global_value.GetValueAsUnsigned(), 10)

        # Change the handling of SIGILL on x86-64 Linux - do not pass it
        # to the inferior, but stop and notify lldb.
        if self.getArchitecture() == "x86_64" and platform == "linux":
            self.runCmd("process handle -p false SIGILL")

        # We should be at the same point as before -- cannot advance
        # past a __builtin_trap().
        process.Continue()
        if self.TraceOn():
            self.runCmd("f")
            self.runCmd("bt")
            self.runCmd("ta v global")

        self.assertEqual(
            process.GetSelectedThread().GetStopReason(), platform_stop_reason
        )

        # "global" is still 10.
        self.assertEqual(global_value.GetValueAsUnsigned(), 10)
