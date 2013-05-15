"""
Test the iteration protocol for frame registers.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class RegistersIteratorTestCase(TestBase):

    mydir = os.path.join("python_api", "lldbutil", "iter")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line1 = line_number('main.cpp', '// Set break point at this line.')

    @expectedFailureLinux # llvm.org/pr14600 - Exception state registers not supported on Linux
    @python_api_test
    def test_iter_registers(self):
        """Test iterator works correctly for lldbutil.iter_registers()."""
        self.buildDefault()
        self.iter_registers()

    def iter_registers(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line1)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        import lldbutil
        for thread in process:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                for frame in thread:
                    # Dump the registers of this frame using lldbutil.get_GPRs() and friends.
                    if self.TraceOn():
                        print frame

                    REGs = lldbutil.get_GPRs(frame)
                    num = len(REGs)
                    if self.TraceOn():
                        print "\nNumber of general purpose registers: %d" % num
                    for reg in REGs:
                        self.assertTrue(reg)
                        if self.TraceOn():
                            print "%s => %s" % (reg.GetName(), reg.GetValue())

                    REGs = lldbutil.get_FPRs(frame)
                    num = len(REGs)
                    if self.TraceOn():
                        print "\nNumber of floating point registers: %d" % num
                    for reg in REGs:
                        self.assertTrue(reg)
                        if self.TraceOn():
                            print "%s => %s" % (reg.GetName(), reg.GetValue())

                    REGs = lldbutil.get_ESRs(frame)
                    num = len(REGs)
                    if self.TraceOn():
                        print "\nNumber of exception state registers: %d" % num
                    for reg in REGs:
                        self.assertTrue(reg)
                        if self.TraceOn():
                            print "%s => %s" % (reg.GetName(), reg.GetValue())

                    # And these should also work.
                    for kind in ["General Purpose Registers",
                                 "Floating Point Registers",
                                 "Exception State Registers"]:
                        REGs = lldbutil.get_registers(frame, kind)
                        self.assertTrue(REGs)

                    # We've finished dumping the registers for frame #0.
                    break


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
