"""
Test the iteration protocol for frame registers.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegistersIteratorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line1 = line_number(
            'main.cpp', '// Set break point at this line.')

    def test_iter_registers(self):
        """Test iterator works correctly for lldbutil.iter_registers()."""
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

        import lldbsuite.test.lldbutil as lldbutil
        for thread in process:
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                for frame in thread:
                    # Dump the registers of this frame using
                    # lldbutil.get_GPRs() and friends.
                    if self.TraceOn():
                        print(frame)

                    REGs = lldbutil.get_GPRs(frame)
                    num = len(REGs)
                    if self.TraceOn():
                        print(
                            "\nNumber of general purpose registers: %d" %
                            num)
                    for reg in REGs:
                        self.assertTrue(reg)
                        if self.TraceOn():
                            print("%s => %s" % (reg.GetName(), reg.GetValue()))

                    REGs = lldbutil.get_FPRs(frame)
                    num = len(REGs)
                    if self.TraceOn():
                        print("\nNumber of floating point registers: %d" % num)
                    for reg in REGs:
                        self.assertTrue(reg)
                        if self.TraceOn():
                            print("%s => %s" % (reg.GetName(), reg.GetValue()))

                    REGs = lldbutil.get_ESRs(frame)
                    if self.platformIsDarwin():
                        if self.getArchitecture() != 'armv7' and self.getArchitecture() != 'armv7k':
                            num = len(REGs)
                            if self.TraceOn():
                                print(
                                    "\nNumber of exception state registers: %d" %
                                    num)
                            for reg in REGs:
                                self.assertTrue(reg)
                                if self.TraceOn():
                                    print(
                                        "%s => %s" %
                                        (reg.GetName(), reg.GetValue()))
                    else:
                        self.assertIsNone(REGs)

                    # And these should also work.
                    for kind in ["General Purpose Registers",
                                 "Floating Point Registers"]:
                        REGs = lldbutil.get_registers(frame, kind)
                        self.assertTrue(REGs)

                    REGs = lldbutil.get_registers(
                        frame, "Exception State Registers")
                    if self.platformIsDarwin():
                        if self.getArchitecture() != 'armv7' and self.getArchitecture() != 'armv7k':
                            self.assertIsNotNone(REGs)
                    else:
                        self.assertIsNone(REGs)

                    # We've finished dumping the registers for frame #0.
                    break
