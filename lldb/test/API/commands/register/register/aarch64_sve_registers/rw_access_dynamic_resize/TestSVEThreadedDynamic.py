"""
Test the AArch64 SVE registers dynamic resize with multiple threads.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):

    def check_sve_registers(self, vg_test_value):
        z_reg_size = vg_test_value * 8
        p_reg_size = int(z_reg_size / 8)

        p_value_bytes = ['0xff', '0x55', '0x11', '0x01', '0x00']

        for i in range(32):
            s_reg_value = 's%i = 0x' % (i) + \
                ''.join('{:02x}'.format(i + 1) for _ in range(4))

            d_reg_value = 'd%i = 0x' % (i) + \
                ''.join('{:02x}'.format(i + 1) for _ in range(8))

            v_reg_value = 'v%i = 0x' % (i) + \
                ''.join('{:02x}'.format(i + 1) for _ in range(16))

            z_reg_value = '{' + \
                ' '.join('0x{:02x}'.format(i + 1)
                         for _ in range(z_reg_size)) + '}'

            self.expect("register read -f hex " + 's%i' %
                        (i), substrs=[s_reg_value])

            self.expect("register read -f hex " + 'd%i' %
                        (i), substrs=[d_reg_value])

            self.expect("register read -f hex " + 'v%i' %
                        (i), substrs=[v_reg_value])

            self.expect("register read " + 'z%i' %
                        (i), substrs=[z_reg_value])

        for i in range(16):
            p_regs_value = '{' + \
                ' '.join(p_value_bytes[i % 5] for _ in range(p_reg_size)) + '}'
            self.expect("register read " + 'p%i' % (i), substrs=[p_regs_value])

        self.expect("register read ffr", substrs=[p_regs_value])

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(['linux']))
    def test_sve_registers_dynamic_config(self):
        """Test AArch64 SVE registers multi-threaded dynamic resize. """

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        if not self.isAArch64SVE():
            self.skipTest('SVE registers must be supported.')

        main_thread_stop_line = line_number(
            "main.c", "// Break in main thread")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", main_thread_stop_line)

        thX_break_line1 = line_number("main.c", "// Thread X breakpoint 1")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", thX_break_line1)

        thX_break_line2 = line_number("main.c", "// Thread X breakpoint 2")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", thX_break_line2)

        thY_break_line1 = line_number("main.c", "// Thread Y breakpoint 1")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", thY_break_line1)

        thY_break_line2 = line_number("main.c", "// Thread Y breakpoint 2")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", thY_break_line2)

        self.runCmd("run", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()

        thread1 = process.GetThreadAtIndex(0)

        self.expect("thread info 1", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint"])

        self.check_sve_registers(8)

        self.runCmd("process continue", RUN_SUCCEEDED)

        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
                self.runCmd("thread continue %d" % (idx + 1))
                self.assertEqual(thread.GetStopReason(),
                                 lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(
                0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(4)
                self.runCmd('register write vg 2')

            elif stopped_at_line_number == thY_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(2)
                self.runCmd('register write vg 4')

        self.runCmd("thread continue 2")
        self.runCmd("thread continue 3")

        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            self.assertEqual(thread.GetStopReason(),
                             lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(
                0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(2)

            elif stopped_at_line_number == thY_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(4)
