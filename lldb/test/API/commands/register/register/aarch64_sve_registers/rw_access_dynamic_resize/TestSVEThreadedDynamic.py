"""
Test the AArch64 SVE registers dynamic resize with multiple threads.

This test assumes a minimum supported vector length (VL) of 256 bits
and will test 512 bits if possible. We refer to "vg" which is the
register shown in lldb. This is in units of 64 bits. 256 bit VL is
the same as a vg of 4.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):
    def get_supported_vg(self):
        # Changing VL trashes the register state, so we need to run the program
        # just to test this. Then run it again for the test.
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        main_thread_stop_line = line_number("main.c", "// Break in main thread")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", main_thread_stop_line)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        # Write back the current vg to confirm read/write works at all.
        current_vg = self.match("register read vg", ["(0x[0-9]+)"])
        self.assertTrue(current_vg is not None)
        self.expect("register write vg {}".format(current_vg.group()))

        # Aka 128, 256 and 512 bit.
        supported_vg = []
        for vg in [2, 4, 8]:
            # This could mask other errors but writing vg is tested elsewhere
            # so we assume the hardware rejected the value.
            self.runCmd("register write vg {}".format(vg), check=False)
            if not self.res.GetError():
                supported_vg.append(vg)

        return supported_vg

    def check_sve_registers(self, vg_test_value):
        z_reg_size = vg_test_value * 8
        p_reg_size = int(z_reg_size / 8)

        p_value_bytes = ["0xff", "0x55", "0x11", "0x01", "0x00"]

        for i in range(32):
            s_reg_value = "s%i = 0x" % (i) + "".join(
                "{:02x}".format(i + 1) for _ in range(4)
            )

            d_reg_value = "d%i = 0x" % (i) + "".join(
                "{:02x}".format(i + 1) for _ in range(8)
            )

            v_reg_value = "v%i = 0x" % (i) + "".join(
                "{:02x}".format(i + 1) for _ in range(16)
            )

            z_reg_value = (
                "{"
                + " ".join("0x{:02x}".format(i + 1) for _ in range(z_reg_size))
                + "}"
            )

            self.expect("register read -f hex " + "s%i" % (i), substrs=[s_reg_value])

            self.expect("register read -f hex " + "d%i" % (i), substrs=[d_reg_value])

            self.expect("register read -f hex " + "v%i" % (i), substrs=[v_reg_value])

            self.expect("register read " + "z%i" % (i), substrs=[z_reg_value])

        for i in range(16):
            p_regs_value = (
                "{" + " ".join(p_value_bytes[i % 5] for _ in range(p_reg_size)) + "}"
            )
            self.expect("register read " + "p%i" % (i), substrs=[p_regs_value])

        self.expect("register read ffr", substrs=[p_regs_value])

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_sve_registers_dynamic_config(self):
        """Test AArch64 SVE registers multi-threaded dynamic resize."""

        if not self.isAArch64SVE():
            self.skipTest("SVE registers must be supported.")

        self.build()
        supported_vg = self.get_supported_vg()

        if not (2 in supported_vg and 4 in supported_vg):
            self.skipTest("Not all required SVE vector lengths are supported.")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        main_thread_stop_line = line_number("main.c", "// Break in main thread")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", main_thread_stop_line)

        thX_break_line1 = line_number("main.c", "// Thread X breakpoint 1")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", thX_break_line1)

        thX_break_line2 = line_number("main.c", "// Thread X breakpoint 2")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", thX_break_line2)

        thY_break_line1 = line_number("main.c", "// Thread Y breakpoint 1")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", thY_break_line1)

        thY_break_line2 = line_number("main.c", "// Thread Y breakpoint 2")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", thY_break_line2)

        self.runCmd("run", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        if 8 in supported_vg:
            self.check_sve_registers(8)
        else:
            self.check_sve_registers(4)

        self.runCmd("process continue", RUN_SUCCEEDED)

        # If we start the checks too quickly, thread 3 may not have started.
        while (process.GetNumThreads() < 3):
            pass

        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
                self.runCmd("thread continue %d" % (idx + 1))
                self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(4)
                self.runCmd("register write vg 2")

            elif stopped_at_line_number == thY_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(2)
                self.runCmd("register write vg 4")

        self.runCmd("thread continue 2")
        self.runCmd("thread continue 3")

        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(2)

            elif stopped_at_line_number == thY_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_sve_registers(4)
