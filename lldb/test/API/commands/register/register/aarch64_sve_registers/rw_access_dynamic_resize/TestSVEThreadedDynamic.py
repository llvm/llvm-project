"""
Test the AArch64 SVE and Streaming SVE (SSVE) registers dynamic resize with
multiple threads.

This test assumes a minimum supported vector length (VL) of 256 bits
and will test 512 bits if possible. We refer to "vg" which is the
register shown in lldb. This is in units of 64 bits. 256 bit VL is
the same as a vg of 4.
"""

from enum import Enum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Mode(Enum):
    SVE = 0
    SSVE = 1


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

        self.runCmd("breakpoint delete 1")
        self.runCmd("continue")

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

    def build_for_mode(self, mode):
        cflags = "-march=armv8-a+sve -lpthread"
        if mode == Mode.SSVE:
            cflags += " -DUSE_SSVE"
        self.build(dictionary={"CFLAGS_EXTRAS": cflags})

    def run_sve_test(self, mode):
        if (mode == Mode.SVE) and not self.isAArch64SVE():
            self.skipTest("SVE registers must be supported.")

        if (mode == Mode.SSVE) and not self.isAArch64SME():
            self.skipTest("Streaming SVE registers must be supported.")

        self.build_for_mode(mode)

        supported_vg = self.get_supported_vg()

        if not (2 in supported_vg and 4 in supported_vg):
            self.skipTest("Not all required SVE vector lengths are supported.")

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

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_sve_registers_dynamic_config(self):
        """Test AArch64 SVE registers multi-threaded dynamic resize."""
        self.run_sve_test(Mode.SVE)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_ssve_registers_dynamic_config(self):
        """Test AArch64 SSVE registers multi-threaded dynamic resize."""
        self.run_sve_test(Mode.SSVE)

    def setup_svg_test(self, mode):
        # Even when running in SVE mode, we need access to SVG for these tests.
        if not self.isAArch64SME():
            self.skipTest("Streaming SVE registers must be present.")

        self.build_for_mode(mode)

        supported_vg = self.get_supported_vg()

        main_thread_stop_line = line_number("main.c", "// Break in main thread")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", main_thread_stop_line)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        return process, supported_vg

    def read_reg(self, process, regset, reg):
        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sve_registers = registerSets.GetFirstValueByName(regset)
        return sve_registers.GetChildMemberWithName(reg).GetValueAsUnsigned()

    def read_vg(self, process):
        return self.read_reg(process, "Scalable Vector Extension Registers", "vg")

    def read_svg(self, process):
        return self.read_reg(process, "Scalable Matrix Extension Registers", "svg")

    def do_svg_test(self, process, vgs, expected_svgs):
        for vg, svg in zip(vgs, expected_svgs):
            self.runCmd("register write vg {}".format(vg))
            self.assertEqual(svg, self.read_svg(process))

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_svg_sve_mode(self):
        """When in SVE mode, svg should remain constant as we change vg."""
        process, supported_vg = self.setup_svg_test(Mode.SVE)
        svg = self.read_svg(process)
        self.do_svg_test(process, supported_vg, [svg] * len(supported_vg))

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_svg_ssve_mode(self):
        """When in SSVE mode, changing vg should change svg to the same value."""
        process, supported_vg = self.setup_svg_test(Mode.SSVE)
        self.do_svg_test(process, supported_vg, supported_vg)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_sme_not_present(self):
        """When there is no SME, we should not show the SME register sets."""
        if self.isAArch64SME():
            self.skipTest("Streaming SVE registers must not be present.")

        self.build_for_mode(Mode.SVE)

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This test may run on a non-sve system, but we'll stop before any
        # SVE instruction would be run.
        self.runCmd("b main")
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        registerSets = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        sme_registers = registerSets.GetFirstValueByName(
            "Scalable Matrix Extension Registers"
        )
        self.assertFalse(sme_registers.IsValid())
