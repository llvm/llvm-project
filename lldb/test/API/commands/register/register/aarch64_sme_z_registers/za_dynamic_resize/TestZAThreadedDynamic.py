"""
Test the AArch64 SME Array Storage (ZA) register dynamic resize with
multiple threads.
"""

from enum import Enum
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64ZAThreadedTestCase(TestBase):
    def get_supported_vg(self):
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        main_thread_stop_line = line_number("main.c", "// Break in main thread")
        lldbutil.run_break_set_by_file_and_line(self, "main.c", main_thread_stop_line)

        self.runCmd("settings set target.run-args 0")
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

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

    def gen_za_value(self, svg, value_generator):
        svl = svg * 8

        rows = []
        for row in range(svl):
            byte = "0x{:02x}".format(value_generator(row))
            rows.append(" ".join([byte] * svl))

        return "{" + " ".join(rows) + "}"

    def check_za_register(self, svg, value_offset):
        self.expect(
            "register read za",
            substrs=[self.gen_za_value(svg, lambda r: r + value_offset)],
        )

    def check_disabled_za_register(self, svg):
        self.expect("register read za", substrs=[self.gen_za_value(svg, lambda r: 0)])

    def za_test_impl(self, enable_za):
        # Although the test program doesn't obviously do any operations that
        # would need smefa64, calls to libc functions like memset may do.
        if not self.isAArch64SMEFA64():
            self.skipTest("SME and the sm3fa64 extension must be present")

        self.build()
        supported_vg = self.get_supported_vg()

        self.runCmd("settings set target.run-args {}".format("1" if enable_za else "0"))

        if not (2 in supported_vg and 4 in supported_vg):
            self.skipTest("Not all required streaming vector lengths are supported.")

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

        self.expect(
            "thread info 1",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint"],
        )

        if 8 in supported_vg:
            if enable_za:
                self.check_za_register(8, 1)
            else:
                self.check_disabled_za_register(8)
        else:
            if enable_za:
                self.check_za_register(4, 1)
            else:
                self.check_disabled_za_register(4)

        self.runCmd("process continue", RUN_SUCCEEDED)

        process = self.dbg.GetSelectedTarget().GetProcess()
        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
                self.runCmd("thread continue %d" % (idx + 1))
                self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_za_register(4, 2)
                self.runCmd("register write vg 2")
                self.check_disabled_za_register(2)

            elif stopped_at_line_number == thY_break_line1:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_za_register(2, 3)
                self.runCmd("register write vg 4")
                self.check_disabled_za_register(4)

        self.runCmd("thread continue 2")
        self.runCmd("thread continue 3")

        for idx in range(1, process.GetNumThreads()):
            thread = process.GetThreadAtIndex(idx)
            self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

            stopped_at_line_number = thread.GetFrameAtIndex(0).GetLineEntry().GetLine()

            if stopped_at_line_number == thX_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_za_register(2, 2)

            elif stopped_at_line_number == thY_break_line2:
                self.runCmd("thread select %d" % (idx + 1))
                self.check_za_register(4, 3)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_register_dynamic_config_main_enabled(self):
        """Test multiple threads resizing ZA, with the main thread's ZA
        enabled."""
        self.za_test_impl(True)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_za_register_dynamic_config_main_disabled(self):
        """Test multiple threads resizing ZA, with the main thread's ZA
        disabled."""
        self.za_test_impl(False)
