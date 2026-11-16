""" Testing debugging of a binary with "mixed" dwarf (with/without fission). """
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMixedDwarfBinary(TestBase):
    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    @add_test_categories(["dwo"])
    @requireLinux
    def test_mixed_dwarf(self):
        """Test that 'frame variable' works
        for the executable built from two source files compiled
        with/whithout -gsplit-dwarf correspondingly."""

        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "g", bkpt_module="a.out"
        )

        frame = thread.GetFrameAtIndex(0)
        x = frame.FindVariable("x")
        self.assertTrue(x.IsValid(), "x is not valid")
        y = frame.FindVariable("y")
        self.assertTrue(y.IsValid(), "y is not valid")
