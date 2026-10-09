import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThunkTest(TestBase):
    def test_step_through_thunk(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "testit")

        # Make sure we step through the thunk into Derived1::doit
        self.expect(
            "step",
            STEP_IN_SUCCEEDED,
            substrs=["stop reason = step in", "Derived1::doit"],
        )

        self.runCmd("continue")

        self.expect(
            "step",
            STEP_IN_SUCCEEDED,
            substrs=["stop reason = step in", "Derived2::doit"],
        )

    @skipIfWindows
    def test_step_out_thunk(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "testit_debug")

        # Make sure we step out of the thunk and end up in testit_debug.
        source = "main.cpp"
        line = line_number(source, "// Step here")
        self.expect(
            "step",
            STEP_IN_SUCCEEDED,
            substrs=["stop reason = step in", "{}:{}".format(source, line)],
        )

        self.runCmd("continue")

        self.expect(
            "step",
            STEP_IN_SUCCEEDED,
            substrs=["stop reason = step in", "Derived2::doit_debug"],
        )
