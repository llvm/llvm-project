import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThunkTest(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "testit")

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
