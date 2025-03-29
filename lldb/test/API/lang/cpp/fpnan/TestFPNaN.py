"""
Test floating point expressions with zero, NaN, dernormalized and infinite
numbers.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FPNaNTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)
        # Zero and denorm
        self.expect(
            "expr +0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "0"],
        )
        self.expect(
            "expr -0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "0"],
        )
        self.expect(
            "expr 0.0 / 0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "NaN"],
        )
        self.expect(
            "expr 0 / 0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "NaN"],
        )
        self.expect(
            "expr 1 / +0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "+Inf"],
        )
        self.expect(
            "expr 1 / -0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "-Inf"],
        )
        self.expect(
            "expr +0.0 / +0.0 != +0.0 / +0.0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "true"],
        )
        self.expect(
            "expr -1.f * 0",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["float", "-0"],
        )
        self.expect(
            "expr 0x0.123p-1",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["double", "0.0355224609375"],
        )
        # NaN
        self.expect(
            "expr fnan < fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr fnan <= fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr fnan > fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr fnan >= fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr fnan == fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr fnan != fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "true"],
        )
        self.expect(
            "expr 1.0 <= fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr 1.0f < fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "false"],
        )
        self.expect(
            "expr 1.0f != fnan",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["bool", "true"],
        )
        self.expect(
            "expr (unsigned int) fdenorm",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["int", "0"],
        )
        self.expect(
            "expr (unsigned int) (1.0f + fdenorm)",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["int", "1"],
        )
