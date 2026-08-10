"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PrintArrayTestCase(TestBase):
    def test_print_array(self):
        """Test that expr -Z works"""
        self.build()
        self.printarray_data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "break here")

    def printarray_data_formatter_commands(self):
        """Test that expr -Z works"""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "expr --element-count 3 -- data", substrs=["[0] = 1", "[1] = 3", "[2] = 5"]
        )
        self.expect("expr data", substrs=["int *", "$", "0x"])
        self.expect(
            "expr -f binary --element-count 0 -- data", substrs=["int *", "$", "0b"]
        )
        self.expect(
            "expr -f hex --element-count 3 -- data",
            substrs=["[0] = 0x", "1", "[1] = 0x", "3", "[2] = 0x", "5"],
        )
        self.expect(
            "expr -f binary --element-count 2 -- data",
            substrs=["int *", "$", "0x", "[0] = 0b", "1", "[1] = 0b", "11"],
        )
        self.expect("parray 3 data", substrs=["[0] = 1", "[1] = 3", "[2] = 5"])
        self.expect(
            "parray `1 + 1 + 1` data", substrs=["[0] = 1", "[1] = 3", "[2] = 5"]
        )
        self.expect("parray `data[1]` data", substrs=["[0] = 1", "[1] = 3", "[2] = 5"])
        self.expect(
            "parray/x 3 data",
            substrs=["[0] = 0x", "1", "[1] = 0x", "3", "[2] = 0x", "5"],
        )
        self.expect(
            "parray/x `data[1]` data",
            substrs=["[0] = 0x", "1", "[1] = 0x", "3", "[2] = 0x", "5"],
        )

        # check error conditions
        self.expect(
            "expr --element-count 10 -- 123",
            error=True,
            substrs=[
                "expression cannot be used with --element-count as it does not refer to a pointer"
            ],
        )
        self.expect(
            "expr --element-count 10 -- (void*)123",
            error=True,
            substrs=[
                "expression cannot be used with --element-count as it refers to a pointer to void"
            ],
        )
        self.expect("parray data", error=True, substrs=["invalid element count 'data'"])
        self.expect(
            "parray data data", error=True, substrs=["invalid element count 'data'"]
        )
        self.expect("parray", error=True, substrs=["Not enough arguments provided"])
