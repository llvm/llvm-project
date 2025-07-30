"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdIteratorDataFormatterTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")
        self.namespace = "std"

    def do_test(self):
        """Test that iterators format properly."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1
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
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect("frame variable ivI", substrs=["item = 3"])
        self.expect("expr ivI", substrs=["item = 3"])

        self.expect("frame variable iimI", substrs=["first = 43981", "second = 61681"])
        self.expect("expr iimI", substrs=["first = 43981", "second = 61681"])

        self.expect("frame variable iimI.first", substrs=["first = 43981"])
        self.expect("frame variable iimI.first", substrs=["second"], matching=False)
        self.expect("frame variable iimI.second", substrs=["second = 61681"])
        self.expect("frame variable iimI.second", substrs=["first"], matching=False)

        self.expect("frame variable simI", substrs=['first = "world"', "second = 42"])
        self.expect("expr simI", substrs=['first = "world"', "second = 42"])

        self.expect("frame variable simI.first", substrs=['first = "world"'])
        self.expect("frame variable simI.first", substrs=["second"], matching=False)
        self.expect("frame variable simI.second", substrs=["second = 42"])
        self.expect("frame variable simI.second", substrs=["first"], matching=False)

        self.expect("frame variable svI", substrs=['item = "hello"'])
        self.expect("expr svI", substrs=['item = "hello"'])

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()
