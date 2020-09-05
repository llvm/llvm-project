"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ValueObjectRecursionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @no_debug_info_test
    def test_with_run_command(self):
        """Test that deeply nested ValueObjects still work."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        root = self.frame().FindVariable("root")
        child = root.GetChildAtIndex(1)
        if self.TraceOn():
            print(root)
            print(child)
        for i in range(0, 15000):
            child = child.GetChildAtIndex(1)
        if self.TraceOn():
            print(child)
        self.assertTrue(
            child.IsValid(),
            "could not retrieve the deep ValueObject")
        self.assertTrue(
            child.GetChildAtIndex(0).IsValid(),
            "the deep ValueObject has no value")
        self.assertTrue(
            child.GetChildAtIndex(0).GetValueAsUnsigned() != 0,
            "the deep ValueObject has a zero value")
        self.assertTrue(
            child.GetChildAtIndex(1).GetValueAsUnsigned() != 0,
            "the deep ValueObject has no next")
