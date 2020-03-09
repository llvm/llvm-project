"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class Radar9974002DataFormatterTestCase(TestBase):

    # test for rdar://problem/9974002 ()
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        if "clang" in self.getCompiler() and "3.4" in self.getCompilerVersion():
            self.skipTest(
                "llvm.org/pr16214 -- clang emits partial DWARF for structures referenced via typedef")

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
            self.runCmd('type summary clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer.first}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '<parent is NULL>'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer%S}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd("type summary add -s foo contained")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', 'foo'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', 'foo'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer%V}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer.first}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '<parent is NULL>'])

        self.runCmd("type summary delete contained")
        self.runCmd("n")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '<parent is NULL>'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer%S}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd("type summary add -s foo contained")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', 'foo'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', 'foo'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer%V}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '0x000000'])

        self.runCmd(
            "type summary add -s \"${var.scalar} and ${var.pointer.first}\" container")

        self.expect('frame variable mine',
                    substrs=['mine = ',
                             '1', '<parent is NULL>'])
