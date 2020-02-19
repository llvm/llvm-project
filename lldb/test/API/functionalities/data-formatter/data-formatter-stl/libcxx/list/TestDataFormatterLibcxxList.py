"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxListDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')
        self.line2 = line_number('main.cpp',
                                 '// Set second break point at this line.')
        self.line3 = line_number('main.cpp',
                                 '// Set third break point at this line.')
        self.line4 = line_number('main.cpp',
                                 '// Set fourth break point at this line.')

    @add_test_categories(["libc++"])
    @skipIf(debug_info="gmodules",
            bugnumber="https://bugs.llvm.org/show_bug.cgi?id=36048")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line2, num_expected_locations=-1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line3, num_expected_locations=-1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line4, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        lldbutil.skip_if_library_missing(
            self, self.target(), lldbutil.PrintableRegex("libc\+\+"))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("frame variable numbers_list --show-types")
        self.runCmd(
            "type summary add std::int_list std::string_list int_list string_list --summary-string \"list has ${svar%#} items\" -e")
        self.runCmd("type format add -f hex int")

        self.expect("frame variable numbers_list --raw", matching=False,
                    substrs=['list has 0 items',
                             '{}'])

        self.expect("frame variable numbers_list",
                    substrs=['list has 0 items',
                             '{}'])

        self.expect("p numbers_list",
                    substrs=['list has 0 items',
                             '{}'])

        self.runCmd("n") # This gets up past the printf
        self.runCmd("n") # Now advance over the first push_back.

        self.expect("frame variable numbers_list",
                    substrs=['list has 1 items',
                             '[0] = ',
                             '0x12345678'])

        self.runCmd("n")
        self.runCmd("n")
        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs=['list has 4 items',
                             '[0] = ',
                             '0x12345678',
                             '[1] =',
                             '0x11223344',
                             '[2] =',
                             '0xbeeffeed',
                             '[3] =',
                             '0x00abba00'])

        self.runCmd("n")
        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs=['list has 6 items',
                             '[0] = ',
                             '0x12345678',
                             '0x11223344',
                             '0xbeeffeed',
                             '0x00abba00',
                             '[4] =',
                             '0x0abcdef0',
                             '[5] =',
                             '0x0cab0cab'])

        self.expect("p numbers_list",
                    substrs=['list has 6 items',
                             '[0] = ',
                             '0x12345678',
                             '0x11223344',
                             '0xbeeffeed',
                             '0x00abba00',
                             '[4] =',
                             '0x0abcdef0',
                             '[5] =',
                             '0x0cab0cab'])

        # check access-by-index
        self.expect("frame variable numbers_list[0]",
                    substrs=['0x12345678'])
        self.expect("frame variable numbers_list[1]",
                    substrs=['0x11223344'])

        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs=['list has 0 items',
                             '{}'])

        self.runCmd("n")
        self.runCmd("n")
        self.runCmd("n")
        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs=['list has 4 items',
                             '[0] = ', '1',
                             '[1] = ', '2',
                             '[2] = ', '3',
                             '[3] = ', '4'])

        ListPtr = self.frame().FindVariable("list_ptr")
        self.assertTrue(ListPtr.GetChildAtIndex(
            0).GetValueAsUnsigned(0) == 1, "[0] = 1")

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("numbers_list").MightHaveChildren(),
            "numbers_list.MightHaveChildren() says False for non empty!")

        self.runCmd("type format delete int")

        self.runCmd("c")

        self.expect("frame variable text_list",
                    substrs=['list has 3 items',
                             '[0]', 'goofy',
                             '[1]', 'is',
                             '[2]', 'smart'])

        # check that MightHaveChildren() gets it right
        self.assertTrue(
            self.frame().FindVariable("text_list").MightHaveChildren(),
            "text_list.MightHaveChildren() says False for non empty!")

        self.expect("p text_list",
                    substrs=['list has 3 items',
                             '\"goofy\"',
                             '\"is\"',
                             '\"smart\"'])

        self.runCmd("n") # This gets us past the printf
        self.runCmd("n")
        self.runCmd("n")

        # check access-by-index
        self.expect("frame variable text_list[0]",
                    substrs=['goofy'])
        self.expect("frame variable text_list[3]",
                    substrs=['!!!'])

        self.runCmd("continue")

        # check that the list provider correctly updates if elements move
        countingList = self.frame().FindVariable("countingList")
        countingList.SetPreferDynamicValue(True)
        countingList.SetPreferSyntheticValue(True)

        self.assertTrue(countingList.GetChildAtIndex(
            0).GetValueAsUnsigned(0) == 3141, "list[0] == 3141")
        self.assertTrue(countingList.GetChildAtIndex(
            1).GetValueAsUnsigned(0) == 3141, "list[1] == 3141")

        self.runCmd("continue")

        self.assertTrue(
            countingList.GetChildAtIndex(0).GetValueAsUnsigned(0) == 3141,
            "uniqued list[0] == 3141")
        self.assertTrue(
            countingList.GetChildAtIndex(1).GetValueAsUnsigned(0) == 3142,
            "uniqued list[1] == 3142")
