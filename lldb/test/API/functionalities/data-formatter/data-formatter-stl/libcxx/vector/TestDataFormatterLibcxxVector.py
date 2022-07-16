"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxVectorDataFormatterTestCase(TestBase):

    def check_numbers(self, var_name):
        self.expect("frame variable " + var_name,
                    substrs=[var_name + ' = size=7',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '[4] = 12345',
                             '[5] = 123456',
                             '[6] = 1234567',
                             '}'])

        self.expect_expr(var_name, result_summary="size=7", result_children=[
            ValueCheck(value="1"),
            ValueCheck(value="12"),
            ValueCheck(value="123"),
            ValueCheck(value="1234"),
            ValueCheck(value="12345"),
            ValueCheck(value="123456"),
            ValueCheck(value="1234567"),
        ])

        # check access-by-index
        self.expect("frame variable " + var_name + "[0]",
                    substrs=['1'])
        self.expect("frame variable " + var_name + "[1]",
                    substrs=['12'])
        self.expect("frame variable " + var_name + "[2]",
                    substrs=['123'])
        self.expect("frame variable " + var_name + "[3]",
                    substrs=['1234'])

    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False))

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

        # empty vectors (and storage pointers SHOULD BOTH BE NULL..)
        self.expect("frame variable numbers",
                    substrs=['numbers = size=0'])

        lldbutil.continue_to_breakpoint(process, bkpt)

        # first value added
        self.expect("frame variable numbers",
                    substrs=['numbers = size=1',
                             '[0] = 1',
                             '}'])

        # add some more data
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("frame variable numbers",
                    substrs=['numbers = size=4',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '}'])

        self.expect("p numbers",
                    substrs=['$', 'size=4',
                             '[0] = 1',
                             '[1] = 12',
                             '[2] = 123',
                             '[3] = 1234',
                             '}'])

        # check access to synthetic children
        self.runCmd(
            "type summary add --summary-string \"item 0 is ${var[0]}\" std::int_vect int_vect")
        self.expect('frame variable numbers',
                    substrs=['item 0 is 1'])

        self.runCmd(
            "type summary add --summary-string \"item 0 is ${svar[0]}\" std::int_vect int_vect")
        self.expect('frame variable numbers',
                    substrs=['item 0 is 1'])
        # move on with synths
        self.runCmd("type summary delete std::int_vect")
        self.runCmd("type summary delete int_vect")

        # add some more data
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check_numbers("numbers")

        # clear out the vector and see that we do the right thing once again
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("frame variable numbers",
                    substrs=['numbers = size=0'])

        lldbutil.continue_to_breakpoint(process, bkpt)

        # first value added
        self.expect("frame variable numbers",
                    substrs=['numbers = size=1',
                             '[0] = 7',
                             '}'])

        # check if we can display strings
        self.expect("frame variable strings",
                    substrs=['goofy',
                             'is',
                             'smart'])

        self.expect("p strings",
                    substrs=['goofy',
                             'is',
                             'smart'])

        # test summaries based on synthetic children
        self.runCmd(
            "type summary add std::string_vect string_vect --summary-string \"vector has ${svar%#} items\" -e")
        self.expect("frame variable strings",
                    substrs=['vector has 3 items',
                             'goofy',
                             'is',
                             'smart'])

        self.expect("p strings",
                    substrs=['vector has 3 items',
                             'goofy',
                             'is',
                             'smart'])

        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("frame variable strings",
                    substrs=['vector has 4 items'])

        # check access-by-index
        self.expect("frame variable strings[0]",
                    substrs=['goofy'])
        self.expect("frame variable strings[1]",
                    substrs=['is'])

        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("frame variable strings",
                    substrs=['vector has 0 items'])

    @add_test_categories(["libc++"])
    def test_ref_and_ptr(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to check by ref", lldb.SBFileSpec("main.cpp", False))

        # The reference should display the same was as the value did
        self.check_numbers("ref")

        # The pointer should just show the right number of elements:

        self.expect("frame variable ptr", substrs=['ptr =', ' size=7'])

        self.expect("p ptr", substrs=['$', 'size=7'])
