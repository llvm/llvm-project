"""
Set breakpoints on objective-c class and instance methods in foundation.
Also lookup objective-c data types and evaluate expressions.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")

    @dsym_test
    def test_break_with_dsym(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        self.buildDsym()
        self.break_on_objc_methods()

    @dwarf_test
    def test_break_with_dwarf(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        self.buildDwarf()
        self.break_on_objc_methods()

    #@unittest2.expectedFailure
    # rdar://problem/8542091
    # rdar://problem/8492646
    @dsym_test
    def test_data_type_and_expr_with_dsym(self):
        """Lookup objective-c data types and evaluate expressions."""
        self.buildDsym()
        self.data_type_and_expr_objc()

    #@unittest2.expectedFailure
    # rdar://problem/8542091
    # rdar://problem/8492646
    @dwarf_test
    def test_data_type_and_expr_with_dwarf(self):
        """Lookup objective-c data types and evaluate expressions."""
        self.buildDwarf()
        self.data_type_and_expr_objc()

    @python_api_test
    @dsym_test
    def test_print_ivars_correctly_with_dsym (self):
        self.buildDsym()
        self.print_ivars_correctly()

    @python_api_test
    @dwarf_test
    def test_print_ivars_correctly_with_dwarf (self):
        self.buildDwarf()
        self.print_ivars_correctly()

    def break_on_objc_methods(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at +[NSString stringWithFormat:].
        break_results = lldbutil.run_break_set_command(self, "_regexp-break +[NSString stringWithFormat:]")
        lldbutil.check_breakpoint_result (self, break_results, symbol_name='+[NSString stringWithFormat:]', num_locations=1)

        # Stop at -[MyString initWithNSString:].
        lldbutil.run_break_set_by_symbol (self, '-[MyString initWithNSString:]', num_expected_locations=1, sym_exact=True)

        # Stop at the "description" selector.
        lldbutil.run_break_set_by_selector (self, 'description', num_expected_locations=1, module_name='a.out')

        # Stop at -[NSAutoreleasePool release].
        break_results = lldbutil.run_break_set_command(self, "_regexp-break -[NSAutoreleasePool release]")
        lldbutil.check_breakpoint_result (self, break_results, symbol_name='-[NSAutoreleasePool release]', num_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # First stop is +[NSString stringWithFormat:].
        self.expect("thread backtrace", "Stop at +[NSString stringWithFormat:]",
            substrs = ["Foundation`+[NSString stringWithFormat:]"])

        self.runCmd("process continue")

        # Second stop is still +[NSString stringWithFormat:].
        self.expect("thread backtrace", "Stop at +[NSString stringWithFormat:]",
            substrs = ["Foundation`+[NSString stringWithFormat:]"])

        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect("thread backtrace", "Stop at a.out`-[MyString initWithNSString:]",
            substrs = ["a.out`-[MyString initWithNSString:]"])

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        self.runCmd("process continue")

        # Followed by the same -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect("thread backtrace", "Stop at -[NSAutoreleasePool release]",
            substrs = ["Foundation`-[NSAutoreleasePool release]"])

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(self.main_source, '// Set break point at this line.')

    def data_type_and_expr_objc(self):
        """Lookup objective-c data types and evaluate expressions."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at -[MyString description].
        lldbutil.run_break_set_by_symbol (self, '-[MyString description]', num_expected_locations=1, sym_exact=True)
#        self.expect("breakpoint set -n '-[MyString description]", BREAKPOINT_CREATED,
#            startstr = "Breakpoint created: 1: name = '-[MyString description]', locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # The backtrace should show we stop at -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        # Lookup objc data type MyString and evaluate some expressions.

        self.expect("image lookup -t NSString", DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['name = "NSString"',
                       'clang_type = "@interface NSString'])

        self.expect("image lookup -t MyString", DATA_TYPES_DISPLAYED_CORRECTLY,
            substrs = ['name = "MyString"',
                       'clang_type = "@interface MyString',
                       'NSString * str;',
                       'NSDate * date;'])

        self.expect("frame variable --show-types --scope", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["ARG: (MyString *) self"],
            patterns = ["ARG: \(.*\) _cmd",
                        "(objc_selector *)|(SEL)"])

        # rdar://problem/8651752
        # don't crash trying to ask clang how many children an empty record has
        self.runCmd("frame variable *_cmd")

        # rdar://problem/8492646
        # test/foundation fails after updating to tot r115023
        # self->str displays nothing as output
        self.expect("frame variable --show-types self->str", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(NSString *) self->str")

        # rdar://problem/8447030
        # 'frame variable self->date' displays the wrong data member
        self.expect("frame variable --show-types self->date", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = "(NSDate *) self->date")

        # This should display the str and date member fields as well.
        self.expect("frame variable --show-types *self", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(MyString) *self",
                       "(NSString *) str",
                       "(NSDate *) date"])
        
        # isa should be accessible.
        self.expect("expression self->isa", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ["(Class)"])

        # This should fail expectedly.
        self.expect("expression self->non_existent_member",
                    COMMAND_FAILED_AS_EXPECTED, error=True,
            startstr = "error: 'MyString' does not have a member named 'non_existent_member'")

        # Use expression parser.
        self.runCmd("expression self->str")
        self.runCmd("expression self->date")

        # (lldb) expression self->str
        # error: instance variable 'str' is protected
        # error: 1 errors parsing expression
        #
        # (lldb) expression self->date
        # error: instance variable 'date' is protected
        # error: 1 errors parsing expression
        #

        self.runCmd("breakpoint delete 1")
        lldbutil.run_break_set_by_file_and_line (self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("process continue")

        # rdar://problem/8542091
        # test/foundation: expr -o -- my not working?
        #
        # Test new feature with r115115:
        # Add "-o" option to "expression" which prints the object description if available.
        self.expect("expression --object-description -- my", "Object description displayed correctly",
            patterns = ["Hello from.*a.out.*with timestamp: "])

    # See: <rdar://problem/8717050> lldb needs to use the ObjC runtime symbols for ivar offsets
    # Only fails for the ObjC 2.0 runtime.
    def print_ivars_correctly(self) :
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        break1 = target.BreakpointCreateByLocation(self.main_source, self.line)
        self.assertTrue(break1, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # Make sure we stopped at the first breakpoint.

        cur_frame = thread.GetFrameAtIndex(0)

        line_number = cur_frame.GetLineEntry().GetLine()
        self.assertTrue (line_number == self.line, "Hit the first breakpoint.")

        my_var = cur_frame.FindVariable("my")
        self.assertTrue(my_var, "Made a variable object for my")

        str_var = cur_frame.FindVariable("str")
        self.assertTrue(str_var, "Made a variable object for str")

        # Now make sure that the my->str == str:

        my_str_var = my_var.GetChildMemberWithName("str")
        self.assertTrue(my_str_var, "Found a str ivar in my")

        str_value = int(str_var.GetValue(), 0)

        my_str_value = int(my_str_var.GetValue(), 0)

        self.assertTrue(str_value == my_str_value, "Got the correct value for my->str")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
