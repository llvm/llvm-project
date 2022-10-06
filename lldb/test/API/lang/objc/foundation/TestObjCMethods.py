"""
Set breakpoints on objective-c class and instance methods in foundation.
Also lookup objective-c data types and evaluate expressions.
"""

import os
import os.path
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

file_index = 0


class FoundationTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.main_source = "main.m"
        self.line = line_number(
            self.main_source,
            '// Set break point at this line.')

    def test_break(self):
        """Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at +[NSString stringWithFormat:].
        break_results = lldbutil.run_break_set_command(
            self, "_regexp-break +[NSString stringWithFormat:]")
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            symbol_name='+[NSString stringWithFormat:]',
            num_locations=1)

        # Stop at -[MyString initWithNSString:].
        lldbutil.run_break_set_by_symbol(
            self,
            '-[MyString initWithNSString:]',
            num_expected_locations=1,
            sym_exact=True)

        # Stop at the "description" selector.
        lldbutil.run_break_set_by_selector(
            self,
            'description',
            num_expected_locations=1,
            module_name='a.out')

        # Stop at -[NSAutoreleasePool release].
        break_results = lldbutil.run_break_set_command(
            self, "_regexp-break -[NSAutoreleasePool release]")
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            symbol_name='-[NSAutoreleasePool release]',
            num_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # First stop is +[NSString stringWithFormat:].
        self.expect(
            "thread backtrace",
            "Stop at +[NSString stringWithFormat:]",
            substrs=["Foundation`+[NSString stringWithFormat:]"])

        self.runCmd("process continue")

        # Second stop is still +[NSString stringWithFormat:].
        self.expect(
            "thread backtrace",
            "Stop at +[NSString stringWithFormat:]",
            substrs=["Foundation`+[NSString stringWithFormat:]"])

        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect(
            "thread backtrace",
            "Stop at a.out`-[MyString initWithNSString:]",
            substrs=["a.out`-[MyString initWithNSString:]"])

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
                    substrs=["a.out`-[MyString description]"])

        self.runCmd("process continue")

        # Followed by the same -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
                    substrs=["a.out`-[MyString description]"])

        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect("thread backtrace", "Stop at -[NSAutoreleasePool release]",
                    substrs=["Foundation`-[NSAutoreleasePool release]"])

    # rdar://problem/8542091
    # rdar://problem/8492646
    def test_data_type_and_expr(self):
        """Lookup objective-c data types and evaluate expressions."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at -[MyString description].
        lldbutil.run_break_set_by_symbol(
            self,
            '-[MyString description]',
            num_expected_locations=1,
            sym_exact=True)
#        self.expect("breakpoint set -n '-[MyString description]", BREAKPOINT_CREATED,
# startstr = "Breakpoint created: 1: name = '-[MyString description]',
# locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("settings set target.prefer-dynamic-value no-dynamic-values")

        # The backtrace should show we stop at -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
                    substrs=["a.out`-[MyString description]"])

        # Lookup objc data type MyString and evaluate some expressions.

        self.expect("image lookup -t NSString", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=['name = "NSString"',
                             'compiler_type = "@interface NSString'])

        self.expect("image lookup -t MyString", DATA_TYPES_DISPLAYED_CORRECTLY,
                    substrs=['name = "MyString"',
                             'compiler_type = "@interface MyString',
                             'NSString * str;',
                             'NSDate * date;'])

        self.expect(
            "frame variable --show-types --scope",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["ARG: (MyString *) self"],
            patterns=[
                "ARG: \(.*\) _cmd",
                "(objc_selector *)|(SEL)"])

        # rdar://problem/8651752
        # don't crash trying to ask clang how many children an empty record has
        self.runCmd("frame variable *_cmd")

        # rdar://problem/8492646
        # test/foundation fails after updating to tot r115023
        # self->str displays nothing as output
        self.expect(
            "frame variable --show-types self->str",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr="(NSString *) self->str")

        # rdar://problem/8447030
        # 'frame variable self->date' displays the wrong data member
        self.expect(
            "frame variable --show-types self->date",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr="(NSDate *) self->date")

        # This should display the str and date member fields as well.
        self.expect(
            "frame variable --show-types *self",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(MyString) *self",
                "(NSString *) str",
                "(NSDate *) date"])

        # isa should be accessible.
        self.expect("expression self->isa", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=["Class)"])

        # This should fail expectedly.
        self.expect(
            "expression self->non_existent_member",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["error:", "'MyString' does not have a member named 'non_existent_member'"])

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
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("process continue")

        # rdar://problem/8542091
        # test/foundation: expr -o -- my not working?
        #
        # Test new feature with r115115:
        # Add "-o" option to "expression" which prints the object description
        # if available.
        self.expect(
            "expression --object-description -- my",
            "Object description displayed correctly",
            patterns=["Hello from.*a.out.*with timestamp: "])

    @add_test_categories(['pyapi'])
    def test_print_ivars_correctly(self):
        self.build()
        # See: <rdar://problem/8717050> lldb needs to use the ObjC runtime symbols for ivar offsets
        # Only fails for the ObjC 2.0 runtime.
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        break1 = target.BreakpointCreateByLocation(self.main_source, self.line)
        self.assertTrue(break1, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread = process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbsuite.test.lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # Make sure we stopped at the first breakpoint.

        cur_frame = thread.GetFrameAtIndex(0)

        line_number = cur_frame.GetLineEntry().GetLine()
        self.assertEqual(line_number, self.line, "Hit the first breakpoint.")

        my_var = cur_frame.FindVariable("my")
        self.assertTrue(my_var, "Made a variable object for my")

        str_var = cur_frame.FindVariable("str")
        self.assertTrue(str_var, "Made a variable object for str")

        # Now make sure that the my->str == str:

        my_str_var = my_var.GetChildMemberWithName("str")
        self.assertTrue(my_str_var, "Found a str ivar in my")

        str_value = int(str_var.GetValue(), 0)

        my_str_value = int(my_str_var.GetValue(), 0)

        self.assertEqual(
            str_value, my_str_value,
            "Got the correct value for my->str")

    def test_expression_lookups_objc(self):
        """Test running an expression detect spurious debug info lookups (DWARF)."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at -[MyString initWithNSString:].
        lldbutil.run_break_set_by_symbol(
            self,
            '-[MyString initWithNSString:]',
            num_expected_locations=1,
            sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        global file_index
        # Log any DWARF lookups
        ++file_index
        logfile = os.path.join(
            self.getBuildDir(),
            "dwarf-lookups-" +
            self.getArchitecture() +
            "-" +
            str(file_index) +
            ".txt")
        self.runCmd("log enable -f %s dwarf lookups" % (logfile))
        self.runCmd("expr self")
        self.runCmd("log disable dwarf lookups")

        def cleanup():
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)

        if os.path.exists(logfile):
            f = open(logfile)
            lines = f.readlines()
            num_errors = 0
            for line in lines:
                if "$__lldb" in line:
                    if num_errors == 0:
                        print(
                            "error: found spurious name lookups when evaluating an expression:")
                    num_errors += 1
                    print(line, end='')
            self.assertEqual(num_errors, 0, "Spurious lookups detected")
            f.close()
