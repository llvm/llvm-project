"""
Test many basic expression commands and SBFrame.EvaluateExpression() API.

Test cases:

o test_many_expr_commands:
  Test many basic expression commands.
o test_evaluate_expression_python:
  Use Python APIs (SBFrame.EvaluateExpression()) to evaluate expressions.
o test_expr_commands_can_handle_quotes:
  Throw some expression commands with quotes at lldb.
"""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class BasicExprCommandsTestCase(TestBase):

    mydir = os.path.join("expression_command", "test")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test many expressions while stopped at this line:')

    def test_many_expr_commands(self):
        """These basic expression commands should work as expected."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expression 2",
            patterns = ["\(int\) \$.* = 2"])
        # (int) $0 = 1

        self.expect("expression 2ull",
            patterns = ["\(unsigned long long\) \$.* = 2"])
        # (unsigned long long) $1 = 2

        self.expect("expression 2.234f",
            patterns = ["\(float\) \$.* = 2\.234"])
        # (float) $2 = 2.234

        self.expect("expression 2.234",
            patterns = ["\(double\) \$.* = 2\.234"])
        # (double) $3 = 2.234

        self.expect("expression 2+3",
            patterns = ["\(int\) \$.* = 5"])
        # (int) $4 = 5

        self.expect("expression argc",
            patterns = ["\(int\) \$.* = 1"])
        # (int) $5 = 1

        self.expect("expression argc + 22",
            patterns = ["\(int\) \$.* = 23"])
        # (int) $6 = 23

        self.expect("expression argv",
            patterns = ["\(const char \*\*\) \$.* = 0x"])
        # (const char *) $7 = ...

        self.expect("expression argv[0]",
            substrs = ["(const char *)", 
                       "a.out"])
        # (const char *) $8 = 0x... "/Volumes/data/lldb/svn/trunk/test/expression_command/test/a.out"

    @python_api_test
    def test_evaluate_expression_python(self):
        """Test SBFrame.EvaluateExpression() API for evaluating an expression."""
        self.buildDefault()

        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint.
        filespec = lldb.SBFileSpec("main.cpp", False)
        breakpoint = target.BreakpointCreateByLocation(filespec, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(str(breakpoint), BREAKPOINT_CREATED, exe=False,
            substrs = ['main.cpp',
                       str(self.line)])

        # Launch the process, and do not stop at the entry point.
        # Pass 'X Y Z' as the args, which makes argc == 4.
        process = target.LaunchSimple(['X', 'Y', 'Z'], None, os.getcwd())

        if not process:
            self.fail("SBTarget.LaunchProcess() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        # The stop reason of the thread should be breakpoint.
        thread = process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # The filename of frame #0 should be 'main.cpp' and function is main.
        self.expect(lldbutil.get_filenames(thread)[0],
                    "Break correctly at main.cpp", exe=False,
            startstr = "main.cpp")
        self.expect(lldbutil.get_function_names(thread)[0],
                    "Break correctly at main()", exe=False,
            startstr = "main")

        # We should be stopped on the breakpoint with a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        #
        # Use Python API to evaluate expressions while stopped in a stack frame.
        #
        frame = thread.GetFrameAtIndex(0)

        val = frame.EvaluateExpression("2.234")
        self.expect(val.GetValue(), "2.345 evaluated correctly", exe=False,
            startstr = "2.234")
        self.expect(val.GetTypeName(), "2.345 evaluated correctly", exe=False,
            startstr = "double")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("argc")
        self.expect(val.GetValue(), "Argc evaluated correctly", exe=False,
            startstr = "4")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[1]")
        self.expect(val.GetValue(), "Argv[1] evaluated correctly", exe=False,
            startstr = "'X'")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[2]")
        self.expect(val.GetValue(), "Argv[2] evaluated correctly", exe=False,
            startstr = "'Y'")
        self.DebugSBValue(val)

        val = frame.EvaluateExpression("*argv[3]")
        self.expect(val.GetValue(), "Argv[3] evaluated correctly", exe=False,
            startstr = "'Z'")
        self.DebugSBValue(val)

    # rdar://problem/8686536
    # CommandInterpreter::HandleCommand is stripping \'s from input for WantsRawCommand commands
    def test_expr_commands_can_handle_quotes(self):
        """Throw some expression commands with quotes at lldb."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line, num_expected_locations=1,loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # runCmd: expression 'a'
        # output: (char) $0 = 'a'
        self.expect("expression 'a'",
            substrs = ['(char) $',
                       "'a'"])

        # runCmd: expression (int) printf ("\n\n\tHello there!\n")
        # output: (int) $1 = 16
        self.expect(r'''expression (int) printf ("\n\n\tHello there!\n")''',
            substrs = ['(int) $',
                       '16'])

        # runCmd: expression (int) printf("\t\x68\n")
        # output: (int) $2 = 3
        self.expect(r'''expression (int) printf("\t\x68\n")''',
            substrs = ['(int) $',
                       '3'])

        # runCmd: expression (int) printf("\"\n")
        # output: (int) $3 = 2
        self.expect(r'''expression (int) printf("\"\n")''',
            substrs = ['(int) $',
                       '2'])

        # runCmd: expression (int) printf("'\n")
        # output: (int) $4 = 2
        self.expect(r'''expression (int) printf("'\n")''',
            substrs = ['(int) $',
                       '2'])

        # runCmd: command alias print_hi expression (int) printf ("\n\tHi!\n")
        # output: 
        self.runCmd(r'''command alias print_hi expression (int) printf ("\n\tHi!\n")''')
        # This fails currently.
        self.expect('print_hi',
            substrs = ['(int) $',
                       '6'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
