"""Test breakpoint by file/line number; and list variables with array types."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ArrayTypesTestCase(TestBase):

    mydir = os.path.join("lang", "c", "array_types")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test 'frame variable var_name' on some variables with array types."""
        self.buildDsym()
        self.array_types()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test    
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Use Python APIs to inspect variables with array types."""
        self.buildDsym()
        self.array_types_python()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Test 'frame variable var_name' on some variables with array types."""
        self.buildDwarf()
        self.array_types()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Use Python APIs to inspect variables with array types."""
        self.buildDwarf()
        self.array_types_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def array_types(self):
        """Test 'frame variable var_name' on some variables with array types."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line (self, "main.c", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # The test suite sometimes shows that the process has exited without stopping.
        #
        # CC=clang ./dotest.py -v -t array_types
        # ...
        # Process 76604 exited with status = 0 (0x00000000)
        self.runCmd("process status")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
            substrs = ['resolved, hit count = 1'])

        # Issue 'variable list' command on several array-type variables.

        self.expect("frame variable --show-types strings", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(char *[4])',
            substrs = ['(char *) [0]',
                       '(char *) [1]',
                       '(char *) [2]',
                       '(char *) [3]',
                       'Hello',
                       'Hola',
                       'Bonjour',
                       'Guten Tag'])

        self.expect("frame variable --show-types --raw -- char_16", VARIABLES_DISPLAYED_CORRECTLY,
            substrs = ['(char) [0]',
                       '(char) [15]'])

        self.expect("frame variable --show-types ushort_matrix", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(unsigned short [2][3])')

        self.expect("frame variable --show-types long_6", VARIABLES_DISPLAYED_CORRECTLY,
            startstr = '(long [6])')

    def array_types_python(self):
        """Use Python APIs to inspect variables with array types."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.c", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Sanity check the print representation of breakpoint.
        bp = str(breakpoint)
        self.expect(bp, msg="Breakpoint looks good", exe=False,
            substrs = ["file ='main.c'",
                       "line = %d" % self.line,
                       "locations = 1"])
        self.expect(bp, msg="Breakpoint is not resolved as yet", exe=False, matching=False,
            substrs = ["resolved = 1"])

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Sanity check the print representation of process.
        proc = str(process)
        self.expect(proc, msg="Process looks good", exe=False,
            substrs = ["state = stopped",
                       "executable = a.out"])

        # The stop reason of the thread should be breakpoint.
        thread = process.GetThreadAtIndex(0)
        if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
            from lldbutil import stop_reason_to_str
            self.fail(STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS %
                      stop_reason_to_str(thread.GetStopReason()))

        # Sanity check the print representation of thread.
        thr = str(thread)
        self.expect(thr, "Thread looks good with stop reason = breakpoint", exe=False,
            substrs = ["tid = 0x%4.4x" % thread.GetThreadID()])

        # The breakpoint should have a hit count of 1.
        self.assertTrue(breakpoint.GetHitCount() == 1, BREAKPOINT_HIT_ONCE)

        # The breakpoint should be resolved by now.
        bp = str(breakpoint)
        self.expect(bp, "Breakpoint looks good and is resolved", exe=False,
            substrs = ["file ='main.c'",
                       "line = %d" % self.line,
                       "locations = 1"])

        # Sanity check the print representation of frame.
        frame = thread.GetFrameAtIndex(0)
        frm = str(frame)
        self.expect(frm,
                    "Frame looks good with correct index %d" % frame.GetFrameID(),
                    exe=False,
            substrs = ["#%d" % frame.GetFrameID()])

        # Lookup the "strings" string array variable and sanity check its print
        # representation.
        variable = frame.FindVariable("strings")
        var = str(variable)
        self.expect(var, "Variable for 'strings' looks good with correct name", exe=False,
            substrs = ["%s" % variable.GetName()])
        self.DebugSBValue(variable)
        self.assertTrue(variable.GetNumChildren() == 4,
                        "Variable 'strings' should have 4 children")

        child3 = variable.GetChildAtIndex(3)
        self.DebugSBValue(child3)
        self.assertTrue(child3.GetSummary() == '"Guten Tag"',
                        'strings[3] == "Guten Tag"')

        # Lookup the "char_16" char array variable.
        variable = frame.FindVariable("char_16")
        self.DebugSBValue(variable)
        self.assertTrue(variable.GetNumChildren() == 16,
                        "Variable 'char_16' should have 16 children")

        # Lookup the "ushort_matrix" ushort[] array variable.
        # Notice the pattern of int(child0_2.GetValue(), 0).  We pass a
        # base of 0 so that the proper radix is determined based on the contents
        # of the string.  Same applies to long().
        variable = frame.FindVariable("ushort_matrix")
        self.DebugSBValue(variable)
        self.assertTrue(variable.GetNumChildren() == 2,
                        "Variable 'ushort_matrix' should have 2 children")
        child0 = variable.GetChildAtIndex(0)
        self.DebugSBValue(child0)
        self.assertTrue(child0.GetNumChildren() == 3,
                        "Variable 'ushort_matrix[0]' should have 3 children")
        child0_2 = child0.GetChildAtIndex(2)
        self.DebugSBValue(child0_2)
        self.assertTrue(int(child0_2.GetValue(), 0) == 3,
                        "ushort_matrix[0][2] == 3")

        # Lookup the "long_6" char array variable.
        variable = frame.FindVariable("long_6")
        self.DebugSBValue(variable)
        self.assertTrue(variable.GetNumChildren() == 6,
                        "Variable 'long_6' should have 6 children")
        child5 = variable.GetChildAtIndex(5)
        self.DebugSBValue(child5)
        self.assertTrue(long(child5.GetValue(), 0) == 6,
                        "long_6[5] == 6")

        # Last, check that "long_6" has a value type of eValueTypeVariableLocal
        # and "argc" has eValueTypeVariableArgument.
        from lldbutil import value_type_to_str
        self.assertTrue(variable.GetValueType() == lldb.eValueTypeVariableLocal,
                        "Variable 'long_6' should have '%s' value type." %
                        value_type_to_str(lldb.eValueTypeVariableLocal))
        argc = frame.FindVariable("argc")
        self.DebugSBValue(argc)
        self.assertTrue(argc.GetValueType() == lldb.eValueTypeVariableArgument,
                        "Variable 'argc' should have '%s' value type." %
                        value_type_to_str(lldb.eValueTypeVariableArgument))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
