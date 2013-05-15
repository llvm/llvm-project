"""
Test C++ virtual function and virtual inheritance.
"""

import os, time
import re
import lldb
from lldbtest import *
import lldbutil

def Msg(expr, val):
    return "'expression %s' matches the output (from compiled code): %s" % (expr, val)

class CppVirtualMadness(TestBase):

    mydir = os.path.join("lang", "cpp", "virtual")

    # This is the pattern by design to match the "my_expr = 'value'" output from
    # printf() stmts (see main.cpp).
    pattern = re.compile("^([^=]*) = '([^=]*)'$")

    # Assert message.
    PRINTF_OUTPUT_GROKKED = "The printf output from compiled code is parsed correctly"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_virtual_madness_dsym(self):
        """Test that expression works correctly with virtual inheritance as well as virtual function."""
        self.buildDsym()
        self.virtual_madness_test()

    def test_virtual_madness_dwarf(self):
        """Test that expression works correctly with virtual inheritance as well as virtual function."""
        self.buildDwarf()
        self.virtual_madness_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp', '// Set first breakpoint here.')

    def virtual_madness_test(self):
        """Test that variable expressions with basic types are evaluated correctly."""

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        go = system("./a.out", sender=self)[0]
        # This golden list contains a list of "my_expr = 'value' pairs extracted
        # from the golden output.
        gl = []

        # Scan the golden output line by line, looking for the pattern:
        #
        #     my_expr = 'value'
        #
        for line in go.split(os.linesep):
            match = self.pattern.search(line)
            if match:
                my_expr, val = match.group(1), match.group(2)
                gl.append((my_expr, val))
        #print "golden list:", gl

        # Bring the program to the point where we can issue a series of
        # 'expression' command to compare against the golden output.
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)
        self.runCmd("run", RUN_SUCCEEDED)

        # Now iterate through the golden list, comparing against the output from
        # 'expression var'.
        for my_expr, val in gl:
            # Don't overwhelm the expression mechanism.
            # This slows down the test suite quite a bit, to enable it, define
            # the environment variable LLDB_TYPES_EXPR_TIME_WAIT.  For example:
            #
            #     export LLDB_TYPES_EXPR_TIME_WAIT=0.5
            #
            # causes a 0.5 second delay between 'expression' commands.
            if "LLDB_TYPES_EXPR_TIME_WAIT" in os.environ:
                time.sleep(float(os.environ["LLDB_TYPES_EXPR_TIME_WAIT"]))

            self.runCmd("expression %s" % my_expr)
            output = self.res.GetOutput()
            
            # The expression output must match the oracle.
            self.expect(output, Msg(my_expr, val), exe=False,
                substrs = [val])
