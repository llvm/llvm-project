"""
Abstract base class of basic types provides a generic type tester method.
"""

import os, time
import re
import lldb
from lldbtest import *
import lldbutil

def Msg(var, val, using_frame_variable):
    return "'%s %s' matches the output (from compiled code): %s" % (
        'frame variable --show-types' if using_frame_variable else 'expression' ,var, val)

class GenericTester(TestBase):

    # This is the pattern by design to match the " var = 'value'" output from
    # printf() stmts (see basic_type.cpp).
    pattern = re.compile(" (\*?a[^=]*) = '([^=]*)'$")

    # Assert message.
    DATA_TYPE_GROKKED = "Data type from expr parser output is parsed correctly"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        # There are a bunch of test cases under test/types and we don't want the
        # module cacheing subsystem to be confused with executable name "a.out"
        # used for all the test cases.
        self.exe_name = self.testMethodName

    def tearDown(self):
        """Cleanup the test byproducts."""
        TestBase.tearDown(self)
        #print "Removing golden-output.txt..."
        os.remove("golden-output.txt")

    #==========================================================================#
    # Functions build_and_run() and build_and_run_expr() are generic functions #
    # which are called from the Test*Types*.py test cases.  The API client is  #
    # responsible for supplying two mandatory arguments: the source file, e.g.,#
    # 'int.cpp', and the atoms, e.g., set(['unsigned', 'long long']) to the    #
    # functions.  There are also three optional keyword arguments of interest, #
    # as follows:                                                              #
    #                                                                          #
    # dsym -> build for dSYM (defaulted to True)                               #
    #         True: build dSYM file                                            #
    #         False: build DWARF map                                           #
    # bc -> blockCaptured (defaulted to False)                                 #
    #         True: testing vars of various basic types from isnide a block    #
    #         False: testing vars of various basic types from a function       #
    # qd -> quotedDisplay (defaulted to False)                                 #
    #         True: the output from 'frame var' or 'expr var' contains a pair  #
    #               of single quotes around the value                          #
    #         False: no single quotes are to be found around the value of      #
    #                variable                                                  #
    #==========================================================================#

    def build_and_run(self, source, atoms, dsym=True, bc=False, qd=False):
        self.build_and_run_with_source_atoms_expr(source, atoms, expr=False, dsym=dsym, bc=bc, qd=qd)

    def build_and_run_expr(self, source, atoms, dsym=True, bc=False, qd=False):
        self.build_and_run_with_source_atoms_expr(source, atoms, expr=True, dsym=dsym, bc=bc, qd=qd)

    def build_and_run_with_source_atoms_expr(self, source, atoms, expr, dsym=True, bc=False, qd=False):
        # See also Makefile and basic_type.cpp:177.
        if bc:
            d = {'CXX_SOURCES': source, 'EXE': self.exe_name, 'CFLAGS_EXTRAS': '-DTEST_BLOCK_CAPTURED_VARS'}
        else:
            d = {'CXX_SOURCES': source, 'EXE': self.exe_name}
        if dsym:
            self.buildDsym(dictionary=d)
        else:
            self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        if expr:
            self.generic_type_expr_tester(self.exe_name, atoms, blockCaptured=bc, quotedDisplay=qd)
        else:
            self.generic_type_tester(self.exe_name, atoms, blockCaptured=bc, quotedDisplay=qd)

    def generic_type_tester(self, exe_name, atoms, quotedDisplay=False, blockCaptured=False):
        """Test that variables with basic types are displayed correctly."""

        self.runCmd("file %s" % exe_name, CURRENT_EXECUTABLE_SET)

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        self.runCmd("process launch -o golden-output.txt")
        with open("golden-output.txt") as f:
            go = f.read()

        # This golden list contains a list of (variable, value) pairs extracted
        # from the golden output.
        gl = []

        # Scan the golden output line by line, looking for the pattern:
        #
        #     variable = 'value'
        #
        for line in go.split(os.linesep):
            # We'll ignore variables of array types from inside a block.
            if blockCaptured and '[' in line:
                continue
            match = self.pattern.search(line)
            if match:
                var, val = match.group(1), match.group(2)
                gl.append((var, val))
        #print "golden list:", gl

        # This test uses a #include of a the "basic_type.cpp" so we need to enable
        # always setting inlined breakpoints.
        self.runCmd('settings set target.inline-breakpoint-strategy always')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings set target.inline-breakpoint-strategy headers"))

        # Bring the program to the point where we can issue a series of
        # 'frame variable --show-types' command.
        if blockCaptured:
            break_line = line_number ("basic_type.cpp", "// Break here to test block captured variables.")
        else:
            break_line = line_number ("basic_type.cpp", "// Here is the line we will break on to check variables.")
        lldbutil.run_break_set_by_file_and_line (self, "basic_type.cpp", break_line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at basic_type.cpp:%d" % break_line,
                       "stop reason = breakpoint"])

        #self.runCmd("frame variable --show-types")

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable --show-types var'.
        for var, val in gl:
            self.runCmd("frame variable --show-types %s" % var)
            output = self.res.GetOutput()

            # The input type is in a canonical form as a set of named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: frame variable --show-types a_array_bounded[0]
            #     output: (char) a_array_bounded[0] = 'a'
            #
            try:
                dt = re.match("^\((.*)\)", output).group(1)
            except:
                self.fail(self.DATA_TYPE_GROKKED)

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The (var, val) pair must match, too.
            nv = ("%s = '%s'" if quotedDisplay else "%s = %s") % (var, val)
            self.expect(output, Msg(var, val, True), exe=False,
                substrs = [nv])

    def generic_type_expr_tester(self, exe_name, atoms, quotedDisplay=False, blockCaptured=False):
        """Test that variable expressions with basic types are evaluated correctly."""

        self.runCmd("file %s" % exe_name, CURRENT_EXECUTABLE_SET)

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        self.runCmd("process launch -o golden-output.txt")
        with open("golden-output.txt") as f:
            go = f.read()

        # This golden list contains a list of (variable, value) pairs extracted
        # from the golden output.
        gl = []

        # Scan the golden output line by line, looking for the pattern:
        #
        #     variable = 'value'
        #
        for line in go.split(os.linesep):
            # We'll ignore variables of array types from inside a block.
            if blockCaptured and '[' in line:
                continue
            match = self.pattern.search(line)
            if match:
                var, val = match.group(1), match.group(2)
                gl.append((var, val))
        #print "golden list:", gl

        # This test uses a #include of a the "basic_type.cpp" so we need to enable
        # always setting inlined breakpoints.
        self.runCmd('settings set target.inline-breakpoint-strategy always')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(
            lambda: self.runCmd("settings set target.inline-breakpoint-strategy headers"))

        # Bring the program to the point where we can issue a series of
        # 'expr' command.
        if blockCaptured:
            break_line = line_number ("basic_type.cpp", "// Break here to test block captured variables.")
        else:
            break_line = line_number ("basic_type.cpp", "// Here is the line we will break on to check variables.")
        lldbutil.run_break_set_by_file_and_line (self, "basic_type.cpp", break_line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
            substrs = [" at basic_type.cpp:%d" % break_line,
                       "stop reason = breakpoint"])

        #self.runCmd("frame variable --show-types")

        # Now iterate through the golden list, comparing against the output from
        # 'expr var'.
        for var, val in gl:
            # Don't overwhelm the expression mechanism.
            # This slows down the test suite quite a bit, to enable it, define
            # the environment variable LLDB_TYPES_EXPR_TIME_WAIT.  For example:
            #
            #     export LLDB_TYPES_EXPR_TIME_WAIT=0.5
            #
            # causes a 0.5 second delay between 'expression' commands.
            if "LLDB_TYPES_EXPR_TIME_WAIT" in os.environ:
                time.sleep(float(os.environ["LLDB_TYPES_EXPR_TIME_WAIT"]))

            self.runCmd("expression %s" % var)
            output = self.res.GetOutput()

            # The input type is in a canonical form as a set of named atoms.
            # The display type string must conatin each and every element.
            #
            # Example:
            #     runCmd: expr a
            #     output: (double) $0 = 1100.12
            #
            try:
                dt = re.match("^\((.*)\) \$[0-9]+ = ", output).group(1)
            except:
                self.fail(self.DATA_TYPE_GROKKED)

            # Expect the display type string to contain each and every atoms.
            self.expect(dt,
                        "Display type: '%s' must contain the type atoms: '%s'" %
                        (dt, atoms),
                        exe=False,
                substrs = list(atoms))

            # The val part must match, too.
            valPart = ("'%s'" if quotedDisplay else "%s") % val
            self.expect(output, Msg(var, val, False), exe=False,
                substrs = [valPart])
