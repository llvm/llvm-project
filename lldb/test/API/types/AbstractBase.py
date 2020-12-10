"""
Abstract base class of basic types provides a generic type tester method.
"""

from __future__ import print_function

import os
import re
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


def Msg(var, val, using_frame_variable):
    return "'%s %s' matches the output (from compiled code): %s" % (
        'frame variable --show-types' if using_frame_variable else 'expression', var, val)


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
        golden = "{}-golden-output.txt".format(self.testMethodName)
        if configuration.is_reproducer():
            self.golden_filename = self.getReproducerArtifact(golden)
        else:
            self.golden_filename = self.getBuildArtifact(golden)

    def tearDown(self):
        """Cleanup the test byproducts."""
        if os.path.exists(self.golden_filename) and not configuration.is_reproducer():
            os.remove(self.golden_filename)
        TestBase.tearDown(self)

    #==========================================================================#
    # Functions build_and_run() and build_and_run_expr() are generic functions #
    # which are called from the Test*Types*.py test cases.  The API client is  #
    # responsible for supplying two mandatory arguments: the source file, e.g.,#
    # 'int.cpp', and the atoms, e.g., set(['unsigned', 'long long']) to the    #
    # functions.  There are also three optional keyword arguments of interest, #
    # as follows:                                                              #
    #                                                                          #
    # bc -> blockCaptured (defaulted to False)                                 #
    #         True: testing vars of various basic types from inside a block    #
    #         False: testing vars of various basic types from a function       #
    # qd -> quotedDisplay (defaulted to False)                                 #
    #         True: the output from 'frame var' or 'expr var' contains a pair  #
    #               of single quotes around the value                          #
    #         False: no single quotes are to be found around the value of      #
    #                variable                                                  #
    #==========================================================================#

    def build_and_run(self, source, atoms, bc=False, qd=False):
        self.build_and_run_with_source_atoms_expr(
            source, atoms, expr=False, bc=bc, qd=qd)

    def build_and_run_expr(self, source, atoms, bc=False, qd=False):
        self.build_and_run_with_source_atoms_expr(
            source, atoms, expr=True, bc=bc, qd=qd)

    def build_and_run_with_source_atoms_expr(
            self, source, atoms, expr, bc=False, qd=False):
        # See also Makefile and basic_type.cpp:177.
        if bc:
            d = {'CXX_SOURCES': source, 'EXE': self.exe_name,
                 'CFLAGS_EXTRAS': '-DTEST_BLOCK_CAPTURED_VARS'}
        else:
            d = {'CXX_SOURCES': source, 'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        if expr:
            self.generic_type_expr_tester(
                self.exe_name, atoms, blockCaptured=bc, quotedDisplay=qd)
        else:
            self.generic_type_tester(
                self.exe_name,
                atoms,
                blockCaptured=bc,
                quotedDisplay=qd)

    def process_launch_o(self):
        # process launch command output redirect always goes to host the
        # process is running on
        if lldb.remote_platform:
            # process launch -o requires a path that is valid on the target
            self.assertIsNotNone(lldb.remote_platform.GetWorkingDirectory())
            remote_path = lldbutil.append_to_process_working_directory(self,
                "lldb-stdout-redirect.txt")
            self.runCmd(
                'process launch -- {remote}'.format(remote=remote_path))
            # copy remote_path to local host
            self.runCmd('platform get-file {remote} "{local}"'.format(
                remote=remote_path, local=self.golden_filename))
        elif configuration.is_reproducer_replay():
            # Don't overwrite the golden file generated at capture time.
            self.runCmd('process launch')
        else:
            self.runCmd(
                'process launch -o "{local}"'.format(local=self.golden_filename))

    def get_golden_list(self, blockCaptured=False):
        with open(self.golden_filename, 'r') as f:
            go = f.read()

        golden_list = []
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
                golden_list.append((var, val))
        return golden_list

    def generic_type_tester(
            self,
            exe_name,
            atoms,
            quotedDisplay=False,
            blockCaptured=False):
        """Test that variables with basic types are displayed correctly."""
        self.runCmd("file %s" % self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        self.process_launch_o()

        # This golden list contains a list of (variable, value) pairs extracted
        # from the golden output.
        gl = self.get_golden_list(blockCaptured)

        # This test uses a #include of "basic_type.cpp" so we need to enable
        # always setting inlined breakpoints.
        self.runCmd('settings set target.inline-breakpoint-strategy always')

        # Inherit TCC permissions. We can leave this set.
        self.runCmd('settings set target.inherit-tcc true')

        # Kill rather than detach from the inferior if something goes wrong.
        self.runCmd('settings set target.detach-on-error false')

        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd(
            "settings set target.inline-breakpoint-strategy headers"))

        # Bring the program to the point where we can issue a series of
        # 'frame variable --show-types' command.
        if blockCaptured:
            break_line = line_number(
                "basic_type.cpp",
                "// Break here to test block captured variables.")
        else:
            break_line = line_number(
                "basic_type.cpp",
                "// Here is the line we will break on to check variables.")
        lldbutil.run_break_set_by_file_and_line(
            self,
            "basic_type.cpp",
            break_line,
            num_expected_locations=1,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint",
                             " at basic_type.cpp:%d" % break_line,])

        #self.runCmd("frame variable --show-types")

        # Now iterate through the golden list, comparing against the output from
        # 'frame variable --show-types var'.
        for var, val in gl:
            self.runCmd("frame variable --show-types %s" % var)
            output = self.res.GetOutput()

            # The input type is in a canonical form as a set of named atoms.
            # The display type string must contain each and every element.
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
            self.expect(
                dt, "Display type: '%s' must contain the type atoms: '%s'" %
                (dt, atoms), exe=False, substrs=list(atoms))

            # The (var, val) pair must match, too.
            nv = ("%s = '%s'" if quotedDisplay else "%s = %s") % (var, val)
            self.expect(output, Msg(var, val, True), exe=False,
                        substrs=[nv])

    def generic_type_expr_tester(
            self,
            exe_name,
            atoms,
            quotedDisplay=False,
            blockCaptured=False):
        """Test that variable expressions with basic types are evaluated correctly."""

        self.runCmd("file %s" % self.getBuildArtifact(exe_name),
                    CURRENT_EXECUTABLE_SET)

        # First, capture the golden output emitted by the oracle, i.e., the
        # series of printf statements.
        self.process_launch_o()

        # This golden list contains a list of (variable, value) pairs extracted
        # from the golden output.
        gl = self.get_golden_list(blockCaptured)

        # This test uses a #include of "basic_type.cpp" so we need to enable
        # always setting inlined breakpoints.
        self.runCmd('settings set target.inline-breakpoint-strategy always')
        # And add hooks to restore the settings during tearDown().
        self.addTearDownHook(lambda: self.runCmd(
            "settings set target.inline-breakpoint-strategy headers"))

        # Bring the program to the point where we can issue a series of
        # 'expr' command.
        if blockCaptured:
            break_line = line_number(
                "basic_type.cpp",
                "// Break here to test block captured variables.")
        else:
            break_line = line_number(
                "basic_type.cpp",
                "// Here is the line we will break on to check variables.")
        lldbutil.run_break_set_by_file_and_line(
            self,
            "basic_type.cpp",
            break_line,
            num_expected_locations=1,
            loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("process status", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint",
                             " at basic_type.cpp:%d" % break_line])

        #self.runCmd("frame variable --show-types")

        # Now iterate through the golden list, comparing against the output from
        # 'expr var'.
        for var, val in gl:
            self.runCmd("expression %s" % var)
            output = self.res.GetOutput()

            # The input type is in a canonical form as a set of named atoms.
            # The display type string must contain each and every element.
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
            self.expect(
                dt, "Display type: '%s' must contain the type atoms: '%s'" %
                (dt, atoms), exe=False, substrs=list(atoms))

            # The val part must match, too.
            valPart = ("'%s'" if quotedDisplay else "%s") % val
            self.expect(output, Msg(var, val, False), exe=False,
                        substrs=[valPart])
