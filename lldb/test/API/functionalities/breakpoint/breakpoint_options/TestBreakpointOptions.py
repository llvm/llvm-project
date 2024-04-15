"""
Test breakpoint command for different options.
"""


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class BreakpointOptionsTestCase(TestBase):
    def test(self):
        """Test breakpoint command for different options."""
        self.build()
        self.breakpoint_options_test()
        self.breakpoint_options_language_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def breakpoint_options_test(self):
        """Test breakpoint command for different options."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 locations.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, extra_options="-K 1", num_expected_locations=1
        )
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, extra_options="-K 0", num_expected_locations=1
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 2."],
        )

        # Check the list of breakpoint.
        self.expect(
            "breakpoint list -f",
            "Breakpoint locations shown correctly",
            substrs=[
                "1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.line,
                "2: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.line,
            ],
        )

        # Continue the program, there should be another stop.
        self.runCmd("process continue")

        # Stopped again.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # Continue the program, we should exit.
        self.runCmd("process continue")

        # We should exit.
        self.expect(
            "process status",
            "Process exited successfully",
            patterns=["^Process [0-9]+ exited with status = 0"],
        )

    def breakpoint_options_language_test(self):
        """Test breakpoint command for language option."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 locations.
        bp_id = lldbutil.run_break_set_by_symbol(
            self,
            "ns::func",
            sym_exact=False,
            extra_options="-L c++",
            num_expected_locations=1,
        )

        location = self.target().FindBreakpointByID(bp_id).GetLocationAtIndex(0)
        function = location.GetAddress().GetFunction()
        self.expect(
            "breakpoint list -v",
            "Verbose breakpoint list contains mangled names",
            # The demangled function name is system-dependent, e.g.
            # 'int ns::func(void)' on Windows and 'ns::func()' on Linux.
            substrs=[
                f"function = {function.GetName()}",
                f"mangled function = {function.GetMangledName()}",
            ],
        )

        # This should create a breakpoint with 0 locations.
        lldbutil.run_break_set_by_symbol(
            self,
            "ns::func",
            sym_exact=False,
            extra_options="-L c",
            num_expected_locations=0,
        )
        self.runCmd("settings set target.language c")
        lldbutil.run_break_set_by_symbol(
            self, "ns::func", sym_exact=False, num_expected_locations=0
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # Continue the program, we should exit.
        self.runCmd("process continue")

        # We should exit.
        self.expect(
            "process status",
            "Process exited successfully",
            patterns=["^Process [0-9]+ exited with status = 0"],
        )
