"""
Test _regexp-break command which uses regular expression matching to dispatch to other built in breakpoint commands.
"""


import os
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class RegexpBreakCommandTestCase(TestBase):
    def test_set_version(self):
        """Test _regexp-break command."""
        self.build()
        self.regexp_break_command("_regexp-break")

    def test_add_version(self):
        """Test _regexp-break-add command."""
        self.build()
        self.regexp_break_command("_regexp-break-add")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "main.c"
        self.line = line_number(self.source, "// Set break point at this line.")

    def regexp_break_command(self, cmd_name):
        """Test the super consie "b" command, which is analias for _regexp-break."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        break_results = lldbutil.run_break_set_command(self, f"{cmd_name} {self.line}")
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            file_name="main.c",
            line_number=self.line,
            num_locations=1,
        )

        break_results = lldbutil.run_break_set_command(
            self, f"{cmd_name} {self.source}:{self.line}"
        )
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            file_name="main.c",
            line_number=self.line,
            num_locations=1,
        )

        # Check breakpoint with full file path.
        full_path = os.path.join(self.getSourceDir(), self.source)
        break_results = lldbutil.run_break_set_command(
            self, f"{cmd_name} {full_path}:{self.line}"
        )
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            file_name="main.c",
            line_number=self.line,
            num_locations=1,
        )

        # Check breakpoint with symbol name.  I'm also passing in
        # the module so I can check the number of locations.
        exe_spec = lldb.SBFileSpec(exe)
        exe_filename = exe_spec.basename
        cmd = f"{cmd_name} {exe_filename}`main"
        print(f"About to run: '{cmd}'")
        break_results = lldbutil.run_break_set_command(self, cmd)
        lldbutil.check_breakpoint_result(
            self, break_results, symbol_name="main", num_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
