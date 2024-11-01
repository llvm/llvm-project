"""
Test lldb breakpoint setting by source regular expression.
This test just tests the source file & function restrictions.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSourceRegexBreakpoints(TestBase):
    def test_location(self):
        self.build()
        self.source_regex_locations()

    def test_restrictions(self):
        self.build()
        self.source_regex_restrictions()

    def source_regex_locations(self):
        """Test that restricting source expressions to files & to functions."""
        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # First look just in main:
        target_files = lldb.SBFileSpecList()
        target_files.Append(lldb.SBFileSpec("a.c"))

        func_names = lldb.SBStringList()
        func_names.AppendString("a_func")

        source_regex = "Set . breakpoint here"
        main_break = target.BreakpointCreateBySourceRegex(
            source_regex, lldb.SBFileSpecList(), target_files, func_names
        )
        num_locations = main_break.GetNumLocations()
        self.assertEqual(
            num_locations,
            1,
            "a.c in a_func should give one breakpoint, got %d." % (num_locations),
        )

        loc = main_break.GetLocationAtIndex(0)
        self.assertTrue(loc.IsValid(), "Got a valid location.")
        address = loc.GetAddress()
        self.assertTrue(address.IsValid(), "Got a valid address from the location.")

        a_func_line = line_number("a.c", "Set A breakpoint here")
        line_entry = address.GetLineEntry()
        self.assertTrue(line_entry.IsValid(), "Got a valid line entry.")
        self.assertEquals(
            line_entry.line,
            a_func_line,
            "Our line number matches the one lldbtest found.",
        )

    def source_regex_restrictions(self):
        """Test that restricting source expressions to files & to functions."""
        # Create a target by the debugger.
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # First look just in main:
        target_files = lldb.SBFileSpecList()
        target_files.Append(lldb.SBFileSpec("main.c"))
        source_regex = "Set . breakpoint here"
        main_break = target.BreakpointCreateBySourceRegex(
            source_regex, lldb.SBFileSpecList(), target_files, lldb.SBStringList()
        )

        num_locations = main_break.GetNumLocations()
        self.assertEqual(
            num_locations, 2, "main.c should have 2 matches, got %d." % (num_locations)
        )

        # Now look in both files:
        target_files.Append(lldb.SBFileSpec("a.c"))

        main_break = target.BreakpointCreateBySourceRegex(
            source_regex, lldb.SBFileSpecList(), target_files, lldb.SBStringList()
        )

        num_locations = main_break.GetNumLocations()
        self.assertEqual(
            num_locations,
            4,
            "main.c and a.c should have 4 matches, got %d." % (num_locations),
        )

        # Now restrict it to functions:
        func_names = lldb.SBStringList()
        func_names.AppendString("main_func")
        main_break = target.BreakpointCreateBySourceRegex(
            source_regex, lldb.SBFileSpecList(), target_files, func_names
        )

        num_locations = main_break.GetNumLocations()
        self.assertEqual(
            num_locations,
            2,
            "main_func in main.c and a.c should have 2 matches, got %d."
            % (num_locations),
        )
