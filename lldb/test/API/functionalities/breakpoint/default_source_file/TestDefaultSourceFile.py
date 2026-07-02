import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DefaultSourceFileTestCase(TestBase):
    def test_default_source_file_is_entry_point(self):
        """The default file for a line breakpoint is main's file, not Foo::main's."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        entry_line = line_number("main.cpp", "BREAK: entry point")

        # "break set -l" with no file picks the default source file: the one
        # containing the entry point. Foo::main in other.cpp shares the base
        # name "main" and must not be chosen instead.
        lldbutil.run_break_set_by_file_and_line(
            self, None, entry_line, num_expected_locations=1, loc_exact=True
        )

        bp = target.GetBreakpointAtIndex(0)
        loc = bp.GetLocationAtIndex(0)
        line_entry = loc.GetAddress().GetLineEntry()
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "main.cpp")
        self.assertEqual(line_entry.GetLine(), entry_line)
