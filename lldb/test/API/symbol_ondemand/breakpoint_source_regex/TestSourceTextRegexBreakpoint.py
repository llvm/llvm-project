"""
Test source text regex breakpoint hydrates module debug info
in symbol on-demand mode.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSourceTextRegexBreakpoint(TestBase):
    @skipIfWindows
    def test_with_run_command(self):
        self.build()

        # Load symbols on-demand
        self.runCmd("settings set symbols.load-on-demand true")

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp(self, "Set break point at this line.")
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        frame = self.frame()
        self.assertTrue(frame.IsValid())
        self.assertEqual(frame.GetLineEntry().GetFileSpec().GetFilename(), "main.cpp")
        self.assertEqual(frame.GetLineEntry().GetLine(), 4)
