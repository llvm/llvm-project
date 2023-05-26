"""Test breaking inside functions defined within a BSD archive file libfoo.a."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BSDArchivesTestCase(TestBase):
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr24527.  Makefile.rules doesn't know how to build static libs on Windows",
    )
    def test(self):
        """Break inside a() and b() defined within libfoo.a."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on a() and b() symbols
        lldbutil.run_break_set_by_symbol(self, "a", sym_exact=True)
        lldbutil.run_break_set_by_symbol(self, "b", sym_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Break at a(int) first.
        self.expect(
            "frame variable", VARIABLES_DISPLAYED_CORRECTLY, substrs=["(int) arg = 1"]
        )
        self.expect(
            "frame variable __a_global",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(int) __a_global = 1"],
        )

        # Continue the program, we should break at b(int) next.
        self.runCmd("continue")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )
        self.expect(
            "frame variable", VARIABLES_DISPLAYED_CORRECTLY, substrs=["(int) arg = 2"]
        )
        self.expect(
            "frame variable __b_global",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=["(int) __b_global = 2"],
        )
