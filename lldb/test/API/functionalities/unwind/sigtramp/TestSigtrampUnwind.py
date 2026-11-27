"""
Test that we can backtrace correctly with 'sigtramp' functions on the stack
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SigtrampUnwind(TestBase):
    # On different platforms the "_sigtramp" and "__kill" frames are likely to be different.
    # This test could probably be adapted to run on linux/*bsd easily enough.
    @skipUnlessDarwin
    @expectedFailureAll(
        archs=["arm64"], bugnumber="<rdar://problem/34006863>"
    )  # lldb skips 1 frame on arm64 above _sigtramp
    def test(self):
        """Test that we can backtrace correctly with _sigtramp on the stack"""
        self.build()
        self.setTearDownCleanup()

        _, process, _, _ = lldbutil.run_to_line_breakpoint(
            self,
            lldb.SBFileSpec("main.c"),
            line_number("main.c", "// Set breakpoint here"),
        )

        self.expect(
            "proc handle  -n false -p true -s false SIGUSR1",
            "Have lldb pass SIGUSR1 signals",
            substrs=["SIGUSR1", "true", "false", "false"],
        )

        lldbutil.run_break_set_by_symbol(
            self, "handler", num_expected_locations=1, module_name="a.out"
        )

        self.runCmd("continue")

        thread = process.GetThreadAtIndex(0)

        found_handler = False
        found_sigtramp = False
        found_kill = False
        found_main = False

        for f in thread.frames:
            if f.GetFunctionName() == "handler":
                found_handler = True
            if f.GetFunctionName() == "_sigtramp":
                found_sigtramp = True
            if f.GetFunctionName() == "__kill":
                found_kill = True
            if f.GetFunctionName() == "main":
                found_main = True

        if self.TraceOn():
            print("Backtrace once we're stopped:")
            for f in thread.frames:
                print("  %d %s" % (f.GetFrameID(), f.GetFunctionName()))

        if not found_handler:
            self.fail("Unable to find handler() in backtrace.")

        if not found_sigtramp:
            self.fail("Unable to find _sigtramp() in backtrace.")

        if not found_kill:
            self.fail("Unable to find kill() in backtrace.")

        if not found_main:
            self.fail("Unable to find main() in backtrace.")
