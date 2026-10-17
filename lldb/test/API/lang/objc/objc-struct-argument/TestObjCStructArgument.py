"""Test passing structs to Objective-C methods."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCStructArgument(TestBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "test.m"
        self.break_line = line_number(self.main_source, "// Set breakpoint here.")

    # this test program only builds for ios with -gmodules
    @add_test_categories(["gmodules", "pyapi"])
    @skipIf(oslist=["ios", "watchos", "tvos", "bridgeos"], archs=["armv7", "arm64"])
    def test_with_python_api(self):
        """Test passing structs to Objective-C methods."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.main_source), self.break_line
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        self.expect("expression [summer sumThings:tts]", substrs=["9"])

        self.expect(
            "po [NSValue valueWithRect:rect]", substrs=["NSRect: {{0, 0}, {10, 20}}"]
        )

        # Now make sure we can call a method that returns a struct without
        # crashing.
        cmd_value = frame.EvaluateExpression("[provider getRange]")
        self.assertTrue(cmd_value.IsValid())
