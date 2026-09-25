"""
Test utility functions for the frame object.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FrameUtilsTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.c", "// Find the line number here.")

    def test_frame_utils(self):
        """Test utility functions for the frame object."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.c"), self.line
        )

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0)
        frame1 = thread.GetFrameAtIndex(1)
        self.assertTrue(frame1)
        parent = lldbutil.get_parent_frame(frame0)
        self.assertTrue(parent and parent.GetFrameID() == frame1.GetFrameID())
        frame0_args = lldbutil.get_args_as_string(frame0)
        parent_args = lldbutil.get_args_as_string(parent)
        self.assertTrue(frame0_args and parent_args and "(int)val=1" in frame0_args)
        if self.TraceOn():
            lldbutil.print_stacktrace(thread)
            print("Current frame: %s" % frame0_args)
            print("Parent frame: %s" % parent_args)
