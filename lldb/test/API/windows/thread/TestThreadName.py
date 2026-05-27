"""
Test that lldb reports the Windows thread description set via SetThreadDescription.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestThreadName(TestBase):
    @skipUnlessWindows
    @skipIfWindows(windows_version=["<", "10.0.14393"])
    def test_with_thread_description(self):
        """SBThread.GetName() reflects SetThreadDescription on Windows."""
        self.build()
        source = lldb.SBFileSpec("main.c")

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", source
        )
        self.assertEqual(
            bkpt.GetNumLocations(), 2,
            "expected breakpoints at both '// break here' markers",
        )

        # No thread name yet.
        self.assertFalse(
            thread.GetName(),
            "thread should have no name before SetThreadDescription",
        )

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "no thread stopped at breakpoint")
        self.assertEqual(thread.GetName(), "ThreadName")