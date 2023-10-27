"""
Test change libc++ std::atomic values.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxChangeValueTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @add_test_categories(["libc++"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24772")
    def test(self):
        """Test that we can change values of libc++ std::atomic."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."
            )
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # Get Frame #0.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition",
        )
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Got a valid frame.")

        q_value = frame0.FindVariable("Q")
        self.assertTrue(q_value.IsValid(), "Got the SBValue for val")
        inner_val = q_value.GetChildAtIndex(0)
        self.assertTrue(inner_val.IsValid(), "Got the SBValue for inner atomic val")
        result = inner_val.SetValueFromCString("42")
        self.assertTrue(result, "Setting val returned True.")
        result = inner_val.GetValueAsUnsigned()
        self.assertTrue(result == 42, "Got correct value (42)")
