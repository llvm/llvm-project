"""
Test the "process continue --reverse" and "--forward" options.
"""


import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbreverse import ReverseTestBase
from lldbsuite.test import lldbutil


class TestReverseContinue(ReverseTestBase):
    @skipIfRemote
    def test_reverse_continue(self):
        target, _, _ = self.setup_recording()

        # Set breakpoint and reverse-continue
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        self.assertTrue(trigger_bkpt.GetNumLocations() > 0)
        self.expect(
            "process continue --reverse",
            substrs=["stop reason = breakpoint {0}.1".format(trigger_bkpt.GetID())],
        )
        # `process continue` should preserve current base direction.
        self.expect(
            "process continue",
            STOPPED_DUE_TO_HISTORY_BOUNDARY,
            substrs=["stopped", "stop reason = history boundary"],
        )
        self.expect(
            "process continue --forward",
            substrs=["stop reason = breakpoint {0}.1".format(trigger_bkpt.GetID())],
        )

    def setup_recording(self):
        """
        Record execution of code between "start_recording" and "stop_recording" breakpoints.

        Returns with the target stopped at "stop_recording", with recording disabled,
        ready to reverse-execute.
        """
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        process = self.connect(target)

        # Record execution from the start of the function "start_recording"
        # to the start of the function "stop_recording". We want to keep the
        # interval that we record as small as possible to minimize the run-time
        # of our single-stepping recorder.
        start_recording_bkpt = target.BreakpointCreateByName("start_recording", None)
        self.assertTrue(start_recording_bkpt.GetNumLocations() > 0)
        initial_threads = lldbutil.continue_to_breakpoint(process, start_recording_bkpt)
        self.assertEqual(len(initial_threads), 1)
        target.BreakpointDelete(start_recording_bkpt.GetID())
        self.start_recording()
        stop_recording_bkpt = target.BreakpointCreateByName("stop_recording", None)
        self.assertTrue(stop_recording_bkpt.GetNumLocations() > 0)
        lldbutil.continue_to_breakpoint(process, stop_recording_bkpt)
        target.BreakpointDelete(stop_recording_bkpt.GetID())
        self.stop_recording()

        self.dbg.SetAsync(False)

        return target, process, initial_threads
