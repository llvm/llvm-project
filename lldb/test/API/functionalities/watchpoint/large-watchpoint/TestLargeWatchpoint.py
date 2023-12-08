"""
Watch larger-than-8-bytes regions of memory, confirm that
writes to those regions are detected.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnalignedWatchpointTestCase(TestBase):
    def continue_and_report_stop_reason(self, process, iter_str):
        process.Continue()
        self.assertIn(
            process.GetState(), [lldb.eStateStopped, lldb.eStateExited], iter_str
        )
        thread = process.GetSelectedThread()
        return thread.GetStopReason()

    NO_DEBUG_INFO_TESTCASE = True

    # debugserver on AArch64 has this feature.
    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    @skipUnlessDarwin

    # debugserver only gained the ability to watch larger regions
    # with this patch.
    @skipIfOutOfTreeDebugserver
    def test_large_watchpoint(self):
        """Test watchpoint that covers a large region of memory."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        array_addr = frame.GetValueForVariablePath("array").GetValueAsUnsigned()

        # watch 256 uint32_t elements in the middle of the array,
        # don't assume that the heap allocated array is aligned
        # to a 1024 byte boundary to begin with, force alignment.
        wa_256_addr = (array_addr + 1024) & ~(1024 - 1)
        err = lldb.SBError()
        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)
        wp = target.WatchpointCreateByAddress(wa_256_addr, 1024, wp_opts, err)
        self.assertTrue(wp.IsValid())
        self.assertSuccess(err)

        c_count = 0
        reason = self.continue_and_report_stop_reason(process, "continue #%d" % c_count)
        while reason == lldb.eStopReasonWatchpoint:
            c_count = c_count + 1
            reason = self.continue_and_report_stop_reason(
                process, "continue #%d" % c_count
            )
            self.assertLessEqual(c_count, 16)

        self.assertEqual(c_count, 16)
