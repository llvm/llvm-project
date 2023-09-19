"""
Watch 4 bytes which spawn two doubleword aligned regions.
On a target that supports 8 byte watchpoints, this will
need to be implemented with a hardware watchpoint on both
doublewords.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UnalignedWatchpointTestCase(TestBase):
    def hit_watchpoint_and_continue(self, process, iter_str):
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateStopped, iter_str)
        thread = process.GetSelectedThread()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonWatchpoint, iter_str)
        self.assertEqual(thread.GetStopReasonDataCount(), 1, iter_str)
        wp_num = thread.GetStopReasonDataAtIndex(0)
        self.assertEqual(wp_num, 1, iter_str)

    NO_DEBUG_INFO_TESTCASE = True

    # debugserver on AArch64 has this feature.
    @skipIf(archs=no_match(["x86_64", "arm64", "arm64e", "aarch64"]))
    @skipUnlessDarwin
    # debugserver only started returning an exception address within
    # a range lldb expects in https://reviews.llvm.org/D147820 2023-04-12.
    # older debugservers will return the base address of the doubleword
    # which lldb doesn't understand, and will stop executing without a
    # proper stop reason.
    @skipIfOutOfTreeDebugserver
    def test_unaligned_watchpoint(self):
        """Test a watchpoint that is handled by two hardware watchpoint registers."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        a_bytebuf_6 = frame.GetValueForVariablePath("a.bytebuf[6]")
        a_bytebuf_6_addr = a_bytebuf_6.GetAddress().GetLoadAddress(target)
        err = lldb.SBError()
        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)
        wp = target.WatchpointCreateByAddress(a_bytebuf_6_addr, 4, wp_opts, err)
        self.assertTrue(err.Success())
        self.assertTrue(wp.IsEnabled())
        self.assertEqual(wp.GetWatchSize(), 4)
        self.assertGreater(
            wp.GetWatchAddress() % 8, 4, "watched region spans two doublewords"
        )

        # We will hit our watchpoint 6 times during the execution
        # of the inferior.  If the remote stub does not actually split
        # the watched region into two doubleword watchpoints, we will
        # exit before we get to 6 watchpoint hits.
        for i in range(1, 7):
            self.hit_watchpoint_and_continue(process, "wp hit number %s" % i)
