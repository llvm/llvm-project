"""
Test basics of a mini dump taken of a 32-bit process running in WoW64

WoW64 is the subsystem that lets 32-bit processes run in 64-bit Windows.  If you
capture a mini dump of a process running under WoW64 with a 64-bit debugger, you
end up with a dump of the WoW64 layer.  In that case, LLDB must do extra work to
get the 32-bit register contexts.
"""

from six import iteritems


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Wow64MiniDumpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_wow64_mini_dump(self):
        """Test that lldb can read the process information from the minidump."""
        # target create -c fizzbuzz_wow64.dmp
        target = self.dbg.CreateTarget("")
        process = target.LoadCore("fizzbuzz_wow64.dmp")
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), 0x1E9C)

    def test_thread_info_in_wow64_mini_dump(self):
        """Test that lldb can read the thread information from the minidump."""
        # target create -c fizzbuzz_wow64.dmp
        target = self.dbg.CreateTarget("")
        process = target.LoadCore("fizzbuzz_wow64.dmp")
        # This process crashed due to an access violation (0xc0000005), but the
        # minidump doesn't have an exception record--perhaps the crash handler
        # ate it.
        # TODO:  See if we can recover the exception information from the TEB,
        # which, according to Windbg, has a pointer to an exception list.

        # In the dump, none of the threads are stopped, so we cannot use
        # lldbutil.get_stopped_thread.
        thread = process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)

    def test_stack_info_in_wow64_mini_dump(self):
        """Test that we can see a trivial stack in a VS-generate mini dump."""
        # target create -c fizzbuzz_no_heap.dmp
        target = self.dbg.CreateTarget("")
        process = target.LoadCore("fizzbuzz_wow64.dmp")
        self.assertGreaterEqual(process.GetNumThreads(), 1)
        # This process crashed due to an access violation (0xc0000005), but the
        # minidump doesn't have an exception record--perhaps the crash handler
        # ate it.
        # TODO:  See if we can recover the exception information from the TEB,
        # which, according to Windbg, has a pointer to an exception list.

        # In the dump, none of the threads are stopped, so we cannot use
        # lldbutil.get_stopped_thread.
        thread = process.GetThreadAtIndex(0)
        # The crash is in main, so there should be at least one frame on the
        # stack.
        self.assertGreaterEqual(thread.GetNumFrames(), 1)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        pc = frame.GetPC()
        eip = frame.FindRegister("pc")
        self.assertTrue(eip.IsValid())
        self.assertEqual(pc, eip.GetValueAsUnsigned())
