"""
Test NetBSD core file debugging.
"""

from __future__ import division, print_function

import signal
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NetBSDCoreCommonTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_memory_regions(self, process, region_count):
        region_list = process.GetMemoryRegions()
        self.assertEqual(region_list.GetSize(), region_count)

        region = lldb.SBMemoryRegionInfo()

        # Check we have the right number of regions.
        self.assertEqual(region_list.GetSize(), region_count)

        # Check that getting a region beyond the last in the list fails.
        self.assertFalse(
            region_list.GetMemoryRegionAtIndex(
                region_count, region))

        # Check each region is valid.
        for i in range(region_list.GetSize()):
            # Check we can actually get this region.
            self.assertTrue(region_list.GetMemoryRegionAtIndex(i, region))

            # Every region in the list should be mapped.
            self.assertTrue(region.IsMapped())

            # Test the address at the start of a region returns it's enclosing
            # region.
            begin_address = region.GetRegionBase()
            region_at_begin = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(begin_address, region_at_begin)
            self.assertEqual(region, region_at_begin)

            # Test an address in the middle of a region returns it's enclosing
            # region.
            middle_address = (region.GetRegionBase() +
                              region.GetRegionEnd()) // 2
            region_at_middle = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                middle_address, region_at_middle)
            self.assertEqual(region, region_at_middle)

            # Test the address at the end of a region returns it's enclosing
            # region.
            end_address = region.GetRegionEnd() - 1
            region_at_end = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(end_address, region_at_end)
            self.assertEqual(region, region_at_end)

            # Check that quering the end address does not return this region but
            # the next one.
            next_region = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                region.GetRegionEnd(), next_region)
            self.assertNotEqual(region, next_region)
            self.assertEqual(
                region.GetRegionEnd(),
                next_region.GetRegionBase())

        # Check that query beyond the last region returns an unmapped region
        # that ends at LLDB_INVALID_ADDRESS
        last_region = lldb.SBMemoryRegionInfo()
        region_list.GetMemoryRegionAtIndex(region_count - 1, last_region)
        end_region = lldb.SBMemoryRegionInfo()
        error = process.GetMemoryRegionInfo(
            last_region.GetRegionEnd(), end_region)
        self.assertFalse(end_region.IsMapped())
        self.assertEqual(
            last_region.GetRegionEnd(),
            end_region.GetRegionBase())
        self.assertEqual(end_region.GetRegionEnd(), lldb.LLDB_INVALID_ADDRESS)

    def check_state(self, process):
        with open(os.devnull) as devnul:
            # sanitize test output
            self.dbg.SetOutputFileHandle(devnul, False)
            self.dbg.SetErrorFileHandle(devnul, False)

            self.assertTrue(process.is_stopped)

            # Process.Continue
            error = process.Continue()
            self.assertFalse(error.Success())
            self.assertTrue(process.is_stopped)

            # Thread.StepOut
            thread = process.GetSelectedThread()
            thread.StepOut()
            self.assertTrue(process.is_stopped)

            # command line
            self.dbg.HandleCommand('s')
            self.assertTrue(process.is_stopped)
            self.dbg.HandleCommand('c')
            self.assertTrue(process.is_stopped)

            # restore file handles
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)

    def check_backtrace(self, thread, filename, backtrace):
        self.assertGreaterEqual(thread.GetNumFrames(), len(backtrace))
        src = filename.rpartition('.')[0] + '.c'
        for i in range(len(backtrace)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame)
            if not backtrace[i].startswith('_'):
                self.assertEqual(frame.GetFunctionName(), backtrace[i])
                self.assertEqual(frame.GetLineEntry().GetLine(),
                                 line_number(src, "Frame " + backtrace[i]))
                self.assertEqual(
                    frame.FindVariable("F").GetValueAsUnsigned(), ord(
                        backtrace[i][0]))

    def do_test(self, filename, pid, region_count):
        target = self.dbg.CreateTarget(filename)
        process = target.LoadCore(filename + ".core")

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), self.THREAD_COUNT)
        self.assertEqual(process.GetProcessID(), pid)

        self.check_state(process)

        self.check_stack(process, pid, filename)

        self.check_memory_regions(process, region_count)

        self.dbg.DeleteTarget(target)


class NetBSD1LWPCoreTestCase(NetBSDCoreCommonTestCase):
    THREAD_COUNT = 1

    def check_stack(self, process, pid, filename):
        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), 1)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataCount(), 1)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signal.SIGSEGV)
        backtrace = ["bar", "foo", "main"]
        self.check_backtrace(thread, filename, backtrace)

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64(self):
        """Test single-threaded aarch64 core dump."""
        self.do_test("1lwp_SIGSEGV.aarch64", pid=8339, region_count=32)

    @skipIfLLVMTargetMissing("X86")
    def test_amd64(self):
        """Test single-threaded amd64 core dump."""
        self.do_test("1lwp_SIGSEGV.amd64", pid=693, region_count=21)


class NetBSD2LWPT2CoreTestCase(NetBSDCoreCommonTestCase):
    THREAD_COUNT = 2

    def check_stack(self, process, pid, filename):
        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), 2)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataCount(), 1)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signal.SIGSEGV)
        backtrace = ["bar", "foo", "lwp_main"]
        self.check_backtrace(thread, filename, backtrace)

        # thread 1 should have no signal
        thread = process.GetThreadByID(1)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataCount(), 1)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), 0)

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64(self):
        """Test double-threaded aarch64 core dump where thread 2 is signalled."""
        self.do_test("2lwp_t2_SIGSEGV.aarch64", pid=14142, region_count=31)

    @skipIfLLVMTargetMissing("X86")
    def test_amd64(self):
        """Test double-threaded amd64 core dump where thread 2 is signalled."""
        self.do_test("2lwp_t2_SIGSEGV.amd64", pid=622, region_count=24)


class NetBSD2LWPProcessSigCoreTestCase(NetBSDCoreCommonTestCase):
    THREAD_COUNT = 2

    def check_stack(self, process, pid, filename):
        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), 2)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataCount(), 1)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signal.SIGSEGV)
        backtrace = ["bar", "foo", "lwp_main"]
        self.check_backtrace(thread, filename, backtrace)

        # thread 1 should have the same signal
        thread = process.GetThreadByID(1)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        self.assertEqual(thread.GetStopReasonDataCount(), 1)
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signal.SIGSEGV)

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64(self):
        """Test double-threaded aarch64 core dump where process is signalled."""
        self.do_test("2lwp_process_SIGSEGV.aarch64", pid=1403, region_count=30)

    @skipIfLLVMTargetMissing("X86")
    def test_amd64(self):
        """Test double-threaded amd64 core dump where process is signalled."""
        self.do_test("2lwp_process_SIGSEGV.amd64", pid=665, region_count=24)
