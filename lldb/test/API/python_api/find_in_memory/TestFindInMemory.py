"""
Test Process::FindInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from address_ranges_helper import *


class FindInMemoryTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        live_pi = ProcessInfo()

        self.build()
        (
            live_pi.target,
            live_pi.process,
            live_pi.thread,
            live_pi.bp,
        ) = lldbutil.run_to_source_breakpoint(
            self,
            "break here",
            lldb.SBFileSpec("main.cpp"),
        )
        live_pi.frame = live_pi.thread.GetFrameAtIndex(0)
        self.assertTrue(live_pi.bp.IsValid())
        self.assertTrue(live_pi.process, PROCESS_IS_VALID)
        self.assertState(
            live_pi.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED
        )

        self.live_pi = live_pi

    def test_check_stack_pointer(self):
        """Make sure the 'stack_pointer' variable lives on the stack"""
        ex = self.live_pi.frame.EvaluateExpression("&stack_pointer")
        variable_region = lldb.SBMemoryRegionInfo()
        self.assertTrue(
            self.live_pi.process.GetMemoryRegionInfo(
                ex.GetValueAsUnsigned(), variable_region
            ).Success(),
        )

        stack_region = lldb.SBMemoryRegionInfo()
        self.assertTrue(
            self.live_pi.process.GetMemoryRegionInfo(
                self.live_pi.frame.GetSP(), stack_region
            ).Success(),
        )

        self.assertEqual(variable_region, stack_region)

    def test_find_in_memory_ok(self):
        """Make sure a match exists in the heap memory and the right address ranges are provided"""
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            GetStackRange(self, self.live_pi),
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_double_instance_ok(self):
        """Make sure a match exists in the heap memory and the right address ranges are provided"""
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self, self.live_pi)[0],
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            GetStackRange(self, self.live_pi),
            0,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_address_range(self):
        """Make sure invalid address range is failing"""
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            lldb.SBAddressRange(),
            1,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_invalid_buffer(self):
        """Make sure the empty buffer is failing"""
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            "",
            GetStackRange(self, self.live_pi),
            1,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)

    def test_find_in_memory_unaligned(self):
        """Make sure the unaligned match exists in the heap memory and is not found with alignment 8"""
        range = GetAlignedRange(self, self.live_pi)

        # First we make sure the pattern is found with alignment 1
        error = lldb.SBError()
        addr = self.live_pi.process.FindInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            range,
            1,
            error,
        )
        self.assertSuccess(error)
        self.assertNotEqual(addr, lldb.LLDB_INVALID_ADDRESS)

        # With alignment 8 the pattern should not be found
        addr = self.live_pi.process.FindInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            range,
            8,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(addr, lldb.LLDB_INVALID_ADDRESS)
