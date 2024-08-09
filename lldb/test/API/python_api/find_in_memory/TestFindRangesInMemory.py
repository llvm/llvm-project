"""
Test Process::FindRangesInMemory.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from address_ranges_helper import *


class FindRangesInMemoryTestCase(TestBase):
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

    def test_find_ranges_in_memory_two_matches(self):
        """Make sure two matches exist in the heap memory and the right address ranges are provided"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self, self.live_pi),
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 2)

    def test_find_ranges_in_memory_one_match(self):
        """Make sure exactly one match exists in the heap memory and the right address ranges are provided"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            GetStackRanges(self, self.live_pi),
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_one_match_multiple_ranges(self):
        """Make sure exactly one match exists in the heap memory and multiple address ranges are provided"""
        addr_ranges = GetRanges(self, self.live_pi)
        addr_ranges.Append(lldb.SBAddressRange())
        self.assertGreater(addr_ranges.GetSize(), 2)
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_one_match_max(self):
        """Make sure at least one matche exists in the heap memory and the right address ranges are provided"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self, self.live_pi),
            1,
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self, self.live_pi),
            0,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_range(self):
        """Make sure the alignment 0 is failing"""
        addr_ranges = lldb.SBAddressRangeList()
        addr_ranges.Append(lldb.SBAddressRange())
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertIn("unable to resolve any ranges", str(error))
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_empty_ranges(self):
        """Make sure the empty ranges is failing"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            lldb.SBAddressRangeList(),
            1,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_buffer(self):
        """Make sure the empty buffer is failing"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            "",
            GetHeapRanges(self, self.live_pi),
            1,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_max_matches(self):
        """Make sure the empty buffer is failing"""
        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            GetHeapRanges(self, self.live_pi),
            1,
            0,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_in_memory_unaligned(self):
        """Make sure the unaligned match exists in the heap memory and is not found with alignment 8"""
        addr_ranges = lldb.SBAddressRangeList()
        addr_ranges.Append(GetAlignedRange(self, self.live_pi))

        error = lldb.SBError()
        matches = self.live_pi.process.FindRangesInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

        matches = self.live_pi.process.FindRangesInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            8,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 0)
