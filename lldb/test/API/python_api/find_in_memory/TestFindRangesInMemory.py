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

        self.build()
        (
            self.target,
            self.process,
            self.thread,
            self.bp,
        ) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(self.bp.IsValid())

    def test_find_ranges_in_memory_two_matches(self):
        """Make sure two matches exist in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetHeapRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 2)

    def test_find_ranges_in_memory_one_match(self):
        """Make sure exactly one match exists in the heap memory and the right address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetStackRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            SINGLE_INSTANCE_PATTERN_STACK,
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_one_match_multiple_ranges(self):
        """Make sure exactly one match exists in the heap memory and multiple address ranges are provided"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetRanges(self)
        addr_ranges.Append(lldb.SBAddressRange())
        self.assertGreater(addr_ranges.GetSize(), 2)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
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
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetHeapRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            1,
            error,
        )

        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

    def test_find_ranges_in_memory_invalid_alignment(self):
        """Make sure the alignment 0 is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetHeapRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            0,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_range(self):
        """Make sure the alignment 0 is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = lldb.SBAddressRangeList()
        addr_ranges.Append(lldb.SBAddressRange())
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
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
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = lldb.SBAddressRangeList()
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_buffer(self):
        """Make sure the empty buffer is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetHeapRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            "",
            addr_ranges,
            1,
            10,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_ranges_in_memory_invalid_max_matches(self):
        """Make sure the empty buffer is failing"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = GetHeapRanges(self)
        error = lldb.SBError()
        matches = self.process.FindRangesInMemory(
            DOUBLE_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            0,
            error,
        )

        self.assertFailure(error)
        self.assertEqual(matches.GetSize(), 0)

    def test_find_in_memory_unaligned(self):
        """Make sure the unaligned match exists in the heap memory and is not found with alignment 8"""
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertState(self.process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        addr_ranges = lldb.SBAddressRangeList()
        addr_ranges.Append(GetAlignedRange(self))
        error = lldb.SBError()

        matches = self.process.FindRangesInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            1,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 1)

        matches = self.process.FindRangesInMemory(
            UNALIGNED_INSTANCE_PATTERN_HEAP,
            addr_ranges,
            8,
            10,
            error,
        )
        self.assertSuccess(error)
        self.assertEqual(matches.GetSize(), 0)
