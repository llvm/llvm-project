import lldb

SINGLE_INSTANCE_PATTERN_STACK = "stack_there_is_only_one_of_me"
DOUBLE_INSTANCE_PATTERN_HEAP = "heap_there_is_exactly_two_of_me"
ALIGNED_INSTANCE_PATTERN_HEAP = "i_am_unaligned_string_on_the_heap"
UNALIGNED_INSTANCE_PATTERN_HEAP = ALIGNED_INSTANCE_PATTERN_HEAP[1:]


def GetAlignedRange(test_base, shrink=False):
    frame = test_base.thread.GetSelectedFrame()
    ex = frame.EvaluateExpression("aligned_string_ptr")
    test_base.assertTrue(ex.IsValid())
    return GetRangeFromAddrValue(test_base, ex, shrink)


def GetStackRange(test_base, shrink=False):
    frame = test_base.thread.GetSelectedFrame()
    ex = frame.EvaluateExpression("&stack_pointer")
    test_base.assertTrue(ex.IsValid())
    return GetRangeFromAddrValue(test_base, ex, shrink)


def GetStackRanges(test_base, shrink=False):
    addr_ranges = lldb.SBAddressRangeList()
    addr_ranges.Append(GetStackRange(test_base))
    return addr_ranges


def GetRangeFromAddrValue(test_base, addr, shrink=False):
    """Returns a memory region containing 'addr'.
    If 'shrink' is True, the address range will be reduced to not exceed 2K.
    """
    region = lldb.SBMemoryRegionInfo()
    test_base.assertTrue(
        test_base.process.GetMemoryRegionInfo(
            addr.GetValueAsUnsigned(), region
        ).Success(),
    )

    test_base.assertTrue(region.IsReadable())
    test_base.assertFalse(region.IsExecutable())

    base = region.GetRegionBase()
    end = region.GetRegionEnd()

    if shrink:
        addr2 = addr.GetValueAsUnsigned()
        addr2 -= addr2 % 512
        base = max(base, addr2 - 1024)
        end = min(end, addr2 + 1024)

    start = lldb.SBAddress(base, test_base.target)
    size = end - base

    return lldb.SBAddressRange(start, size)


def GetHeapRanges(test_base, shrink=False):
    frame = test_base.thread.GetSelectedFrame()

    ex = frame.EvaluateExpression("heap_pointer1")
    test_base.assertTrue(ex.IsValid())
    range1 = GetRangeFromAddrValue(test_base, ex, shrink)
    range1_start = range1.GetBaseAddress().GetLoadAddress(test_base.target)
    range1_end = range1_start + range1.GetByteSize()

    ex = frame.EvaluateExpression("heap_pointer2")
    test_base.assertTrue(ex.IsValid())
    range2 = GetRangeFromAddrValue(test_base, ex, shrink)
    range2_start = range2.GetBaseAddress().GetLoadAddress(test_base.target)
    range2_end = range2_start + range2.GetByteSize()

    addr_ranges = lldb.SBAddressRangeList()

    if range1_end < range2_start or range2_end < range1_start:
        # The ranges do not overlap; add them both.
        addr_ranges.Append(range1)
        addr_ranges.Append(range2)
    else:
        # Merge overlapping ranges.
        base = min(range1_start, range2_start)
        end = max(range1_end, range2_end)
        start = lldb.SBAddress(base, test_base.target)
        size = end - base
        addr_ranges.Append(lldb.SBAddressRange(start, size))

    return addr_ranges


def GetRanges(test_base, shrink=False):
    ranges = GetHeapRanges(test_base, shrink)
    ranges.Append(GetStackRanges(test_base, shrink))

    return ranges
