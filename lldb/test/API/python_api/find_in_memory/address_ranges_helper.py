import lldb

SINGLE_INSTANCE_PATTERN_STACK = "stack_there_is_only_one_of_me"
DOUBLE_INSTANCE_PATTERN_HEAP = "heap_there_is_exactly_two_of_me"
ALIGNED_INSTANCE_PATTERN_HEAP = "i_am_unaligned_string_on_the_heap"
UNALIGNED_INSTANCE_PATTERN_HEAP = ALIGNED_INSTANCE_PATTERN_HEAP[1:]


def GetAlignedRange(test_base):
    frame = test_base.thread.GetSelectedFrame()
    ex = frame.EvaluateExpression("aligned_string_ptr")
    test_base.assertTrue(ex.IsValid())
    return GetRangeFromAddrValue(test_base, ex)


def GetStackRange(test_base):
    frame = test_base.thread.GetSelectedFrame()
    ex = frame.EvaluateExpression("&stack_pointer")
    test_base.assertTrue(ex.IsValid())
    return GetRangeFromAddrValue(test_base, ex)


def GetStackRanges(test_base):
    addr_ranges = lldb.SBAddressRangeList()
    addr_ranges.Append(GetStackRange(test_base))
    return addr_ranges


def GetRangeFromAddrValue(test_base, addr):
    region = lldb.SBMemoryRegionInfo()
    test_base.assertTrue(
        test_base.process.GetMemoryRegionInfo(
            addr.GetValueAsUnsigned(), region
        ).Success(),
    )

    test_base.assertTrue(region.IsReadable())
    test_base.assertFalse(region.IsExecutable())

    address_start = lldb.SBAddress(region.GetRegionBase(), test_base.target)
    stack_size = region.GetRegionEnd() - region.GetRegionBase()
    return lldb.SBAddressRange(address_start, stack_size)


def IsWithinRange(addr, range, target):
    start_addr = range.GetBaseAddress().GetLoadAddress(target)
    end_addr = start_addr + range.GetByteSize()
    addr = addr.GetValueAsUnsigned()
    return addr >= start_addr and addr < end_addr


def GetHeapRanges(test_base):
    frame = test_base.thread.GetSelectedFrame()

    ex = frame.EvaluateExpression("heap_pointer1")
    test_base.assertTrue(ex.IsValid())
    range = GetRangeFromAddrValue(test_base, ex)
    addr_ranges = lldb.SBAddressRangeList()
    addr_ranges.Append(range)

    ex = frame.EvaluateExpression("heap_pointer2")
    test_base.assertTrue(ex.IsValid())
    if not IsWithinRange(ex, addr_ranges[0], test_base.target):
        addr_ranges.Append(GetRangeFromAddrValue(test_base, ex))

    return addr_ranges


def GetRanges(test_base):
    ranges = GetHeapRanges(test_base)
    ranges.Append(GetStackRanges(test_base))

    return ranges
