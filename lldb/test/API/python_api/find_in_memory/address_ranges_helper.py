import lldb

SINGLE_INSTANCE_PATTERN_STACK = "stack_there_is_only_one_of_me"
DOUBLE_INSTANCE_PATTERN_HEAP = "heap_there_is_exactly_two_of_me"
ALIGNED_INSTANCE_PATTERN_HEAP = "i_am_unaligned_string_on_the_heap"
UNALIGNED_INSTANCE_PATTERN_HEAP = ALIGNED_INSTANCE_PATTERN_HEAP[1:]


class ProcessInfo:
    def __init__(self):
        self.target = {}
        self.process = {}
        self.thread = {}
        self.frame = {}
        self.bp = {}


def GetAlignedRange(test_base, pi: ProcessInfo):
    frame = pi.thread.GetFrameAtIndex(0)
    var = frame.FindVariable("aligned_string_ptr")
    test_base.assertTrue(var.IsValid())
    return GetRangeFromAddrValue(test_base, pi, var)


def GetStackRange(test_base, pi: ProcessInfo):
    frame = pi.thread.GetFrameAtIndex(0)
    sp = frame.GetSP()
    region = lldb.SBMemoryRegionInfo()
    test_base.assertTrue(
        pi.process.GetMemoryRegionInfo(sp, region).Success(),
    )
    print(f"stack region: {region}")
    test_base.assertTrue(region.IsReadable())

    address_start = sp - pi.target.GetStackRedZoneSize()
    stack_size = region.GetRegionEnd() - address_start
    return lldb.SBAddressRange(lldb.SBAddress(address_start, pi.target), stack_size)


def GetStackRanges(test_base, pi: ProcessInfo):
    addr_ranges = lldb.SBAddressRangeList()
    addr_ranges.Append(GetStackRange(test_base, pi))
    return addr_ranges


def GetRangeFromAddrValue(test_base, pi: ProcessInfo, var):
    region = lldb.SBMemoryRegionInfo()
    addr = lldb.SBAddress(var.GetValueAsUnsigned(), pi.target)
    addr = addr.GetLoadAddress(pi.target)
    test_base.assertTrue(
        pi.process.GetMemoryRegionInfo(var.GetValueAsUnsigned(), region)
    )
    address_start = lldb.SBAddress(region.GetRegionBase(), pi.target)
    size = region.GetRegionEnd() - region.GetRegionBase()
    test_base.assertTrue(region.IsReadable(), f"Invalid region {region} for {addr}")
    return lldb.SBAddressRange(address_start, size)


def IsWithinRange(addr, range, target):
    start_addr = range.GetBaseAddress().GetLoadAddress(target)
    if start_addr == lldb.LLDB_INVALID_ADDRESS:
        return False
    end_addr = start_addr + range.GetByteSize()
    addr = addr.GetValueAsUnsigned()
    return addr >= start_addr and addr < end_addr


def GetHeapRanges(test_base, pi: ProcessInfo):
    var = pi.frame.FindVariable("heap_pointer1")
    test_base.assertTrue(var.IsValid())

    range = GetRangeFromAddrValue(test_base, pi, var)
    addr_ranges = lldb.SBAddressRangeList()
    test_base.assertTrue(range.IsValid(), f"Invalid range {range} for {var}")
    addr_ranges.Append(range)
    var = pi.frame.FindVariable("heap_pointer2")
    test_base.assertTrue(var.IsValid())

    if not IsWithinRange(var, addr_ranges[0], pi.target):
        # If the second heap pointer is not within the first heap pointer's range,
        # then we need to add the second heap pointer's range to the list.
        addr_ranges.Append(GetRangeFromAddrValue(test_base, var))

    return addr_ranges


def GetRanges(test_base, pi: ProcessInfo):
    ranges = GetHeapRanges(test_base, pi)
    ranges.Append(GetStackRanges(test_base, pi))
    return ranges
