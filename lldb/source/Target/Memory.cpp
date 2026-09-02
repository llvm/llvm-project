//===-- Memory.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Memory.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/Utility/State.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cinttypes>
#include <memory>
#include <utility>

using namespace lldb;
using namespace lldb_private;

llvm::ArrayRef<uint8_t> LineCache::Lookup(addr_t addr) const {
  const auto pos = m_lines.find(IndexOf(addr));
  if (pos == m_lines.end())
    return {};
  const addr_t line_offset = addr % m_line_byte_size;
  return llvm::ArrayRef(pos->second.get(), m_line_byte_size)
      .drop_front(line_offset);
}

void LineCache::Insert(addr_t addr, llvm::ArrayRef<uint8_t> src) {
  assert((addr % m_line_byte_size) == 0 &&
         "whole line inserted at an unaligned address!");
  assert(src.size() == m_line_byte_size &&
         "whole line inserted with a partial buffer!");
  auto line = std::make_unique<uint8_t[]>(m_line_byte_size);
  std::copy(src.begin(), src.end(), line.get());
  m_lines[IndexOf(addr)] = std::move(line);
}

void LineCache::EraseRange(addr_t addr, addr_t size) {
  if (size == 0)
    return;
  // Clamp a range running past the end of the address space to it.
  const addr_t end_addr = llvm::SaturatingAdd(addr, size - 1);
  const uint64_t first_idx = IndexOf(addr);
  const uint64_t last_idx = IndexOf(end_addr);
  m_lines.remove_if([first_idx, last_idx](const auto &entry) {
    return entry.getFirst() >= first_idx && entry.getFirst() <= last_idx;
  });
}

ChunkCache::Collection::const_iterator
ChunkCache::FindChunkContaining(addr_t addr) const {
  if (m_chunks.empty())
    return m_chunks.end();
  Collection::const_iterator pos = m_chunks.upper_bound(addr);
  if (pos == m_chunks.begin())
    return m_chunks.end();
  --pos;
  // Sum pos->first + size wraps at the top of the address space.
  return addr - pos->first < pos->second.size() ? pos : m_chunks.end();
}

llvm::ArrayRef<uint8_t> ChunkCache::Lookup(addr_t addr) const {
  const Collection::const_iterator pos = FindChunkContaining(addr);
  if (pos == m_chunks.end())
    return {};
  return llvm::ArrayRef(pos->second).drop_front(addr - pos->first);
}

void ChunkCache::InsertMissing(addr_t addr, llvm::ArrayRef<uint8_t> src) {
  if (src.empty())
    return;
  // The last addressable byte of the range, clamped if it runs past the end of
  // the address space.
  const addr_t last_addr = llvm::SaturatingAdd<addr_t>(addr, src.size() - 1);
  const uint64_t len = last_addr - addr + 1;

  for (uint64_t offset = 0; offset < len;) {
    const addr_t curr_addr = addr + offset;
    if (const llvm::ArrayRef<uint8_t> held = Lookup(curr_addr); !held.empty()) {
      offset += std::min<uint64_t>(held.size(), len - offset);
      continue;
    }
    // Nothing holds curr_addr, so the gap runs to the next chunk or to the end.
    const Collection::const_iterator next = m_chunks.lower_bound(curr_addr);
    const uint64_t gap_len =
        next == m_chunks.end()
            ? len - offset
            : std::min<uint64_t>(next->first - curr_addr, len - offset);
    const llvm::ArrayRef<uint8_t> gap_bytes = src.slice(offset, gap_len);
    m_chunks[curr_addr].assign(gap_bytes.begin(), gap_bytes.end());
    offset += gap_len;
  }
}

void ChunkCache::EraseRange(addr_t addr, addr_t size) {
  if (size == 0)
    return;
  // Clamp a range running past the end of the address space to it.
  const addr_t end_addr = llvm::SaturatingAdd(addr, size - 1);

  Collection::iterator pos = m_chunks.lower_bound(addr);
  // A chunk starting below addr can still reach into the range.
  if (pos != m_chunks.begin()) {
    const Collection::iterator prev = std::prev(pos);
    if (addr - prev->first < prev->second.size())
      m_chunks.erase(prev);
  }
  while (pos != m_chunks.end() && pos->first <= end_addr)
    pos = m_chunks.erase(pos);
}

// MemoryCache constructor
MemoryCache::MemoryCache(Process &process)
    : m_mutex(), m_L1_cache(), m_L2_cache(process.GetMemoryCacheLineSize()),
      m_invalid_ranges(), m_process(process) {}

// Destructor
MemoryCache::~MemoryCache() = default;

void MemoryCache::Clear(bool clear_invalid_ranges) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_L1_cache.Clear();
  m_L2_cache.Clear(m_process.GetMemoryCacheLineSize());
  if (clear_invalid_ranges)
    m_invalid_ranges.Clear();
}

void MemoryCache::AddCacheData(lldb::addr_t addr, const void *src,
                               size_t src_len) {
  InsertData(addr, {static_cast<const uint8_t *>(src), src_len});
}

void MemoryCache::InsertWholeLine(addr_t line_base_addr,
                                  llvm::ArrayRef<uint8_t> src) {
  m_L2_cache.Insert(line_base_addr, src);
  // The new line holds every byte the L1 entries inside it held.
  m_L1_cache.EraseRange(line_base_addr, src.size());
}

void MemoryCache::InsertPartialLine(addr_t addr, llvm::ArrayRef<uint8_t> src) {
  const uint32_t line_size = m_L2_cache.GetLineByteSize();
  assert(src.size() <= line_size &&
         addr / line_size == (addr + src.size() - 1) / line_size &&
         "a partial-line insert must not cross a cache line boundary");
  // L2 holds only whole lines, so a range inside a resident line is held
  // already.
  if (m_L2_cache.Holds(addr))
    return;
  m_L1_cache.InsertMissing(addr, src);
}

void MemoryCache::InsertData(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src) {
  if (src.empty())
    return;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  // The last addressable byte of the range, clamped if it runs past the end of
  // the address space, so no offset added to addr can wrap to 0.
  const addr_t last_addr = llvm::SaturatingAdd<addr_t>(addr, src.size() - 1);
  const uint64_t len = last_addr - addr + 1;
  const uint32_t line_size = m_L2_cache.GetLineByteSize();

  for (uint64_t offset = 0; offset < len;) {
    const addr_t curr_addr = addr + offset;
    const uint64_t line_offset = curr_addr % line_size;
    const uint64_t piece_len =
        std::min<uint64_t>(line_size - line_offset, len - offset);
    const llvm::ArrayRef<uint8_t> piece_bytes = src.slice(offset, piece_len);
    if (line_offset == 0 && piece_len == line_size)
      InsertWholeLine(curr_addr, piece_bytes);
    else
      InsertPartialLine(curr_addr, piece_bytes);
    offset += piece_len;
  }
}

void MemoryCache::AddCacheData(lldb::addr_t addr,
                               const DataBufferSP &data_buffer_sp) {
  InsertData(addr, {data_buffer_sp->GetBytes(), data_buffer_sp->GetByteSize()});
}

void MemoryCache::Flush(addr_t addr, size_t size) {
  if (size == 0)
    return;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  m_L1_cache.EraseRange(addr, size);
  m_L2_cache.EraseRange(addr, size);
}

void MemoryCache::AddInvalidRange(lldb::addr_t base_addr,
                                  lldb::addr_t byte_size) {
  if (byte_size > 0) {
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    InvalidRanges::Entry range(base_addr, byte_size);
    m_invalid_ranges.Append(range);
    m_invalid_ranges.Sort();
  }
}

bool MemoryCache::RemoveInvalidRange(lldb::addr_t base_addr,
                                     lldb::addr_t byte_size) {
  if (byte_size > 0) {
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    const uint32_t idx = m_invalid_ranges.FindEntryIndexThatContains(base_addr);
    if (idx != UINT32_MAX) {
      const InvalidRanges::Entry *entry = m_invalid_ranges.GetEntryAtIndex(idx);
      if (entry->GetRangeBase() == base_addr &&
          entry->GetByteSize() == byte_size)
        return m_invalid_ranges.RemoveEntryAtIndex(idx);
    }
  }
  return false;
}

size_t MemoryCache::ReadFromCaches(lldb::addr_t addr, void *dst,
                                   size_t len) const {
  size_t bytes_filled = 0;
  // Bytes from addr to the last addressable byte.  The walk must not pass
  // it, or curr_addr wraps to 0.
  const uint64_t space_to_top = UINT64_MAX - addr;
  while (bytes_filled < len) {
    if (bytes_filled > space_to_top)
      break;
    const addr_t curr_addr = addr + bytes_filled;

    // At most one of the caches can hold curr_addr.
    llvm::ArrayRef<uint8_t> cached = m_L2_cache.Lookup(curr_addr);
    if (cached.empty())
      cached = m_L1_cache.Lookup(curr_addr);
    if (cached.empty())
      break;

    const size_t to_copy = std::min(cached.size(), len - bytes_filled);
    memcpy(static_cast<uint8_t *>(dst) + bytes_filled, cached.data(), to_copy);
    bytes_filled += to_copy;
  }
  return bytes_filled;
}

MemoryCache::AddrRange MemoryCache::GrowReadRange(addr_t read_addr,
                                                  addr_t caller_end,
                                                  size_t bytes_filled) const {
  const uint64_t line_size = m_L2_cache.GetLineByteSize();
  const addr_t line_base_addr = llvm::alignDown(read_addr, line_size);
  // Caps read-ahead at this many whole cache lines.
  static constexpr uint32_t kMaxCacheLinesPerRead = 2;
  const uint64_t grow_span = kMaxCacheLinesPerRead * line_size;

  // A request already past the cap spans a line, and one whose growth would
  // wrap cannot be grown, so both are asked for as they stand.
  if (line_base_addr > UINT64_MAX - grow_span ||
      caller_end > line_base_addr + grow_span)
    return AddrRange(read_addr, caller_end - read_addr);

  // Grow down to the line base so the fetch lands in L2 as a whole line rather
  // than an unaligned L1 fragment.
  if (!m_invalid_ranges.FindEntryThatIntersects(
          InvalidRanges::Entry(line_base_addr, read_addr - line_base_addr)) &&
      (caller_end <= line_base_addr + line_size || bytes_filled == 0))
    read_addr = line_base_addr;

  // Read up to the last line the request touches, skipping that line when L2
  // holds it.
  addr_t last_line_addr = llvm::alignDown(caller_end - 1, line_size);
  if (last_line_addr > line_base_addr && m_L2_cache.Holds(last_line_addr))
    last_line_addr -= line_size;
  const addr_t grow_target = last_line_addr + line_size;

  // Growth stops at the first invalid range among the bytes it adds.
  addr_t read_end = grow_target;
  if (grow_target > caller_end) {
    if (const InvalidRanges::Entry *invalid =
            m_invalid_ranges.FindEntryThatIntersects(
                InvalidRanges::Entry(caller_end, grow_target - caller_end)))
      read_end = invalid->GetRangeBase();
  }
  return AddrRange(read_addr, read_end - read_addr);
}

size_t MemoryCache::Read(addr_t addr, void *dst, size_t dst_len,
                         Status &error) {
  if (!dst || dst_len == 0)
    return 0;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  addr_t invalid_addr = LLDB_INVALID_ADDRESS;
  if (const InvalidRanges::Entry *invalid =
          m_invalid_ranges.FindEntryThatIntersects(
              InvalidRanges::Entry(addr, dst_len))) {
    invalid_addr = invalid->GetRangeBase();
    error = Status::FromErrorStringWithFormat(
        "memory read failed for 0x%" PRIx64, invalid_addr);
    if (invalid_addr <= addr)
      return 0;
    dst_len = invalid_addr - addr;
  }

  size_t bytes_from_cache = ReadFromCaches(addr, dst, dst_len);
  if (bytes_from_cache == dst_len)
    return dst_len;

  addr_t read_addr = addr + bytes_from_cache;
  addr_t read_end = addr + dst_len;
  // A request hits the invalid range above, don't grow.
  if (invalid_addr == LLDB_INVALID_ADDRESS) {
    const AddrRange grown =
        GrowReadRange(read_addr, read_end, bytes_from_cache);
    read_addr = grown.GetRangeBase();
    read_end = grown.GetRangeEnd();
  }

  std::vector<uint8_t> read_buf(read_end - read_addr);
  const size_t bytes_from_inferior = m_process.ReadMemoryFromInferior(
      read_addr, read_buf.data(), read_buf.size(), error);
  if (bytes_from_inferior == 0)
    return bytes_from_cache;

  AddCacheData(read_addr, read_buf.data(), bytes_from_inferior);

  // The grown or clipped fetch may not align with what the caller asked for,
  // so pull back only the portion contiguous with what dst already holds.
  uint8_t *dst_tail = static_cast<uint8_t *>(dst) + bytes_from_cache;
  return bytes_from_cache + ReadFromCaches(addr + bytes_from_cache, dst_tail,
                                           dst_len - bytes_from_cache);
}

llvm::SmallVector<llvm::MutableArrayRef<uint8_t>>
MemoryCache::ReadRanges(llvm::ArrayRef<Range<lldb::addr_t, size_t>> ranges,
                        llvm::MutableArrayRef<uint8_t> buffer) {
  // A cache hit writes into `buffer` below, so check its size before that
  // write.  Fail the same way Process::DoReadMemoryRanges does.
  auto total_ranges_len = llvm::sum_of(
      llvm::map_range(ranges, [](auto range) { return range.size; }));
  assert(buffer.size() >= total_ranges_len &&
         "MemoryCache::ReadRanges: provided buffer is too short");
  if (buffer.size() < total_ranges_len) {
    llvm::MutableArrayRef<uint8_t> empty;
    return {ranges.size(), empty};
  }

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> results;
  results.reserve(ranges.size());
  llvm::SmallVector<Range<lldb::addr_t, size_t>> missed_ranges;

  // Iterate once serving requests from the caches.
  for (auto range : ranges) {
    const lldb::addr_t addr = range.GetRangeBase();
    const size_t len = range.GetByteSize();

    if (m_invalid_ranges.FindEntryThatContains(addr)) {
      results.push_back(buffer.take_front(0));
      continue;
    }

    if (ReadFromCaches(addr, buffer.data(), len) == len) {
      results.push_back(buffer.take_front(len));
      buffer = buffer.drop_front(len);
      continue;
    }

    // Use a nullptr to denote this needs fetching.
    results.emplace_back(nullptr, nullptr);
    missed_ranges.push_back(range);
  }

  if (missed_ranges.empty())
    return results;

  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> fetched_buffers_vec =
      m_process.DoReadMemoryRanges(missed_ranges, buffer);
  auto fetched_buffers = llvm::ArrayRef(fetched_buffers_vec);

  for (auto [missed_range, fetched] : llvm::zip(missed_ranges, fetched_buffers))
    AddCacheData(missed_range.GetRangeBase(), fetched);

  // Use the just-fetched memory to fill in the gaps left by the cache.
  for (auto &result : results)
    if (result.data() == nullptr)
      result = fetched_buffers.consume_front();

  return results;
}

AllocatedBlock::AllocatedBlock(lldb::addr_t addr, uint32_t byte_size,
                               uint32_t permissions, uint32_t chunk_size)
    : m_range(addr, byte_size), m_permissions(permissions),
      m_chunk_size(chunk_size)
{
  // The entire address range is free to start with.
  m_free_blocks.Append(m_range);
  assert(byte_size > chunk_size);
}

AllocatedBlock::~AllocatedBlock() = default;

lldb::addr_t AllocatedBlock::ReserveBlock(uint32_t size) {
  // We must return something valid for zero bytes.
  if (size == 0)
    size = 1;
  Log *log = GetLog(LLDBLog::Process);

  const size_t free_count = m_free_blocks.GetSize();
  for (size_t i=0; i<free_count; ++i)
  {
    auto &free_block = m_free_blocks.GetEntryRef(i);
    const lldb::addr_t range_size = free_block.GetByteSize();
    if (range_size >= size)
    {
      // We found a free block that is big enough for our data. Figure out how
      // many chunks we will need and calculate the resulting block size we
      // will reserve.
      addr_t addr = free_block.GetRangeBase();
      size_t num_chunks = CalculateChunksNeededForSize(size);
      lldb::addr_t block_size = num_chunks * m_chunk_size;
      lldb::addr_t bytes_left = range_size - block_size;
      if (bytes_left == 0)
      {
        // The newly allocated block will take all of the bytes in this
        // available block, so we can just add it to the allocated ranges and
        // remove the range from the free ranges.
        m_reserved_blocks.Insert(free_block, false);
        m_free_blocks.RemoveEntryAtIndex(i);
      }
      else
      {
        // Make the new allocated range and add it to the allocated ranges.
        Range<lldb::addr_t, uint32_t> reserved_block(free_block);
        reserved_block.SetByteSize(block_size);
        // Insert the reserved range and don't combine it with other blocks in
        // the reserved blocks list.
        m_reserved_blocks.Insert(reserved_block, false);
        // Adjust the free range in place since we won't change the sorted
        // ordering of the m_free_blocks list.
        free_block.SetRangeBase(reserved_block.GetRangeEnd());
        free_block.SetByteSize(bytes_left);
      }
      LLDB_LOG_VERBOSE(log, "({0}) (size = {1} ({1:x})) => {2:x}", this, size,
                       addr);
      return addr;
    }
  }

  LLDB_LOG_VERBOSE(log, "({0}) (size = {1} ({1:x})) => {2:x}", this, size,
                   LLDB_INVALID_ADDRESS);
  return LLDB_INVALID_ADDRESS;
}

bool AllocatedBlock::FreeBlock(addr_t addr) {
  bool success = false;
  auto entry_idx = m_reserved_blocks.FindEntryIndexThatContains(addr);
  if (entry_idx != UINT32_MAX)
  {
    m_free_blocks.Insert(m_reserved_blocks.GetEntryRef(entry_idx), true);
    m_reserved_blocks.RemoveEntryAtIndex(entry_idx);
    success = true;
  }
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOG_VERBOSE(log, "({0}) (addr = {1:x}) => {2}", this, addr, success);
  return success;
}

AllocatedMemoryCache::AllocatedMemoryCache(Process &process)
    : m_process(process), m_mutex(), m_memory_map() {}

AllocatedMemoryCache::~AllocatedMemoryCache() = default;

void AllocatedMemoryCache::Clear(bool deallocate_memory) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_process.IsAlive() && deallocate_memory) {
    PermissionsToBlockMap::iterator pos, end = m_memory_map.end();
    for (pos = m_memory_map.begin(); pos != end; ++pos)
      m_process.DoDeallocateMemory(pos->second->GetBaseAddress());
  }
  m_memory_map.clear();
}

AllocatedMemoryCache::AllocatedBlockSP
AllocatedMemoryCache::AllocatePage(uint32_t byte_size, uint32_t permissions,
                                   uint32_t chunk_size, Status &error) {
  AllocatedBlockSP block_sp;
  const size_t page_size = 4096;
  const size_t num_pages = (byte_size + page_size - 1) / page_size;
  const size_t page_byte_size = num_pages * page_size;

  addr_t addr = m_process.DoAllocateMemory(page_byte_size, permissions, error);

  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log,
            "Process::DoAllocateMemory (byte_size = 0x%8.8" PRIx32
            ", permissions = %s) => 0x%16.16" PRIx64,
            (uint32_t)page_byte_size, GetPermissionsAsCString(permissions),
            (uint64_t)addr);

  if (addr != LLDB_INVALID_ADDRESS) {
    block_sp = std::make_shared<AllocatedBlock>(addr, page_byte_size,
                                                permissions, chunk_size);
    m_memory_map.insert(std::make_pair(permissions, block_sp));
  }
  return block_sp;
}

lldb::addr_t AllocatedMemoryCache::AllocateMemory(size_t byte_size,
                                                  uint32_t permissions,
                                                  Status &error) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  addr_t addr = LLDB_INVALID_ADDRESS;
  std::pair<PermissionsToBlockMap::iterator, PermissionsToBlockMap::iterator>
      range = m_memory_map.equal_range(permissions);

  for (PermissionsToBlockMap::iterator pos = range.first; pos != range.second;
       ++pos) {
    addr = (*pos).second->ReserveBlock(byte_size);
    if (addr != LLDB_INVALID_ADDRESS)
      break;
  }

  if (addr == LLDB_INVALID_ADDRESS) {
    AllocatedBlockSP block_sp(AllocatePage(byte_size, permissions, 16, error));

    if (block_sp)
      addr = block_sp->ReserveBlock(byte_size);
  }
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log,
            "AllocatedMemoryCache::AllocateMemory (byte_size = 0x%8.8" PRIx32
            ", permissions = %s) => 0x%16.16" PRIx64,
            (uint32_t)byte_size, GetPermissionsAsCString(permissions),
            (uint64_t)addr);
  return addr;
}

bool AllocatedMemoryCache::DeallocateMemory(lldb::addr_t addr) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  PermissionsToBlockMap::iterator pos, end = m_memory_map.end();
  bool success = false;
  for (pos = m_memory_map.begin(); pos != end; ++pos) {
    if (pos->second->Contains(addr)) {
      success = pos->second->FreeBlock(addr);
      break;
    }
  }
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log,
            "AllocatedMemoryCache::DeallocateMemory (addr = 0x%16.16" PRIx64
            ") => %i",
            (uint64_t)addr, success);
  return success;
}

bool AllocatedMemoryCache::IsInCache(lldb::addr_t addr) const {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  return llvm::any_of(m_memory_map, [addr](const auto &block) {
    return block.second->Contains(addr);
  });
}
