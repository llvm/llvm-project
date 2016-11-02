//===-- Memory.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Memory.h"
// C Includes
#include <inttypes.h>
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/State.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// MemoryCache constructor
//----------------------------------------------------------------------
MemoryCache::MemoryCache(Process &process)
    : m_mutex(), m_L1_cache(), m_L2_cache(), m_invalid_ranges(),
      m_process(process),
      m_L2_cache_line_byte_size(process.GetMemoryCacheLineSize()) {}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
MemoryCache::~MemoryCache() {}

void MemoryCache::Clear(bool clear_invalid_ranges) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_L1_cache.clear();
  m_L2_cache.clear();
  if (clear_invalid_ranges)
    m_invalid_ranges.Clear();
  m_L2_cache_line_byte_size = m_process.GetMemoryCacheLineSize();
}

void MemoryCache::AddL1CacheData(lldb::addr_t addr, const void *src,
                                 size_t src_len) {
  AddL1CacheData(
      addr, DataBufferSP(new DataBufferHeap(DataBufferHeap(src, src_len))));
}

void MemoryCache::AddL1CacheData(lldb::addr_t addr,
                                 const DataBufferSP &data_buffer_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_L1_cache[addr] = data_buffer_sp;
}

void MemoryCache::Flush(addr_t addr, size_t size) {
  if (size == 0)
    return;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);

  // Erase any blocks from the L1 cache that intersect with the flush range
  if (!m_L1_cache.empty()) {
    AddrRange flush_range(addr, size);
    BlockMap::iterator pos = m_L1_cache.upper_bound(addr);
    if (pos != m_L1_cache.begin()) {
      --pos;
    }
    while (pos != m_L1_cache.end()) {
      AddrRange chunk_range(pos->first, pos->second->GetByteSize());
      if (!chunk_range.DoesIntersect(flush_range))
        break;
      pos = m_L1_cache.erase(pos);
    }
  }

  if (!m_L2_cache.empty()) {
    const uint32_t cache_line_byte_size = m_L2_cache_line_byte_size;
    const addr_t end_addr = (addr + size - 1);
    const addr_t first_cache_line_addr = addr - (addr % cache_line_byte_size);
    const addr_t last_cache_line_addr =
        end_addr - (end_addr % cache_line_byte_size);
    // Watch for overflow where size will cause us to go off the end of the
    // 64 bit address space
    uint32_t num_cache_lines;
    if (last_cache_line_addr >= first_cache_line_addr)
      num_cache_lines = ((last_cache_line_addr - first_cache_line_addr) /
                         cache_line_byte_size) +
                        1;
    else
      num_cache_lines =
          (UINT64_MAX - first_cache_line_addr + 1) / cache_line_byte_size;

    uint32_t cache_idx = 0;
    for (addr_t curr_addr = first_cache_line_addr; cache_idx < num_cache_lines;
         curr_addr += cache_line_byte_size, ++cache_idx) {
      BlockMap::iterator pos = m_L2_cache.find(curr_addr);
      if (pos != m_L2_cache.end())
        m_L2_cache.erase(pos);
    }
  }
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
        return m_invalid_ranges.RemoveEntrtAtIndex(idx);
    }
  }
  return false;
}

size_t MemoryCache::Read(addr_t addr, void *dst, size_t dst_len, Error &error) {
  size_t bytes_left = dst_len;

  // Check the L1 cache for a range that contain the entire memory read.
  // If we find a range in the L1 cache that does, we use it. Else we fall
  // back to reading memory in m_L2_cache_line_byte_size byte sized chunks.
  // The L1 cache contains chunks of memory that are not required to be
  // m_L2_cache_line_byte_size bytes in size, so we don't try anything
  // tricky when reading from them (no partial reads from the L1 cache).

  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (!m_L1_cache.empty()) {
    AddrRange read_range(addr, dst_len);
    BlockMap::iterator pos = m_L1_cache.upper_bound(addr);
    if (pos != m_L1_cache.begin()) {
      --pos;
    }
    AddrRange chunk_range(pos->first, pos->second->GetByteSize());
    if (chunk_range.Contains(read_range)) {
      memcpy(dst, pos->second->GetBytes() + addr - chunk_range.GetRangeBase(),
             dst_len);
      return dst_len;
    }
  }

  // If this memory read request is larger than the cache line size, then
  // we (1) try to read as much of it at once as possible, and (2) don't
  // add the data to the memory cache.  We don't want to split a big read
  // up into more separate reads than necessary, and with a large memory read
  // request, it is unlikely that the caller function will ask for the next
  // 4 bytes after the large memory read - so there's little benefit to saving
  // it in the cache.
  if (dst && dst_len > m_L2_cache_line_byte_size) {
    size_t bytes_read =
        m_process.ReadMemoryFromInferior(addr, dst, dst_len, error);
    // Add this non block sized range to the L1 cache if we actually read
    // anything
    if (bytes_read > 0)
      AddL1CacheData(addr, dst, bytes_read);
    return bytes_read;
  }

  if (dst && bytes_left > 0) {
    const uint32_t cache_line_byte_size = m_L2_cache_line_byte_size;
    uint8_t *dst_buf = (uint8_t *)dst;
    addr_t curr_addr = addr - (addr % cache_line_byte_size);
    addr_t cache_offset = addr - curr_addr;

    while (bytes_left > 0) {
      if (m_invalid_ranges.FindEntryThatContains(curr_addr)) {
        error.SetErrorStringWithFormat("memory read failed for 0x%" PRIx64,
                                       curr_addr);
        return dst_len - bytes_left;
      }

      BlockMap::const_iterator pos = m_L2_cache.find(curr_addr);
      BlockMap::const_iterator end = m_L2_cache.end();

      if (pos != end) {
        size_t curr_read_size = cache_line_byte_size - cache_offset;
        if (curr_read_size > bytes_left)
          curr_read_size = bytes_left;

        memcpy(dst_buf + dst_len - bytes_left,
               pos->second->GetBytes() + cache_offset, curr_read_size);

        bytes_left -= curr_read_size;
        curr_addr += curr_read_size + cache_offset;
        cache_offset = 0;

        if (bytes_left > 0) {
          // Get sequential cache page hits
          for (++pos; (pos != end) && (bytes_left > 0); ++pos) {
            assert((curr_addr % cache_line_byte_size) == 0);

            if (pos->first != curr_addr)
              break;

            curr_read_size = pos->second->GetByteSize();
            if (curr_read_size > bytes_left)
              curr_read_size = bytes_left;

            memcpy(dst_buf + dst_len - bytes_left, pos->second->GetBytes(),
                   curr_read_size);

            bytes_left -= curr_read_size;
            curr_addr += curr_read_size;

            // We have a cache page that succeeded to read some bytes
            // but not an entire page. If this happens, we must cap
            // off how much data we are able to read...
            if (pos->second->GetByteSize() != cache_line_byte_size)
              return dst_len - bytes_left;
          }
        }
      }

      // We need to read from the process

      if (bytes_left > 0) {
        assert((curr_addr % cache_line_byte_size) == 0);
        std::unique_ptr<DataBufferHeap> data_buffer_heap_ap(
            new DataBufferHeap(cache_line_byte_size, 0));
        size_t process_bytes_read = m_process.ReadMemoryFromInferior(
            curr_addr, data_buffer_heap_ap->GetBytes(),
            data_buffer_heap_ap->GetByteSize(), error);
        if (process_bytes_read == 0)
          return dst_len - bytes_left;

        if (process_bytes_read != cache_line_byte_size)
          data_buffer_heap_ap->SetByteSize(process_bytes_read);
        m_L2_cache[curr_addr] = DataBufferSP(data_buffer_heap_ap.release());
        // We have read data and put it into the cache, continue through the
        // loop again to get the data out of the cache...
      }
    }
  }

  return dst_len - bytes_left;
}

AllocatedBlock::AllocatedBlock(lldb::addr_t addr, uint32_t byte_size,
                               uint32_t permissions, uint32_t chunk_size)
    : m_addr(addr), m_byte_size(byte_size), m_permissions(permissions),
      m_chunk_size(chunk_size), m_offset_to_chunk_size()
//    m_allocated (byte_size / chunk_size)
{
  assert(byte_size > chunk_size);
}

AllocatedBlock::~AllocatedBlock() {}

lldb::addr_t AllocatedBlock::ReserveBlock(uint32_t size) {
  addr_t addr = LLDB_INVALID_ADDRESS;
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_VERBOSE));
  if (size <= m_byte_size) {
    const uint32_t needed_chunks = CalculateChunksNeededForSize(size);

    if (m_offset_to_chunk_size.empty()) {
      m_offset_to_chunk_size[0] = needed_chunks;
      if (log)
        log->Printf("[1] AllocatedBlock::ReserveBlock(%p) (size = %u (0x%x)) "
                    "=> offset = 0x%x, %u %u bit chunks",
                    (void *)this, size, size, 0, needed_chunks, m_chunk_size);
      addr = m_addr;
    } else {
      uint32_t last_offset = 0;
      OffsetToChunkSize::const_iterator pos = m_offset_to_chunk_size.begin();
      OffsetToChunkSize::const_iterator end = m_offset_to_chunk_size.end();
      while (pos != end) {
        if (pos->first > last_offset) {
          const uint32_t bytes_available = pos->first - last_offset;
          const uint32_t num_chunks =
              CalculateChunksNeededForSize(bytes_available);
          if (num_chunks >= needed_chunks) {
            m_offset_to_chunk_size[last_offset] = needed_chunks;
            if (log)
              log->Printf("[2] AllocatedBlock::ReserveBlock(%p) (size = %u "
                          "(0x%x)) => offset = 0x%x, %u %u bit chunks - "
                          "num_chunks %lu",
                          (void *)this, size, size, last_offset, needed_chunks,
                          m_chunk_size, m_offset_to_chunk_size.size());
            addr = m_addr + last_offset;
            break;
          }
        }

        last_offset = pos->first + pos->second * m_chunk_size;

        if (++pos == end) {
          // Last entry...
          const uint32_t chunks_left =
              CalculateChunksNeededForSize(m_byte_size - last_offset);
          if (chunks_left >= needed_chunks) {
            m_offset_to_chunk_size[last_offset] = needed_chunks;
            if (log)
              log->Printf("[3] AllocatedBlock::ReserveBlock(%p) (size = %u "
                          "(0x%x)) => offset = 0x%x, %u %u bit chunks - "
                          "num_chunks %lu",
                          (void *)this, size, size, last_offset, needed_chunks,
                          m_chunk_size, m_offset_to_chunk_size.size());
            addr = m_addr + last_offset;
            break;
          }
        }
      }
    }
    //        const uint32_t total_chunks = m_allocated.size ();
    //        uint32_t unallocated_idx = 0;
    //        uint32_t allocated_idx = m_allocated.find_first();
    //        uint32_t first_chunk_idx = UINT32_MAX;
    //        uint32_t num_chunks;
    //        while (1)
    //        {
    //            if (allocated_idx == UINT32_MAX)
    //            {
    //                // No more bits are set starting from unallocated_idx, so
    //                we
    //                // either have enough chunks for the request, or we don't.
    //                // Either way we break out of the while loop...
    //                num_chunks = total_chunks - unallocated_idx;
    //                if (needed_chunks <= num_chunks)
    //                    first_chunk_idx = unallocated_idx;
    //                break;
    //            }
    //            else if (allocated_idx > unallocated_idx)
    //            {
    //                // We have some allocated chunks, check if there are
    //                enough
    //                // free chunks to satisfy the request?
    //                num_chunks = allocated_idx - unallocated_idx;
    //                if (needed_chunks <= num_chunks)
    //                {
    //                    // Yep, we have enough!
    //                    first_chunk_idx = unallocated_idx;
    //                    break;
    //                }
    //            }
    //
    //            while (unallocated_idx < total_chunks)
    //            {
    //                if (m_allocated[unallocated_idx])
    //                    ++unallocated_idx;
    //                else
    //                    break;
    //            }
    //
    //            if (unallocated_idx >= total_chunks)
    //                break;
    //
    //            allocated_idx = m_allocated.find_next(unallocated_idx);
    //        }
    //
    //        if (first_chunk_idx != UINT32_MAX)
    //        {
    //            const uint32_t end_bit_idx = unallocated_idx + needed_chunks;
    //            for (uint32_t idx = first_chunk_idx; idx < end_bit_idx; ++idx)
    //                m_allocated.set(idx);
    //            return m_addr + m_chunk_size * first_chunk_idx;
    //        }
  }

  if (log)
    log->Printf("AllocatedBlock::ReserveBlock(%p) (size = %u (0x%x)) => "
                "0x%16.16" PRIx64,
                (void *)this, size, size, (uint64_t)addr);
  return addr;
}

bool AllocatedBlock::FreeBlock(addr_t addr) {
  uint32_t offset = addr - m_addr;
  OffsetToChunkSize::iterator pos = m_offset_to_chunk_size.find(offset);
  bool success = false;
  if (pos != m_offset_to_chunk_size.end()) {
    m_offset_to_chunk_size.erase(pos);
    success = true;
  }
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_VERBOSE));
  if (log)
    log->Printf("AllocatedBlock::FreeBlock(%p) (addr = 0x%16.16" PRIx64
                ") => %i, num_chunks: %lu",
                (void *)this, (uint64_t)addr, success,
                m_offset_to_chunk_size.size());
  return success;
}

AllocatedMemoryCache::AllocatedMemoryCache(Process &process)
    : m_process(process), m_mutex(), m_memory_map() {}

AllocatedMemoryCache::~AllocatedMemoryCache() {}

void AllocatedMemoryCache::Clear() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if (m_process.IsAlive()) {
    PermissionsToBlockMap::iterator pos, end = m_memory_map.end();
    for (pos = m_memory_map.begin(); pos != end; ++pos)
      m_process.DoDeallocateMemory(pos->second->GetBaseAddress());
  }
  m_memory_map.clear();
}

AllocatedMemoryCache::AllocatedBlockSP
AllocatedMemoryCache::AllocatePage(uint32_t byte_size, uint32_t permissions,
                                   uint32_t chunk_size, Error &error) {
  AllocatedBlockSP block_sp;
  const size_t page_size = 4096;
  const size_t num_pages = (byte_size + page_size - 1) / page_size;
  const size_t page_byte_size = num_pages * page_size;

  addr_t addr = m_process.DoAllocateMemory(page_byte_size, permissions, error);

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log) {
    log->Printf("Process::DoAllocateMemory (byte_size = 0x%8.8" PRIx32
                ", permissions = %s) => 0x%16.16" PRIx64,
                (uint32_t)page_byte_size, GetPermissionsAsCString(permissions),
                (uint64_t)addr);
  }

  if (addr != LLDB_INVALID_ADDRESS) {
    block_sp.reset(
        new AllocatedBlock(addr, page_byte_size, permissions, chunk_size));
    m_memory_map.insert(std::make_pair(permissions, block_sp));
  }
  return block_sp;
}

lldb::addr_t AllocatedMemoryCache::AllocateMemory(size_t byte_size,
                                                  uint32_t permissions,
                                                  Error &error) {
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
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log)
    log->Printf(
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
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
  if (log)
    log->Printf("AllocatedMemoryCache::DeallocateMemory (addr = 0x%16.16" PRIx64
                ") => %i",
                (uint64_t)addr, success);
  return success;
}
