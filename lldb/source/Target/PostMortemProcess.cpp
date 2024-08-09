//===-- PostMortemProcess.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/PostMortemProcess.h"

#include "lldb/Core/Module.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/lldb-forward.h"

using namespace lldb;
using namespace lldb_private;

lldb::addr_t PostMortemProcess::FindInMemory(lldb::addr_t low,
                                             lldb::addr_t high,
                                             const uint8_t *buf, size_t size) {
  const size_t region_size = high - low;
  if (region_size < size)
    return LLDB_INVALID_ADDRESS;

  size_t mem_size = 0;
  const uint8_t *data = PeekMemory(low, high, mem_size);
  if (data == nullptr || mem_size != region_size) {
    LLDB_LOG(GetLog(LLDBLog::Process),
             "Failed to get contiguous memory region for search. low: 0x{}, "
             "high: 0x{}. Failling back to Process::FindInMemory",
             low, high);
    // In an edge case when the search has to happen across non-contiguous
    // memory, we will have to fall back on the Process::FindInMemory.
    return Process::FindInMemory(low, high, buf, size);
  }

  return Process::FindInMemoryGeneric(data, low, high, buf, size);
}

const uint8_t *PostMortemProcess::DoPeekMemory(
    lldb::ModuleSP &core_module_sp, VMRangeToFileOffset &core_aranges,
    lldb::addr_t low, lldb::addr_t high, size_t &available_bytes) {

  ObjectFile *core_objfile = core_module_sp->GetObjectFile();

  if (core_objfile == nullptr) {
    available_bytes = 0;
    return nullptr;
  }

  const VMRangeToFileOffset::Entry *core_memory_entry =
      core_aranges.FindEntryThatContains(low);
  if (core_memory_entry == nullptr || core_memory_entry->GetRangeEnd() < low) {
    available_bytes = 0;
    return nullptr;
  }
  const lldb::addr_t offset = low - core_memory_entry->GetRangeBase();
  const lldb::addr_t file_start = core_memory_entry->data.GetRangeBase();
  const lldb::addr_t file_end = core_memory_entry->data.GetRangeEnd();

  if (file_start == file_end) {
    available_bytes = 0;
    return nullptr;
  }
  size_t bytes_available = 0;
  if (file_end > file_start + offset)
    bytes_available = file_end - (file_start + offset);

  size_t bytes_to_read = high - low;
  bytes_to_read = std::min(bytes_to_read, bytes_available);
  if (bytes_to_read == 0) {
    available_bytes = 0;
    return nullptr;
  }

  const uint8_t *ret =
      core_objfile->PeekData(core_memory_entry->data.GetRangeBase() + offset,
                             bytes_to_read, available_bytes);
  return ret;
}
