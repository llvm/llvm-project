//===-- MinidumpFileBuilder.h ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Structure holding data neccessary for minidump file creation.
///
/// The class MinidumpFileWriter is used to hold the data that will eventually
/// be dumped to the file.
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_MINIDUMP_MINIDUMPFILEBUILDER_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_MINIDUMP_MINIDUMPFILEBUILDER_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <variant>

#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

#include "llvm/BinaryFormat/Minidump.h"
#include "llvm/Object/Minidump.h"

// Write std::string to minidump in the UTF16 format(with null termination char)
// with the size(without null termination char) preceding the UTF16 string.
// Empty strings are also printed with zero length and just null termination
// char.
lldb_private::Status WriteString(const std::string &to_write,
                                 lldb_private::DataBufferHeap *buffer);

/// \class MinidumpFileBuilder
/// Minidump writer for Linux
///
/// This class provides a Minidump writer that is able to
/// snapshot the current process state. For the whole time, it stores all
/// the data on heap.
class MinidumpFileBuilder {
public:
  MinidumpFileBuilder(lldb::FileUP &&core_file,
                      const lldb::ProcessSP &process_sp)
      : m_process_sp(std::move(process_sp)),
        m_core_file(std::move(core_file)){};

  MinidumpFileBuilder(const MinidumpFileBuilder &) = delete;
  MinidumpFileBuilder &operator=(const MinidumpFileBuilder &) = delete;

  MinidumpFileBuilder(MinidumpFileBuilder &&other) = default;
  MinidumpFileBuilder &operator=(MinidumpFileBuilder &&other) = default;

  ~MinidumpFileBuilder() = default;

  lldb_private::Status AddHeaderAndCalculateDirectories();
  // Add SystemInfo stream, used for storing the most basic information
  // about the system, platform etc...
  lldb_private::Status AddSystemInfo();
  // Add ModuleList stream, containing information about all loaded modules
  // at the time of saving minidump.
  lldb_private::Status AddModuleList();
  // Add ThreadList stream, containing information about all threads running
  // at the moment of core saving. Contains information about thread
  // contexts.
  lldb_private::Status AddThreadList();
  // Add Exception streams for any threads that stopped with exceptions.
  void AddExceptions();
  // Add MiscInfo stream, mainly providing ProcessId
  void AddMiscInfo();
  // Add informative files about a Linux process
  void AddLinuxFileStreams();

  lldb_private::Status AddMemory(lldb::SaveCoreStyle core_style);

  // Run cleanup and write all remaining bytes to file
  lldb_private::Status DumpToFile();

private:
  // Add data to the end of the buffer, if the buffer exceeds the flush level,
  // trigger a flush.
  lldb_private::Status AddData(const void *data, size_t size);
  // Add MemoryList stream, containing dumps of important memory segments
  lldb_private::Status
  AddMemoryList_64(const lldb_private::Process::CoreFileMemoryRanges &ranges);
  lldb_private::Status
  AddMemoryList_32(const lldb_private::Process::CoreFileMemoryRanges &ranges);
  lldb_private::Status FixThreads();
  lldb_private::Status FlushToDisk();

  lldb_private::Status DumpHeader() const;
  lldb_private::Status DumpDirectories() const;
  bool CheckIf_64Bit(const size_t size);
  // Add directory of StreamType pointing to the current end of the prepared
  // file with the specified size.
  void AddDirectory(llvm::minidump::StreamType type, uint64_t stream_size);
  lldb::offset_t GetCurrentDataEndOffset() const;
  // Stores directories to fill in later
  std::vector<llvm::minidump::Directory> m_directories;
  // When we write off the threads for the first time, we need to clean them up
  // and give them the correct RVA once we write the stack memory list.
  std::map<lldb::addr_t, llvm::minidump::Thread> m_thread_by_range_start;
  // Main data buffer consisting of data without the minidump header and
  // directories
  lldb_private::DataBufferHeap m_data;
  lldb::ProcessSP m_process_sp;

  uint m_expected_directories = 0;
  uint64_t m_saved_data_size = 0;
  lldb::offset_t m_thread_list_start = 0;
  // We set the max write amount to 128 mb, this is arbitrary
  // but we want to try to keep the size of m_data small
  // and we will only exceed a 128 mb buffer if we get a memory region
  // that is larger than 128 mb.
  static constexpr size_t m_write_chunk_max = (1024 * 1024 * 128);

  static constexpr size_t header_size = sizeof(llvm::minidump::Header);
  static constexpr size_t directory_size = sizeof(llvm::minidump::Directory);

  // More that one place can mention the register thread context locations,
  // so when we emit the thread contents, remember where it is so we don't have
  // to duplicate it in the exception data.
  std::map<lldb::tid_t, llvm::minidump::LocationDescriptor> m_tid_to_reg_ctx;
  lldb::FileUP m_core_file;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_MINIDUMP_MINIDUMPFILEBUILDER_H
