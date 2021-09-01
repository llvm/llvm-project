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

#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/Status.h"

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
  MinidumpFileBuilder() = default;

  MinidumpFileBuilder(const MinidumpFileBuilder &) = delete;
  MinidumpFileBuilder &operator=(const MinidumpFileBuilder &) = delete;

  MinidumpFileBuilder(MinidumpFileBuilder &&other) = default;
  MinidumpFileBuilder &operator=(MinidumpFileBuilder &&other) = default;

  ~MinidumpFileBuilder() = default;

  // Add SystemInfo stream, used for storing the most basic information
  // about the system, platform etc...
  lldb_private::Status AddSystemInfo(const llvm::Triple &target_triple);
  // Add ModuleList stream, containing information about all loaded modules
  // at the time of saving minidump.
  lldb_private::Status AddModuleList(lldb_private::Target &target);
  // Add ThreadList stream, containing information about all threads running
  // at the moment of core saving. Contains information about thread
  // contexts.
  lldb_private::Status AddThreadList(const lldb::ProcessSP &process_sp);
  // Add Exception stream, this contains information about the exception
  // that stopped the process. In case no thread made exception it return
  // failed status.
  lldb_private::Status AddException(const lldb::ProcessSP &process_sp);
  // Add MemoryList stream, containing dumps of important memory segments
  lldb_private::Status AddMemoryList(const lldb::ProcessSP &process_sp);
  // Add MiscInfo stream, mainly providing ProcessId
  void AddMiscInfo(const lldb::ProcessSP &process_sp);
  // Add informative files about a Linux process
  void AddLinuxFileStreams(const lldb::ProcessSP &process_sp);
  // Dump the prepared data into file. In case of the failure data are
  // intact.
  lldb_private::Status Dump(lldb::FileUP &core_file) const;
  // Returns the current number of directories(streams) that have been so far
  // created. This number of directories will be dumped when calling Dump()
  size_t GetDirectoriesNum() const;

private:
  // Add directory of StreamType pointing to the current end of the prepared
  // file with the specified size.
  void AddDirectory(llvm::minidump::StreamType type, size_t stream_size);
  size_t GetCurrentDataEndOffset() const;

  // Stores directories to later put them at the end of minidump file
  std::vector<llvm::minidump::Directory> m_directories;
  // Main data buffer consisting of data without the minidump header and
  // directories
  lldb_private::DataBufferHeap m_data;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_MINIDUMP_MINIDUMPFILEBUILDER_H