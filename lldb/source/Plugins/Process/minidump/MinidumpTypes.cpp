//===-- MinidumpTypes.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinidumpTypes.h"

// C includes
// C++ includes

using namespace lldb_private;
using namespace minidump;

// MinidumpMiscInfo
const MinidumpMiscInfo *MinidumpMiscInfo::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpMiscInfo *misc_info;
  Status error = consumeObject(data, misc_info);
  if (error.Fail())
    return nullptr;

  return misc_info;
}

llvm::Optional<lldb::pid_t> MinidumpMiscInfo::GetPid() const {
  uint32_t pid_flag = static_cast<uint32_t>(MinidumpMiscInfoFlags::ProcessID);
  if (flags1 & pid_flag)
    return llvm::Optional<lldb::pid_t>(process_id);

  return llvm::None;
}

// Linux Proc Status
// it's stored as an ascii string in the file
llvm::Optional<LinuxProcStatus>
LinuxProcStatus::Parse(llvm::ArrayRef<uint8_t> &data) {
  LinuxProcStatus result;
  result.proc_status =
      llvm::StringRef(reinterpret_cast<const char *>(data.data()), data.size());
  data = data.drop_front(data.size());

  llvm::SmallVector<llvm::StringRef, 0> lines;
  result.proc_status.split(lines, '\n', 42);
  // /proc/$pid/status has 41 lines, but why not use 42?
  for (auto line : lines) {
    if (line.consume_front("Pid:")) {
      line = line.trim();
      if (!line.getAsInteger(10, result.pid))
        return result;
    }
  }

  return llvm::None;
}

lldb::pid_t LinuxProcStatus::GetPid() const { return pid; }

// Exception stuff
const MinidumpExceptionStream *
MinidumpExceptionStream::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpExceptionStream *exception_stream = nullptr;
  Status error = consumeObject(data, exception_stream);
  if (error.Fail())
    return nullptr;

  return exception_stream;
}

std::pair<llvm::ArrayRef<MinidumpMemoryDescriptor64>, uint64_t>
MinidumpMemoryDescriptor64::ParseMemory64List(llvm::ArrayRef<uint8_t> &data) {
  const llvm::support::ulittle64_t *mem_ranges_count;
  Status error = consumeObject(data, mem_ranges_count);
  if (error.Fail() ||
      *mem_ranges_count * sizeof(MinidumpMemoryDescriptor64) > data.size())
    return {};

  const llvm::support::ulittle64_t *base_rva;
  error = consumeObject(data, base_rva);
  if (error.Fail())
    return {};

  return std::make_pair(
      llvm::makeArrayRef(
          reinterpret_cast<const MinidumpMemoryDescriptor64 *>(data.data()),
          *mem_ranges_count),
      *base_rva);
}

std::vector<const MinidumpMemoryInfo *>
MinidumpMemoryInfo::ParseMemoryInfoList(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpMemoryInfoListHeader *header;
  Status error = consumeObject(data, header);
  if (error.Fail() ||
      header->size_of_header < sizeof(MinidumpMemoryInfoListHeader) ||
      header->size_of_entry < sizeof(MinidumpMemoryInfo))
    return {};

  data = data.drop_front(header->size_of_header -
                         sizeof(MinidumpMemoryInfoListHeader));

  if (header->size_of_entry * header->num_of_entries > data.size())
    return {};

  std::vector<const MinidumpMemoryInfo *> result;
  result.reserve(header->num_of_entries);

  for (uint64_t i = 0; i < header->num_of_entries; ++i) {
    result.push_back(reinterpret_cast<const MinidumpMemoryInfo *>(
        data.data() + i * header->size_of_entry));
  }

  return result;
}
