//===-- MinidumpTypes.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinidumpTypes.h"
#include <optional>

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

std::optional<lldb::pid_t> MinidumpMiscInfo::GetPid() const {
  uint32_t pid_flag = static_cast<uint32_t>(MinidumpMiscInfoFlags::ProcessID);
  if (flags1 & pid_flag)
    return std::optional<lldb::pid_t>(process_id);

  return std::nullopt;
}

// Linux Proc Status
// it's stored as an ascii string in the file
std::optional<LinuxProcStatus>
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

  return std::nullopt;
}

lldb::pid_t LinuxProcStatus::GetPid() const { return pid; }
