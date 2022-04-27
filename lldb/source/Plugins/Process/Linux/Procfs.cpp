//===-- Procfs.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Procfs.h"

#include "lldb/Host/linux/Support.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

Expected<ArrayRef<uint8_t>> lldb_private::process_linux::GetProcfsCpuInfo() {
  static Optional<std::vector<uint8_t>> cpu_info;
  if (!cpu_info) {
    auto buffer_or_error = errorOrToExpected(getProcFile("cpuinfo"));
    if (!buffer_or_error)
      return buffer_or_error.takeError();
    MemoryBuffer &buffer = **buffer_or_error;
    cpu_info = std::vector<uint8_t>(
        reinterpret_cast<const uint8_t *>(buffer.getBufferStart()),
        reinterpret_cast<const uint8_t *>(buffer.getBufferEnd()));
  }
  return *cpu_info;
}

Expected<std::vector<int>>
lldb_private::process_linux::GetAvailableLogicalCoreIDs(StringRef cpuinfo) {
  SmallVector<StringRef, 8> lines;
  cpuinfo.split(lines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::vector<int> logical_cores;

  for (StringRef line : lines) {
    std::pair<StringRef, StringRef> key_value = line.split(':');
    auto key = key_value.first.trim();
    auto val = key_value.second.trim();
    if (key == "processor") {
      int processor;
      if (val.getAsInteger(10, processor))
        return createStringError(
            inconvertibleErrorCode(),
            "Failed parsing the /proc/cpuinfo line entry: %s", line.data());
      logical_cores.push_back(processor);
    }
  }
  return logical_cores;
}

llvm::Expected<llvm::ArrayRef<int>>
lldb_private::process_linux::GetAvailableLogicalCoreIDs() {
  static Optional<std::vector<int>> logical_cores_ids;
  if (!logical_cores_ids) {
    // We find the actual list of core ids by parsing /proc/cpuinfo
    Expected<ArrayRef<uint8_t>> cpuinfo = GetProcfsCpuInfo();
    if (!cpuinfo)
      return cpuinfo.takeError();

    Expected<std::vector<int>> core_ids = GetAvailableLogicalCoreIDs(
        StringRef(reinterpret_cast<const char *>(cpuinfo->data())));
    if (!core_ids)
      return core_ids.takeError();

    logical_cores_ids.emplace(std::move(*core_ids));
  }
  return *logical_cores_ids;
}
