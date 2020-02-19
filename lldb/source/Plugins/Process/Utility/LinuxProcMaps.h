//===-- LinuxProcMaps.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LINUXPROCMAPS_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LINUXPROCMAPS_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include <functional>


namespace lldb_private {

typedef std::function<bool(const lldb_private::MemoryRegionInfo &,
                           const lldb_private::Status &)> LinuxMapCallback;

void ParseLinuxMapRegions(llvm::StringRef linux_map,
                          LinuxMapCallback const &callback);

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_LINUXPROCMAPS_H
