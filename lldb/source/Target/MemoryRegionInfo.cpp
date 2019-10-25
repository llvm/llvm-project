//===-- MemoryRegionInfo.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfo.h"

llvm::raw_ostream &lldb_private::operator<<(llvm::raw_ostream &OS,
                                            const MemoryRegionInfo &Info) {
  return OS << llvm::formatv("MemoryRegionInfo([{0}, {1}), {2}, {3}, {4}, {5}, "
                             "`{6}`, {7}, {8})",
                             Info.GetRange().GetRangeBase(),
                             Info.GetRange().GetRangeEnd(), Info.GetReadable(),
                             Info.GetWritable(), Info.GetExecutable(),
                             Info.GetMapped(), Info.GetName(), Info.GetFlash(),
                             Info.GetBlocksize());
}
