//===- GlobalData.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GLOBALDATA_H
#define LLVM_DEBUGINFO_GSYM_GLOBALDATA_H

#include <cstdint>

namespace llvm {
namespace gsym {

enum GlobalInfoType : uint32_t {
  EndOfList = 0u,
  AddrOffsets = 1u,
  AddrInfoOffsets = 2u,
  StringTable = 3u,
  FileTable = 4u,
  FunctionInfo = 5u,
};

/// GlobalData describes a section of data in a GSYM file by its type, file
/// offset, and size. This is used to support 64-bit GSYM files where data
/// sections can be located at arbitrary file offsets.
struct GlobalData {
  GlobalInfoType Type;
  uint64_t FileOffset;
  uint64_t FileSize;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GLOBALDATA_H
