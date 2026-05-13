//===- llvm/MC/DXContainerInfo.cpp - DXContainer Info -----*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerInfo.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Support/SwapByteOrder.h"
#include <type_traits>

using namespace llvm;
using namespace llvm::mcdxbc;

template <typename StructT>
static void writeStruct(raw_ostream &OS, StructT S) {
  static_assert(std::is_class<StructT>() &&
                "This method must be used for writing structure types");
  if (sys::IsBigEndianHost)
    S.swapBytes();
  OS.write(reinterpret_cast<const char *>(&S), sizeof(StructT));
}

static void writeString(raw_ostream &OS, StringRef S) {
  OS.write(S.data(), S.size());
  // Write null terminator.
  OS.write_zeros(1);
}

void DebugName::setFileName(StringRef DebugFileName) {
  BaseData.first.NameLength = DebugFileName.size();
  BaseData.second = DebugFileName;
}

void DebugName::write(raw_ostream &OS) const {
  writeStruct(OS, BaseData.first);
  writeString(OS, BaseData.second.substr(0, BaseData.first.NameLength));
}
