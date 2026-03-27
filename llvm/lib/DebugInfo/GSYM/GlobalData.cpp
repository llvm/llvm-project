//===- GlobalData.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/Support/DataExtractor.h"

using namespace llvm;
using namespace gsym;

llvm::Error GlobalData::encode(FileWriter &O) const {
  O.writeU32(static_cast<uint32_t>(Type));
  O.writeU32(Padding);
  O.writeU64(FileOffset);
  O.writeU64(FileSize);
  return Error::success();
}

llvm::Expected<GlobalData> GlobalData::decode(DataExtractor &Data,
                                              uint64_t &Offset) {
  if (!Data.isValidOffsetForDataOfSize(Offset, 24))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GlobalData entry");
  GlobalData GD;
  GD.Type = static_cast<GlobalInfoType>(Data.getU32(&Offset));
  GD.Padding = Data.getU32(&Offset);
  GD.FileOffset = Data.getU64(&Offset);
  GD.FileSize = Data.getU64(&Offset);
  return GD;
}
