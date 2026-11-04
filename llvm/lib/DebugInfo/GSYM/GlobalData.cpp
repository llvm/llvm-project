//===- GlobalData.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include <inttypes.h>

using namespace llvm;
using namespace gsym;

void GlobalData::encode(FileWriter &O) const {
  O.writeU32(static_cast<uint32_t>(Type));
  O.writeU64(FileOffset);
  O.writeU64(FileSize);
}

llvm::Expected<GlobalData> GlobalData::decode(GsymDataExtractor &GsymData,
                                              uint64_t &Offset) {
  if (!GsymData.isValidOffsetForDataOfSize(Offset, 20))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GlobalData entry");
  GlobalData GD;
  GD.Type = static_cast<GlobalInfoType>(GsymData.getU32(&Offset));
  GD.FileOffset = GsymData.getU64(&Offset);
  GD.FileSize = GsymData.getU64(&Offset);
  return GD;
}

StringRef llvm::gsym::getNameForGlobalInfoType(GlobalInfoType Type) {
  switch (Type) {
  case GlobalInfoType::EndOfList:
    return "EndOfList";
  case GlobalInfoType::AddrOffsets:
    return "AddrOffsets";
  case GlobalInfoType::AddrInfoOffsets:
    return "AddrInfoOffsets";
  case GlobalInfoType::StringTable:
    return "StringTable";
  case GlobalInfoType::FileTable:
    return "FileTable";
  case GlobalInfoType::FunctionInfo:
    return "FunctionInfo";
  case GlobalInfoType::UUID:
    return "UUID";
  }
  return "Unknown";
}
