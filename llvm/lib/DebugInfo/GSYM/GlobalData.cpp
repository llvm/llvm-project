//===- GlobalData.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/Support/DataExtractor.h"
#include <inttypes.h>

using namespace llvm;
using namespace gsym;

void GlobalData::encode(FileWriter &O) const {
  O.writeU32(static_cast<uint32_t>(Type));
  O.writeU64(FileOffset);
  O.writeU64(FileSize);
}

llvm::Expected<StringRef>
GlobalData::getStringRef(DataExtractor &GsymData) const {
  if (!GsymData.isValidOffsetForDataOfSize(FileOffset, FileSize))
    return createStringError(std::errc::invalid_argument,
                             "GlobalData section type %u data not available "
                             "(offset=%" PRIu64 ", size=%" PRIu64
                             ", bufsize=%" PRIu64 ")",
                             static_cast<uint32_t>(Type), FileOffset, FileSize,
                             static_cast<uint64_t>(GsymData.getData().size()));
  return GsymData.getData().substr(FileOffset, FileSize);
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
GlobalData::getBytes(DataExtractor &GsymData) const {
  auto Str = getStringRef(GsymData);
  if (!Str)
    return Str.takeError();
  return arrayRefFromStringRef(*Str);
}

llvm::Expected<GlobalData> GlobalData::decode(DataExtractor &GsymData,
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
