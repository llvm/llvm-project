//===- GsymCreatorV1.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymCreatorV1.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/Header.h"

#include <cassert>

using namespace llvm;
using namespace gsym;

uint64_t GsymCreatorV1::calculateHeaderAndTableSize() const {
  uint64_t Size = sizeof(Header);
  const size_t NumFuncs = Funcs.size();
  Size += NumFuncs * getAddressOffsetSize();
  Size += NumFuncs * sizeof(uint32_t);
  Size += Files.size() * FileEntry::getEncodedSize(getStringOffsetSize());
  Size += StrTab.getSize();
  return Size;
}

llvm::Error GsymCreatorV1::encode(FileWriter &O) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  std::optional<uint64_t> BaseAddress;
  if (auto Err = validateForEncoding(BaseAddress))
    return Err;
  Header Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = Header::getVersion();
  Hdr.AddrOffSize = getAddressOffsetSize();
  Hdr.UUIDSize = static_cast<uint8_t>(UUID.size());
  Hdr.BaseAddress = *BaseAddress;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.StrtabOffset = 0;
  Hdr.StrtabSize = 0;
  memset(Hdr.UUID, 0, sizeof(Hdr.UUID));
  if (UUID.size() > sizeof(Hdr.UUID))
    return createStringError(std::errc::invalid_argument,
                             "invalid UUID size %u", (uint32_t)UUID.size());
  if (UUID.size() > 0)
    memcpy(Hdr.UUID, UUID.data(), UUID.size());
  llvm::Error Err = Hdr.encode(O);
  if (Err)
    return Err;

  O.setStringOffsetSize(getStringOffsetSize());
  encodeAddrOffsets(O, Hdr.AddrOffSize, Hdr.BaseAddress);

  O.alignTo(4);
  const uint64_t AddrInfoOffsetsOffset = O.tell();
  for (size_t i = 0, n = Funcs.size(); i < n; ++i)
    O.writeU32(0);

  O.alignTo(4);
  if (auto Err = encodeFileTable(O))
    return Err;

  const uint64_t StrtabOffset = O.tell();
  StrTab.write(O.get_stream());
  const uint64_t StrtabSize = O.tell() - StrtabOffset;
  std::vector<uint32_t> AddrInfoOffsets;

  if (StrtabSize > UINT32_MAX) {
    return createStringError(std::errc::invalid_argument,
                             "string table size exceeded 32-bit max");
  }

  for (const auto &FuncInfo : Funcs) {
    if (Expected<uint64_t> OffsetOrErr = FuncInfo.encode(O)) {
      uint64_t Offset = OffsetOrErr.get();
      if (Offset > UINT32_MAX) {
        return createStringError(std::errc::invalid_argument,
                                 "address info offset exceeded 32-bit max");
      }
      AddrInfoOffsets.push_back(Offset);
    } else
      return OffsetOrErr.takeError();
  }
  O.fixup32((uint32_t)StrtabOffset, offsetof(Header, StrtabOffset));
  O.fixup32((uint32_t)StrtabSize, offsetof(Header, StrtabSize));

  uint64_t Offset = 0;
  for (auto AddrInfoOffset : AddrInfoOffsets) {
    O.fixup32(AddrInfoOffset, AddrInfoOffsetsOffset + Offset);
    Offset += 4;
  }
  return ErrorSuccess();
}
