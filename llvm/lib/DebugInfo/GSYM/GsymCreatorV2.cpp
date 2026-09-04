//===- GsymCreatorV2.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymCreatorV2.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

using namespace llvm;
using namespace gsym;

uint64_t GsymCreatorV2::calculateHeaderAndTableSize() const {
  const uint64_t HeaderSize = HeaderV2::getEncodedSize();
  const size_t NumFuncs = Funcs.size();
  const uint32_t NumEntries = 5 + (UUID.empty() ? 0 : 1) + 1;
  uint64_t Size = HeaderSize + NumEntries * 20;
  Size = llvm::alignTo(Size, getAddressOffsetSize());
  Size += NumFuncs * getAddressOffsetSize();
  Size = llvm::alignTo(Size, HeaderV2::getAddressInfoOffsetSize());
  Size += NumFuncs * HeaderV2::getAddressInfoOffsetSize();
  Size = llvm::alignTo(Size, 4);
  Size += 4 + Files.size() * FileEntry::getEncodedSize(getStringOffsetSize());
  Size += StrTab.getSize();
  Size += UUID.size();
  return Size;
}

/// For V2 file layout, see HeaderV2.h
llvm::Error GsymCreatorV2::encode(FileWriter &O) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  std::optional<uint64_t> BaseAddr;
  if (auto Err = validateForEncoding(BaseAddr))
    return Err;

  const uint8_t AddrOffSize = getAddressOffsetSize();

  // Pre-encode all FunctionInfo objects into a temporary buffer so we know the
  // total FunctionInfo section size and each function's offset within it.
  SmallVector<char, 0> FIBuf;
  raw_svector_ostream FIOS(FIBuf);
  FileWriter FIFW(FIOS, O.getByteOrder());
  FIFW.setStringOffsetSize(getStringOffsetSize());
  std::vector<uint64_t> FIRelativeOffsets;
  for (const auto &FI : Funcs) {
    if (auto OffOrErr = FI.encode(FIFW))
      FIRelativeOffsets.push_back(*OffOrErr);
    else
      return OffOrErr.takeError();
  }
  const uint64_t FISectionSize = FIBuf.size();
  const uint64_t StringTableSize = StrTab.getSize();

  const uint8_t StrpSize = 8;

  const bool HasUUID = !UUID.empty();
  const uint32_t NumGlobalDataEntries = 5 + (HasUUID ? 1 : 0) + 1;
  const uint64_t GlobalDataArraySize =
      static_cast<uint64_t>(NumGlobalDataEntries) * 20;

  const uint64_t HeaderSize = HeaderV2::getEncodedSize();
  uint64_t CurOffset = HeaderSize + GlobalDataArraySize;

  // UUID section (first, no alignment requirement).
  const uint64_t UUIDOffset = CurOffset;
  const uint64_t UUIDSectionSize = UUID.size();
  if (HasUUID)
    CurOffset += UUIDSectionSize;

  // AddrOffsets section.
  CurOffset = llvm::alignTo(CurOffset, AddrOffSize);
  const uint64_t AddrOffsetsOffset = CurOffset;
  const uint64_t AddrOffsetsSize = Funcs.size() * AddrOffSize;
  CurOffset += AddrOffsetsSize;

  // AddrInfoOffsets section.
  const uint8_t AddrInfoOffSize = 8;
  CurOffset = llvm::alignTo(CurOffset, AddrInfoOffSize);
  const uint64_t AddrInfoOffsetsOffset = CurOffset;
  const uint64_t AddrInfoOffsetsSize = Funcs.size() * AddrInfoOffSize;
  CurOffset += AddrInfoOffsetsSize;

  // FileTable section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FileTableOffset = CurOffset;
  const uint64_t FileTableSize =
      4 + Files.size() * FileEntry::getEncodedSize(StrpSize);
  CurOffset += FileTableSize;

  // StringTable section.
  const uint64_t StringTableOffset = CurOffset;
  CurOffset += StringTableSize;

  // FunctionInfo section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FISectionOffset = CurOffset;
  CurOffset += FISectionSize;

  // Build and write the header.
  HeaderV2 Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = HeaderV2::getVersion();
  Hdr.BaseAddress = *BaseAddr;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.AddrOffSize = AddrOffSize;
  Hdr.StrTableEncoding = StringTableEncoding::Default;
  if (auto Err = Hdr.encode(O))
    return Err;

  // Write GlobalData entries.
  if (HasUUID)
    GlobalData{GlobalInfoType::UUID, UUIDOffset, UUIDSectionSize}.encode(O);
  GlobalData{GlobalInfoType::AddrOffsets, AddrOffsetsOffset, AddrOffsetsSize}
      .encode(O);
  GlobalData{GlobalInfoType::AddrInfoOffsets, AddrInfoOffsetsOffset,
             AddrInfoOffsetsSize}
      .encode(O);
  GlobalData{GlobalInfoType::FileTable, FileTableOffset, FileTableSize}.encode(
      O);
  GlobalData{GlobalInfoType::StringTable, StringTableOffset, StringTableSize}
      .encode(O);
  GlobalData{GlobalInfoType::FunctionInfo, FISectionOffset, FISectionSize}
      .encode(O);
  GlobalData{GlobalInfoType::EndOfList, 0, 0}.encode(O);

  // Write UUID section.
  if (HasUUID) {
    assert(O.tell() == UUIDOffset);
    O.writeData(ArrayRef<uint8_t>(UUID.data(), UUID.size()));
  }

  // Write AddrOffsets section.
  O.alignTo(AddrOffSize);
  assert(O.tell() == AddrOffsetsOffset);
  encodeAddrOffsets(O, AddrOffSize, *BaseAddr);

  // Write AddrInfoOffsets section. Values are relative to FunctionInfo section.
  O.alignTo(AddrInfoOffSize);
  assert(O.tell() == AddrInfoOffsetsOffset);
  for (uint64_t RelOff : FIRelativeOffsets)
    O.writeU64(RelOff);

  // Write FileTable section.
  O.alignTo(4);
  assert(O.tell() == FileTableOffset);
  if (auto Err = encodeFileTable(O))
    return Err;

  // Write StringTable section.
  assert(O.tell() == StringTableOffset);
  StrTab.write(O.get_stream());

  // Write FunctionInfo section.
  O.alignTo(4);
  assert(O.tell() == FISectionOffset);
  O.writeData(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(FIBuf.data()),
                                FIBuf.size()));

  return Error::success();
}
