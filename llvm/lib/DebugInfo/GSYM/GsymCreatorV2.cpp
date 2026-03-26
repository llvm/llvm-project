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

std::unique_ptr<GsymCreator> GsymCreatorV2::createNew(bool Quiet) const {
  return std::make_unique<GsymCreatorV2>(Quiet);
}

uint64_t GsymCreatorV2::calculateHeaderAndTableSize() const {
  const uint64_t HeaderSize = sizeof(HeaderV2);
  const size_t NumFuncs = Funcs.size();
  const uint32_t NumEntries = 5 + (UUID.empty() ? 0 : 1) + 1;
  uint64_t Size = HeaderSize + NumEntries * 24;
  Size = llvm::alignTo(Size, getAddressOffsetSize());
  Size += NumFuncs * getAddressOffsetSize();
  Size = llvm::alignTo(Size, 4);
  Size += NumFuncs * 4;
  Size = llvm::alignTo(Size, 4);
  Size += 4 + Files.size() * sizeof(FileEntry);
  Size += StrTab.getSize();
  Size += UUID.size();
  return Size;
}

llvm::Error GsymCreatorV2::loadCallSitesFromYAML(StringRef YAMLFile) {
  return createStringError(std::errc::not_supported,
                           "call site loading not yet supported in V2");
}

/// Write a single GlobalData entry to the output stream.
static void writeGlobalDataEntry(FileWriter &O, GlobalInfoType Type,
                                 uint64_t FileOffset, uint64_t FileSize) {
  O.writeU32(static_cast<uint32_t>(Type));
  O.writeU32(0); // Padding
  O.writeU64(FileOffset);
  O.writeU64(FileSize);
}

llvm::Error GsymCreatorV2::encode(FileWriter &O) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Funcs.empty())
    return createStringError(std::errc::invalid_argument,
                             "no functions to encode");
  if (!Finalized)
    return createStringError(std::errc::invalid_argument,
                             "GsymCreatorV2 wasn't finalized prior to encoding");
  if (Funcs.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument,
                             "too many FunctionInfos");

  std::optional<uint64_t> BaseAddr = getBaseAddress();
  if (!BaseAddr)
    return createStringError(std::errc::invalid_argument,
                             "invalid base address");

  const uint8_t AddrOffSize = getAddressOffsetSize();

  // Pre-encode all FunctionInfo objects into a temporary buffer so we know the
  // total FunctionInfo section size and each function's offset within it.
  SmallVector<char, 0> FIBuf;
  raw_svector_ostream FIOS(FIBuf);
  FileWriter FIFW(FIOS, O.getByteOrder());
  std::vector<uint64_t> FIRelativeOffsets;
  for (const auto &FI : Funcs) {
    if (auto OffOrErr = FI.encode(FIFW))
      FIRelativeOffsets.push_back(*OffOrErr);
    else
      return OffOrErr.takeError();
  }
  const uint64_t FISectionSize = FIBuf.size();
  const uint64_t StringTableSize = StrTab.getSize();

  const uint8_t StrpSize = (StringTableSize > UINT32_MAX) ? 8 : 4;

  const bool HasUUID = !UUID.empty();
  const uint32_t NumGlobalDataEntries = 5 + (HasUUID ? 1 : 0) + 1;
  const uint64_t GlobalDataArraySize =
      static_cast<uint64_t>(NumGlobalDataEntries) * 24;

  const uint64_t HeaderSize = sizeof(HeaderV2);
  uint64_t CurOffset = HeaderSize + GlobalDataArraySize;

  // AddrOffsets section.
  CurOffset = llvm::alignTo(CurOffset, AddrOffSize);
  const uint64_t AddrOffsetsOffset = CurOffset;
  const uint64_t AddrOffsetsSize = Funcs.size() * AddrOffSize;
  CurOffset += AddrOffsetsSize;

  // Determine AddrInfoOffSize.
  uint8_t AddrInfoOffSize = 4;
  {
    uint64_t Est = CurOffset;
    Est = llvm::alignTo(Est, 4);
    Est += Funcs.size() * 4;
    Est = llvm::alignTo(Est, 4);
    Est += 4 + Files.size() * sizeof(FileEntry);
    Est += StringTableSize;
    Est = llvm::alignTo(Est, 4);
    Est += FISectionSize;
    if (Est > UINT32_MAX)
      AddrInfoOffSize = 8;
  }

  // AddrInfoOffsets section.
  CurOffset = llvm::alignTo(CurOffset, AddrInfoOffSize);
  const uint64_t AddrInfoOffsetsOffset = CurOffset;
  const uint64_t AddrInfoOffsetsSize = Funcs.size() * AddrInfoOffSize;
  CurOffset += AddrInfoOffsetsSize;

  // FileTable section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FileTableOffset = CurOffset;
  const uint64_t FileTableSize = 4 + Files.size() * sizeof(FileEntry);
  CurOffset += FileTableSize;

  // StringTable section.
  const uint64_t StringTableOffset = CurOffset;
  CurOffset += StringTableSize;

  // FunctionInfo section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FISectionOffset = CurOffset;
  CurOffset += FISectionSize;

  // UUID section.
  const uint64_t UUIDOffset = CurOffset;
  const uint64_t UUIDSectionSize = UUID.size();

  // Build and write the header.
  HeaderV2 Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = GSYM_VERSION_2;
  Hdr.Padding = 0;
  Hdr.BaseAddress = *BaseAddr;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.AddrOffSize = AddrOffSize;
  Hdr.AddrInfoOffSize = AddrInfoOffSize;
  Hdr.StrpSize = StrpSize;
  Hdr.StrTableEncoding =
      static_cast<uint8_t>(StrTableEncodingType::Default);
  if (auto Err = Hdr.encode(O))
    return Err;

  // Write GlobalData entries.
  writeGlobalDataEntry(O, GlobalInfoType::AddrOffsets,
                       AddrOffsetsOffset, AddrOffsetsSize);
  writeGlobalDataEntry(O, GlobalInfoType::AddrInfoOffsets,
                       AddrInfoOffsetsOffset, AddrInfoOffsetsSize);
  writeGlobalDataEntry(O, GlobalInfoType::FileTable,
                       FileTableOffset, FileTableSize);
  writeGlobalDataEntry(O, GlobalInfoType::StringTable,
                       StringTableOffset, StringTableSize);
  writeGlobalDataEntry(O, GlobalInfoType::FunctionInfo,
                       FISectionOffset, FISectionSize);
  if (HasUUID)
    writeGlobalDataEntry(O, GlobalInfoType::UUID,
                         UUIDOffset, UUIDSectionSize);
  writeGlobalDataEntry(O, GlobalInfoType::EndOfList, 0, 0);

  // Write AddrOffsets section.
  O.alignTo(AddrOffSize);
  assert(O.tell() == AddrOffsetsOffset);
  const uint64_t MaxAddressOffset = getMaxAddressOffset();
  for (const auto &FI : Funcs) {
    uint64_t AddrOffset = FI.startAddress() - *BaseAddr;
    assert(AddrOffset <= MaxAddressOffset);
    (void)MaxAddressOffset;
    switch (AddrOffSize) {
    case 1: O.writeU8(static_cast<uint8_t>(AddrOffset)); break;
    case 2: O.writeU16(static_cast<uint16_t>(AddrOffset)); break;
    case 4: O.writeU32(static_cast<uint32_t>(AddrOffset)); break;
    case 8: O.writeU64(AddrOffset); break;
    }
  }

  // Write AddrInfoOffsets section.
  O.alignTo(AddrInfoOffSize);
  assert(O.tell() == AddrInfoOffsetsOffset);
  for (uint64_t RelOff : FIRelativeOffsets) {
    uint64_t AbsOff = FISectionOffset + RelOff;
    if (AddrInfoOffSize == 4) {
      if (AbsOff > UINT32_MAX)
        return createStringError(std::errc::invalid_argument,
                                 "addr info offset exceeded 32-bit max");
      O.writeU32(static_cast<uint32_t>(AbsOff));
    } else {
      O.writeU64(AbsOff);
    }
  }

  // Write FileTable section.
  O.alignTo(4);
  assert(O.tell() == FileTableOffset);
  assert(!Files.empty());
  assert(Files[0].Dir == 0);
  assert(Files[0].Base == 0);
  if (Files.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument, "too many files");
  O.writeU32(static_cast<uint32_t>(Files.size()));
  for (const auto &File : Files) {
    O.writeU32(File.Dir);
    O.writeU32(File.Base);
  }

  // Write StringTable section.
  assert(O.tell() == StringTableOffset);
  StrTab.write(O.get_stream());

  // Write FunctionInfo section.
  O.alignTo(4);
  assert(O.tell() == FISectionOffset);
  O.writeData(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(FIBuf.data()),
                                FIBuf.size()));

  // Write UUID section.
  if (HasUUID) {
    assert(O.tell() == UUIDOffset);
    O.writeData(ArrayRef<uint8_t>(UUID.data(), UUID.size()));
  }

  return Error::success();
}
