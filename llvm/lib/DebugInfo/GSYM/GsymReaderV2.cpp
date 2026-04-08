//===- GsymReaderV2.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"

#include <assert.h>
#include <inttypes.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV2::GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer,
                           llvm::endianness Endian)
    : GsymReader(std::move(Buffer), Endian) {}

/// For V2 file layout, see HeaderV2.h
llvm::Error GsymReaderV2::parseHeaderAndGlobalDataDirectory() {
  if (auto Err = parseHeader(Hdr, SwappedHdr))
    return Err;
  return parseGlobalDataEntries(HeaderV2::getEncodedSize());
}

static const char *getGlobalInfoTypeName(GlobalInfoType Type) {
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
  case GlobalInfoType::NumTypes:
    break;
  }
  return "Unknown";
}

void GsymReaderV2::dump(raw_ostream &OS) {
  OS << *Hdr << "\n";

  // Print GlobalData directory sorted by file offset.
  std::vector<GlobalData> Sections;
  for (const auto &[Type, GD] : GlobalDataSections)
    Sections.push_back(GD);
  llvm::sort(Sections, [](const GlobalData &A, const GlobalData &B) {
    return A.FileOffset < B.FileOffset;
  });
  OS << "Global Data Directory:\n";
  OS << "TYPE            FILE OFFSET 64      FILE SIZE 64\n";
  OS << "=============== ==================  ==================\n";
  for (const auto &GD : Sections) {
    OS << format("%-15s ", getGlobalInfoTypeName(GD.Type))
       << HEX64(GD.FileOffset) << "  " << HEX64(GD.FileSize) << "\n";
  }
  OS << "\n";

  // Print UUID if present.
  auto UUIDIt = GlobalDataSections.find(GlobalInfoType::UUID);
  if (UUIDIt != GlobalDataSections.end()) {
    const GlobalData &UUIDGD = UUIDIt->second;
    const StringRef Buf = MemBuffer->getBuffer();
    OS << "UUID:\n";
    for (uint64_t I = 0; I < UUIDGD.FileSize; ++I)
      OS << format_hex_no_prefix(
          static_cast<uint8_t>(Buf[UUIDGD.FileOffset + I]), 2);
    OS << "\n\n";
  }

  OS << "Address Table:\n";
  OS << "INDEX  OFFSET ";
  switch (getAddressOffsetSize()) {
  case 1:
    OS << "8 ";
    break;
  case 2:
    OS << "16";
    break;
  case 4:
    OS << "32";
    break;
  case 8:
    OS << "64";
    break;
  default:
    OS << "??";
    break;
  }
  OS << " (ADDRESS 64)\n";
  OS << "====== ========================================\n";
  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << format("[%4u] ", I);
    switch (getAddressOffsetSize()) {
    case 1:
      OS << HEX8(getAddrOffsets<uint8_t>()[I]);
      break;
    case 2:
      OS << HEX16(getAddrOffsets<uint16_t>()[I]);
      break;
    case 4:
      OS << HEX32(getAddrOffsets<uint32_t>()[I]);
      break;
    case 8:
      OS << HEX32(getAddrOffsets<uint64_t>()[I]);
      break;
    default:
      break;
    }
    OS << " (" << HEX64(*getAddress(I)) << ")\n";
  }
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  OFFSET 64 (FILE OFFSET 64)\n";
  OS << "====== ========================================\n";
  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    uint64_t RelOffset = I * getAddressInfoOffsetSize();
    uint64_t RelValue =
        AddrInfoOffsetsData.getUnsigned(&RelOffset, getAddressInfoOffsetSize());
    OS << format("[%4u] ", I) << HEX64(RelValue) << " ("
       << HEX64(getAddressInfoOffset(I)) << ")\n";
  }
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== "
        "========================================\n";
  // Since we don't store the total number of files in the file table, loop
  // until we get a null entry which means the index is out of range.
  for (uint32_t I = 0;; ++I) {
    auto FE = getFile(I);
    if (!FE)
      break;
    OS << format("[%4u] ", I) << HEX32(FE->Dir) << ' ' << HEX32(FE->Base)
       << ' ';
    dump(OS, FE);
    OS << "\n";
  }
  OS << "\n";
  gsym::dump(OS, StrTab, 8);
  OS << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << "FunctionInfo @ " << HEX32(getAddressInfoOffset(I)) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
