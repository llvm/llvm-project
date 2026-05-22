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
#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV2::GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer,
                           llvm::endianness Endian)
    : GsymReader(std::move(Buffer), Endian) {}

/// For V2 file layout, see HeaderV2.h
llvm::Error GsymReaderV2::parseHeaderAndGlobalDataEntries() {
  if (auto Err = parseHeader(Hdr, SwappedHdr))
    return Err;
  return parseGlobalDataEntries(HeaderV2::getEncodedSize());
}

void GsymReaderV2::dump(raw_ostream &OS) {
  OS << *Hdr << "\n";

  // Print GlobalData entries.
  OS << "Global Data Sections:\n";
  OS << "TYPE            FILE OFFSET         FILE SIZE\n";
  OS << "=============== ==================  ==================\n";
  /// Re-parse the GlobalData entries to ensure we show the GlobalData
  /// in the exact order it appears in the GSYM data.
  const StringRef Buf = MemBuffer->getBuffer();
  const uint64_t BufSize = Buf.size();
  GsymDataExtractor Data(Buf, isLittleEndian());
  uint64_t Offset = HeaderV2::getEncodedSize();
  while (Offset + sizeof(GlobalData) <= BufSize) {
    auto GDOrErr = GlobalData::decode(Data, Offset);
    assert(GDOrErr && "GlobalData::decode() should not fail");
    const GlobalData &GD = *GDOrErr;

    OS << formatv("{0,-15} ", getNameForGlobalInfoType(GD.Type).data())
       << HEX64(GD.FileOffset) << "  " << HEX64(GD.FileSize) << "\n";

    // Stop printing after the end of list entry.
    if (GD.Type == GlobalInfoType::EndOfList)
      break;
  }
  OS << "\n";

  // Print UUID if present.
  if (auto UUIDBytes = getOptionalGlobalDataBytes(GlobalInfoType::UUID)) {
    OS << "UUID:\n";
    for (uint8_t Byte : *UUIDBytes)
      OS << format_hex_no_prefix(Byte, 2);
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
    OS << formatv("[{0,4}] ", I);
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
    OS << formatv("[{0,4}] ", I) << HEX64(RelValue) << " ("
       << HEX64(*getAddressInfoOffset(I)) << ")\n";
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
    OS << formatv("[{0,4}] ", I) << HEX32(FE->Dir) << ' ' << HEX32(FE->Base)
       << ' ';
    dump(OS, FE);
    OS << "\n";
  }
  OS << "\n";
  gsym::dump(OS, StrTab, 8);
  OS << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << "FunctionInfo @ " << HEX32(*getAddressInfoOffset(I)) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
