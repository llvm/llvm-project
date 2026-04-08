//===- GsymReaderV1.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReaderV1.h"

#include <assert.h>
#include <inttypes.h>

#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV1::GsymReaderV1(std::unique_ptr<MemoryBuffer> Buffer,
                           llvm::endianness Endian)
    : GsymReader(std::move(Buffer), Endian) {}

llvm::Error GsymReaderV1::parseHeaderAndGlobalDataDirectory() {
  const StringRef Buf = MemBuffer->getBuffer();
  BinaryStreamReader FileData(Buf, llvm::endianness::native);
  // Read the header. This file format is designed to be mmap'ed into a process
  // and accessed as read only.
  if (FileData.readObject(Hdr))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM header");

  const bool IsLittleEndian = (Endian == llvm::endianness::little);
  // Read a correctly byte swapped header if we need to.
  if (Endian != llvm::endianness::native) {
    SwappedHdr = std::make_unique<Header>();
    DataExtractor Data(Buf, IsLittleEndian, 4);
    if (auto ExpectedHdr = Header::decode(Data))
      *SwappedHdr = ExpectedHdr.get();
    else
      return ExpectedHdr.takeError();
    Hdr = SwappedHdr.get();
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and a valid UUID size.
  if (Error Err = Hdr->checkForError())
    return Err;

  // Compute section offsets from the fixed V1 layout and populate the
  // GlobalDataSections map. V1 sections are laid out sequentially:
  //   [Header] [AddrOffsets] [AddrInfoOffsets] [FileTable] ... [StringTable]
  const uint64_t NumAddrs = Hdr->NumAddresses;
  const uint8_t AddrOffSize = Hdr->AddrOffSize;

  // AddrOffsets
  uint64_t Offset = alignTo(sizeof(Header), AddrOffSize);
  uint64_t AddrOffsetsSize = NumAddrs * AddrOffSize;
  GlobalDataSections[GlobalInfoType::AddrOffsets] = {
      GlobalInfoType::AddrOffsets, Offset, AddrOffsetsSize};
  Offset += AddrOffsetsSize;

  // AddrInfoOffsets
  Offset = alignTo(Offset, 4);
  uint64_t AddrInfoOffsetsSize = NumAddrs * Header::getAddressInfoOffsetSize();
  GlobalDataSections[GlobalInfoType::AddrInfoOffsets] = {
      GlobalInfoType::AddrInfoOffsets, Offset, AddrInfoOffsetsSize};
  Offset += AddrInfoOffsetsSize;

  // FileTable: read NumFiles to compute the size.
  DataExtractor Data(Buf, IsLittleEndian, 4);
  uint64_t FTOffset = Offset;
  uint32_t NumFiles = Data.getU32(&FTOffset);
  uint64_t FileTableSize =
      4 + static_cast<uint64_t>(NumFiles) *
              FileEntry::getEncodedSize(Header::getStringOffsetSize());
  GlobalDataSections[GlobalInfoType::FileTable] = {
      GlobalInfoType::FileTable, Offset, FileTableSize};

  // StringTable: offset and size are in the header.
  GlobalDataSections[GlobalInfoType::StringTable] = {
      GlobalInfoType::StringTable, Hdr->StrtabOffset, Hdr->StrtabSize};

  // FunctionInfo: V1 uses absolute file offsets in AddrInfoOffsets (not
  // relative to a FunctionInfo section start). FileOffset is set to 0 so that
  // GsymReader::getAddressInfoOffset() works automatically. Correspondingly,
  // FileSize is set to the length of the entire file for range checks.
  GlobalDataSections[GlobalInfoType::FunctionInfo] = {
      GlobalInfoType::FunctionInfo, 0, Buf.size()};

  return Error::success();
}

void GsymReaderV1::dump(raw_ostream &OS) {
  OS << *Hdr << "\n";
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET";

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
  OS << " (ADDRESS)\n";
  OS << "====== =============================== \n";
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
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < getNumAddresses(); ++I)
    OS << format("[%4u] ", I) << HEX32(getAddressInfoOffset(I)) << "\n";
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== ==============================\n";
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
  gsym::dump(OS, StrTab, 4);
  OS << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << "FunctionInfo @ " << HEX32(getAddressInfoOffset(I)) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
