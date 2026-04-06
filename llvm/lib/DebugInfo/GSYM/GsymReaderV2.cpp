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
#include <map>

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV2::GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer)
    : GsymReader(std::move(Buffer)) {}

GsymReaderV2::GsymReaderV2(GsymReaderV2 &&RHS) = default;
GsymReaderV2::~GsymReaderV2() = default;

llvm::Expected<GsymReaderV2> GsymReaderV2::openFile(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  auto Err = BuffOrErr.getError();
  if (Err)
    return llvm::errorCodeToError(Err);
  return create(BuffOrErr.get());
}

llvm::Expected<GsymReaderV2> GsymReaderV2::copyBuffer(StringRef Bytes) {
  auto MB = MemoryBuffer::getMemBufferCopy(Bytes, "GSYM bytes");
  return create(MB);
}

llvm::Expected<GsymReaderV2>
GsymReaderV2::create(std::unique_ptr<MemoryBuffer> &MB) {
  if (!MB)
    return createStringError(std::errc::invalid_argument,
                             "invalid memory buffer");
  GsymReaderV2 GR(std::move(MB));
  if (auto Err = GR.parse())
    return std::move(Err);
  return std::move(GR);
}

/// Helper to parse GlobalData entries from a GSYM V2 file.
static llvm::Error
parseGlobalDataEntries(DataExtractor &Data, uint64_t &Offset, uint64_t BufSize,
                       std::map<GlobalInfoType, GlobalData> &Sections) {
  while (Offset + sizeof(GlobalData) <= BufSize) {
    auto GDOrErr = GlobalData::decode(Data, Offset);
    if (!GDOrErr)
      return GDOrErr.takeError();
    const GlobalData &GD = *GDOrErr;

    if (GD.Type == GlobalInfoType::EndOfList)
      return Error::success();

    if (GD.FileOffset + GD.FileSize > BufSize)
      return createStringError(
          std::errc::invalid_argument,
          "GlobalData section type %u extends beyond "
          "buffer (offset=%" PRIu64 ", size=%" PRIu64 ", bufsize=%" PRIu64 ")",
          static_cast<uint32_t>(GD.Type), GD.FileOffset, GD.FileSize, BufSize);

    Sections[GD.Type] = GD;
  }
  return createStringError(std::errc::invalid_argument,
                           "GlobalData array not terminated by EndOfList");
}

/// For V2 file layout, see HeaderV2.h
llvm::Error GsymReaderV2::parse() {
  const StringRef Buf = MemBuffer->getBuffer();
  const uint64_t BufSize = Buf.size();
  if (BufSize < HeaderV2::getEncodedSize())
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM V2 header");
  const auto HostByteOrder = llvm::endianness::native;

  // Check for the magic bytes. This file format is designed to be mmap'ed
  // into a process and accessed as read only. This is done for performance
  // and efficiency for symbolicating and parsing GSYM data.
  uint32_t Magic;
  memcpy(&Magic, Buf.data(), 4);

  switch (Magic) {
  case GSYM_MAGIC:
    Endian = HostByteOrder;
    break;
  case GSYM_CIGAM:
    // This is a GSYM file, but not native endianness.
    Endian =
        sys::IsBigEndianHost ? llvm::endianness::little : llvm::endianness::big;
    SwappedHdr = std::make_unique<HeaderV2>();
    break;
  default:
    return createStringError(std::errc::invalid_argument, "not a GSYM file");
  }

  const bool IsLittleEndian = (Endian == llvm::endianness::little);
  // Read a correctly byte swapped header if we need to.
  DataExtractor Data(Buf, IsLittleEndian, 8);
  if (SwappedHdr) {
    auto ExpectedHdr = HeaderV2::decode(Data);
    if (!ExpectedHdr)
      return ExpectedHdr.takeError();
    *SwappedHdr = *ExpectedHdr;
    Hdr = SwappedHdr.get();
  } else {
    Hdr = reinterpret_cast<const HeaderV2 *>(Buf.data());
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and string table encoding.
  if (Error Err = Hdr->checkForError())
    return Err;

  // Parse GlobalData entries to find section locations.
  uint64_t Offset = HeaderV2::getEncodedSize();
  if (auto Err =
          parseGlobalDataEntries(Data, Offset, BufSize, GlobalDataSections))
    return Err;

  for (auto Type :
       {GlobalInfoType::AddrOffsets, GlobalInfoType::AddrInfoOffsets,
        GlobalInfoType::StringTable, GlobalInfoType::FileTable,
        GlobalInfoType::FunctionInfo})
    if (!GlobalDataSections.count(Type))
      return createStringError(std::errc::invalid_argument,
                               "missing required section type %u",
                               static_cast<uint32_t>(Type));

  const GlobalData &AddrOffsetsGD =
      GlobalDataSections[GlobalInfoType::AddrOffsets];
  const GlobalData &AddrInfoOffsetsGD =
      GlobalDataSections[GlobalInfoType::AddrInfoOffsets];
  const GlobalData &StringTableGD =
      GlobalDataSections[GlobalInfoType::StringTable];
  const GlobalData &FileTableGD = GlobalDataSections[GlobalInfoType::FileTable];

  if (AddrOffsetsGD.FileSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrOffsets section size mismatch");

  if (AddrInfoOffsetsGD.FileSize != static_cast<uint64_t>(Hdr->NumAddresses) *
                                        HeaderV2::getAddressInfoOffsetSize())
    return createStringError(std::errc::invalid_argument,
                             "AddrInfoOffsets section size mismatch");

  // AddrOffsets
  {
    uint64_t AddrOffsetsOff = AddrOffsetsGD.FileOffset;
    if (auto Err =
            parseAddrOffsets(Data, AddrOffsetsOff, SwappedHdr != nullptr))
      return Err;
  }

  // AddrInfoOffsets
  {
    Expected<StringRef> Bytes = AddrInfoOffsetsGD.getStringRef(Data);
    if (!Bytes)
      return Bytes.takeError();
    // The above getStringRef() already returns the correct data range.
    // DataExtractor will ensure that accesses are within the range.
    AddrInfoOffsetsData = DataExtractor(*Bytes, IsLittleEndian, 8);
  }

  // String table
  {
    Expected<StringRef> Bytes = StringTableGD.getStringRef(Data);
    if (!Bytes)
      return Bytes.takeError();
    StrTab.Data = *Bytes;
  }

  // File table
  {
    Expected<StringRef> Bytes = FileTableGD.getStringRef(Data);
    if (!Bytes)
      return Bytes.takeError();
    DataExtractor FileTableDE(*Bytes, IsLittleEndian, 8);
    uint64_t FTOffset = 0;
    if (auto Err = parseFileTable(FileTableDE, FTOffset))
      return Err;
  }
  return Error::success();
}

const HeaderV2 &GsymReaderV2::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

uint64_t GsymReaderV2::getAddressInfoOffset(size_t Index) const {
  return GsymReader::getAddressInfoOffset(Index) +
         GlobalDataSections.at(GlobalInfoType::FunctionInfo).FileOffset;
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
  }
  return "Unknown";
}

void GsymReaderV2::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";

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
  for (uint32_t I = 0; I < getNumAddresses(); ++I)
    OS << format("[%4u] ", I) << HEX64(GsymReader::getAddressInfoOffset(I))
       << " (" << HEX64(getAddressInfoOffset(I)) << ")\n";
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
