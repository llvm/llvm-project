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

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV2::GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer)
    : GsymReader(std::move(Buffer)), AddrInfoOffsetsData(StringRef(), true, 8),
      FileData(StringRef(), true, 8) {}

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
parseGlobalDataEntries(DataExtractor &DE, uint64_t &Offset, uint64_t BufSize,
                       std::map<GlobalInfoType, GlobalData> &Sections) {
  while (Offset + sizeof(GlobalData) <= BufSize) {
    auto GDOrErr = GlobalData::decode(DE, Offset);
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

llvm::Error GsymReaderV2::parse() {
  const StringRef Buf = MemBuffer->getBuffer();
  const uint64_t BufSize = Buf.size();
  if (BufSize < sizeof(HeaderV2))
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
    Swap.reset(new SwappedData);
    break;
  default:
    return createStringError(std::errc::invalid_argument, "not a GSYM file");
  }

  const bool IsLittleEndian = (Endian == llvm::endianness::little);
  // Read a correctly byte swapped header if we need to.
  DataExtractor DE(Buf, IsLittleEndian, 8);
  if (Swap) {
    auto ExpectedHdr = HeaderV2::decode(DE);
    if (!ExpectedHdr)
      return ExpectedHdr.takeError();
    Swap->Hdr = *ExpectedHdr;
    Hdr = &Swap->Hdr;
  } else {
    Hdr = reinterpret_cast<const HeaderV2 *>(Buf.data());
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and a valid UUID size.
  if (Error Err = Hdr->checkForError())
    return Err;

  // Parse GlobalData entries to find section locations.
  uint64_t Offset = sizeof(HeaderV2);
  if (auto Err =
          parseGlobalDataEntries(DE, Offset, BufSize, GlobalDataSections))
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

  if (AddrInfoOffsetsGD.FileSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrInfoOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrInfoOffsets section size mismatch");

  // AddrOffsets
  if (!Swap) {
    llvm::Expected<llvm::ArrayRef<uint8_t>> Bytes = AddrOffsetsGD.getBytes(DE);
    if (!Bytes)
      return Bytes.takeError();
    AddrOffsets = *Bytes;
  } else {
    // Do the byte-swapping for the AddrOffsets section for byte size 1-8.
    uint64_t AOff = AddrOffsetsGD.FileOffset;
    const size_t TotalBytes =
        static_cast<size_t>(Hdr->NumAddresses) * Hdr->AddrOffSize;
    Swap->AddrOffsets.resize(TotalBytes);
    for (uint32_t I = 0; I < Hdr->NumAddresses; ++I) {
      uint64_t Val = DE.getUnsigned(&AOff, Hdr->AddrOffSize);
      memcpy(Swap->AddrOffsets.data() + I * Hdr->AddrOffSize, &Val,
             Hdr->AddrOffSize);
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);
  }

  // AddrInfoOffsets
  {
    Expected<StringRef> Data = AddrInfoOffsetsGD.getStringRef(DE);
    if (!Data)
      return Data.takeError();
    // The above getStringRef() already returns the correct data range.
    // DataExtractor will ensure that accesses are within the range.
    AddrInfoOffsetsData = DataExtractor(*Data, IsLittleEndian, 8);
  }

  // String table
  {
    Expected<StringRef> Data = StringTableGD.getStringRef(DE);
    if (!Data)
      return Data.takeError();
    StrTab.Data = *Data;
  }

  // File table
  //
  // Validate and cache file table metadata for on-demand decoding via
  // getFile(). The file table has variable-width Dir/Base fields (StrpSize),
  // so entries are decoded on access rather than pre-parsed.
  {
    Expected<StringRef> Data = FileTableGD.getStringRef(DE);
    if (!Data)
      return Data.takeError();
    DataExtractor FileTableDE(*Data, IsLittleEndian, 8);
    uint64_t Offset = 0;
    uint32_t NumFiles = FileTableDE.getU32(&Offset);
    uint64_t EntriesSize = static_cast<uint64_t>(NumFiles) * 2 * Hdr->StrpSize;
    if (Data->size() < 4 + EntriesSize)
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small for %u files",
                               NumFiles);
    FileData =
        DataExtractor(Data->substr(Offset, EntriesSize), IsLittleEndian, 8);
    FileData.setStringOffsetSize(Hdr->StrpSize);
  }
  return Error::success();
}

const HeaderV2 &GsymReaderV2::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

std::optional<uint64_t> GsymReaderV2::getAddress(size_t Index) const {
  std::optional<uint64_t> AddressOffset =
      getUnsigned(AddrOffsets, getAddressOffsetByteSize(), Index);
  if (!AddressOffset)
    return std::nullopt;
  return *AddressOffset + getBaseAddress();
}

Expected<uint64_t> GsymReaderV2::getAddressIndex(const uint64_t Addr) const {
  const uint64_t BaseAddress = getBaseAddress();
  if (Addr < BaseAddress)
    return createStringError(std::errc::invalid_argument,
                             "address 0x%" PRIx64 " is not in GSYM", Addr);

  const uint64_t AddrOffset = Addr - BaseAddress;
  const uint8_t ByteSize = getAddressOffsetByteSize();
  const size_t NumAddrs = getNumAddresses();
  AddrOffsetIterator Begin(AddrOffsets, ByteSize, 0);
  AddrOffsetIterator End(AddrOffsets, ByteSize, NumAddrs);
  auto Iter = std::lower_bound(Begin, End, AddrOffset);

  // Watch for addresses that fall between the base address and the first
  // address offset.
  if (Iter == Begin && AddrOffset < *Begin)
    return createStringError(std::errc::invalid_argument,
                             "address 0x%" PRIx64 " is not in GSYM", Addr);
  if (Iter == End || AddrOffset < *Iter)
    --Iter;

  // GSYM files have sorted function infos with the most information (line
  // table and/or inline info) first in the array of function infos, so
  // always backup as much as possible as long as the address offset is the
  // same as the previous entry.
  while (Iter != Begin) {
    auto Prev = Iter - 1;
    if (*Prev == *Iter)
      Iter = Prev;
    else
      break;
  }

  return Iter.getIndex();
}

std::optional<FileEntry> GsymReaderV2::getFile(uint32_t Index) const {
  uint64_t Offset = Index * 2 * FileData.getStringOffsetSize();
  // If the offsset is beyond the end of the file table, the given index is out
  // of range.
  if (!FileData.isValidOffsetForDataOfSize(Offset,
                                           2 * FileData.getStringOffsetSize()))
    return std::nullopt;
  FileEntry FE;
  FE.Dir = FileData.getStringOffset(&Offset);
  FE.Base = FileData.getStringOffset(&Offset);
  return FE;
}

uint64_t GsymReaderV2::getAddressInfoOffset(size_t Index) const {
  uint64_t Offset = Index * getAddressInfoOffsetByteSize();
  uint64_t RelOff =
      AddrInfoOffsetsData.getUnsigned(&Offset, getAddressInfoOffsetByteSize());
  return RelOff +
         GlobalDataSections.at(GlobalInfoType::FunctionInfo).FileOffset;
}

void GsymReaderV2::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET" << format("%-2u", getAddressOffsetByteSize() * 8)
     << " (ADDRESS)\n";
  OS << "====== =============================== \n";
  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << format("[%4u] ", I);
    auto AddrOff = getUnsigned(AddrOffsets, getAddressOffsetByteSize(), I);
    OS << format_hex(AddrOff.value_or(0), getAddressOffsetByteSize() * 2 + 2);
    OS << " (" << HEX64(*getAddress(I)) << ")\n";
  }
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << format("[%4u] ", I) << HEX32(getAddressInfoOffset(I)) << "\n";
  }
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== ==============================\n";
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
  OS << "\n" << StrTab << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << "FunctionInfo @ " << HEX32(getAddressInfoOffset(I)) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
