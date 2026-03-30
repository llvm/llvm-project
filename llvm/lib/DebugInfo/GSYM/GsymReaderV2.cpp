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
      return createStringError(std::errc::invalid_argument,
                               "GlobalData section type %u extends beyond "
                               "buffer (offset=%" PRIu64 ", size=%" PRIu64
                               ", bufsize=%" PRIu64 ")",
                               static_cast<uint32_t>(GD.Type), GD.FileOffset,
                               GD.FileSize, BufSize);

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
    Endian = sys::IsBigEndianHost ? llvm::endianness::little
                                  : llvm::endianness::big;
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
  if (auto Err = parseGlobalDataEntries(DE, Offset, BufSize,
                                        GlobalDataSections))
    return Err;

  for (auto Type : {GlobalInfoType::AddrOffsets, GlobalInfoType::AddrInfoOffsets,
                     GlobalInfoType::StringTable, GlobalInfoType::FileTable,
                     GlobalInfoType::FunctionInfo})
    if (!GlobalDataSections.count(Type))
      return createStringError(std::errc::invalid_argument,
                               "missing required section type %u",
                               static_cast<uint32_t>(Type));

  const GlobalData &AddrOffsetsGD = GlobalDataSections[GlobalInfoType::AddrOffsets];
  const GlobalData &AddrInfoOffsetsGD = GlobalDataSections[GlobalInfoType::AddrInfoOffsets];
  const GlobalData &StringTableGD = GlobalDataSections[GlobalInfoType::StringTable];
  const GlobalData &FileTableGD = GlobalDataSections[GlobalInfoType::FileTable];

  if (AddrOffsetsGD.FileSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrOffsets section size mismatch");

  if (AddrInfoOffsetsGD.FileSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrInfoOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrInfoOffsets section size mismatch");

  // Set up the data references for various GlobalData sections.
  // Handles both swapped and non-swapped cases.
  if (!Swap) {
    AddrOffsets = ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(Buf.data() +
                                          AddrOffsetsGD.FileOffset),
        AddrOffsetsGD.FileSize);

    AddrInfoOffsets = ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(Buf.data() +
                                          AddrInfoOffsetsGD.FileOffset),
        AddrInfoOffsetsGD.FileSize);

    if (FileTableGD.FileSize < 4)
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small");
    uint32_t NumFiles;
    memcpy(&NumFiles, Buf.data() + FileTableGD.FileOffset, 4);
    if (FileTableGD.FileSize < 4 + NumFiles * sizeof(FileEntry))
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small for %u files",
                               NumFiles);
    Files = ArrayRef<FileEntry>(
        reinterpret_cast<const FileEntry *>(Buf.data() +
                                            FileTableGD.FileOffset + 4),
        NumFiles);

    StrTab.Data = Buf.substr(StringTableGD.FileOffset,
                             StringTableGD.FileSize);
  } else {
    // Do the byte-swapping for the AddrOffsets section for byte size 1-8.
    {
      uint64_t AOff = AddrOffsetsGD.FileOffset;
      const size_t TotalBytes =
          static_cast<size_t>(Hdr->NumAddresses) * Hdr->AddrOffSize;
      Swap->AddrOffsets.resize(TotalBytes);
      for (uint32_t I = 0; I < Hdr->NumAddresses; ++I) {
        uint64_t Val = DE.getUnsigned(&AOff, Hdr->AddrOffSize);
        memcpy(Swap->AddrOffsets.data() + I * Hdr->AddrOffSize,
               &Val, Hdr->AddrOffSize);
      }
      AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);
    }

    // Do the byte-swapping for the AddrInfoOffsets section for byte size 1-8.
    {
      uint64_t AIOff = AddrInfoOffsetsGD.FileOffset;
      const size_t TotalBytes =
          static_cast<size_t>(Hdr->NumAddresses) * Hdr->AddrInfoOffSize;
      Swap->AddrInfoOffsets.resize(TotalBytes);
      for (uint32_t I = 0; I < Hdr->NumAddresses; ++I) {
        uint64_t Val = DE.getUnsigned(&AIOff, Hdr->AddrInfoOffSize);
        memcpy(Swap->AddrInfoOffsets.data() + I * Hdr->AddrInfoOffSize,
               &Val, Hdr->AddrInfoOffSize);
      }
      AddrInfoOffsets = ArrayRef<uint8_t>(Swap->AddrInfoOffsets);
    }

    uint64_t FTOff = FileTableGD.FileOffset;
    uint32_t NumFiles = DE.getU32(&FTOff);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (!DE.getU32(&FTOff, &Swap->Files[0].Dir, NumFiles * 2))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
      Files = ArrayRef<FileEntry>(Swap->Files);
    }

    StrTab.Data = Buf.substr(StringTableGD.FileOffset,
                             StringTableGD.FileSize);
  }
  return Error::success();
}

const HeaderV2 &GsymReaderV2::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

std::optional<uint64_t> GsymReaderV2::getAddressInfoOffset(size_t Index) const {
  auto RelOff = getUnsigned(AddrInfoOffsets, getAddressInfoOffsetByteSize(),
                            Index);
  if (!RelOff)
    return std::nullopt;
  return *RelOff +
         GlobalDataSections.at(GlobalInfoType::FunctionInfo).FileOffset;
}

void GsymReaderV2::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET" << format("%-2u", getAddressOffsetByteSize() * 8) << " (ADDRESS)\n";
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
    auto Off = getAddressInfoOffset(I);
    OS << format("[%4u] ", I) << HEX32(Off.value_or(0)) << "\n";
  }
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== ==============================\n";
  for (uint32_t I = 0; I < Files.size(); ++I) {
    OS << format("[%4u] ", I) << HEX32(Files[I].Dir) << ' '
       << HEX32(Files[I].Base) << ' ';
    dump(OS, getFile(I));
    OS << "\n";
  }
  OS << "\n" << StrTab << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    auto Off = getAddressInfoOffset(I);
    OS << "FunctionInfo @ " << HEX32(Off.value_or(0)) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
