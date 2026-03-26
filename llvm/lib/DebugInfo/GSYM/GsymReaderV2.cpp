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

/// Helper to parse GlobalData entries and populate section offsets/sizes.
static llvm::Error
parseGlobalDataEntries(DataExtractor &DE, uint64_t &Offset,
                       uint64_t BufSize,
                       uint64_t &AddrOffsetsOff, uint64_t &AddrOffsetsSize,
                       uint64_t &AddrInfoOffsetsOff, uint64_t &AddrInfoOffsetsSize,
                       uint64_t &StringTableOff, uint64_t &StringTableSize,
                       uint64_t &FileTableOff, uint64_t &FileTableSize,
                       uint64_t &FuncInfoOff, uint64_t &FuncInfoSize) {
  while (Offset + 24 <= BufSize) {
    auto Type = static_cast<GlobalInfoType>(DE.getU32(&Offset));
    uint32_t Pad = DE.getU32(&Offset);
    uint64_t FileOffset = DE.getU64(&Offset);
    uint64_t FileSize = DE.getU64(&Offset);
    (void)Pad;

    if (Type == GlobalInfoType::EndOfList)
      return Error::success();

    if (FileOffset + FileSize > BufSize)
      return createStringError(std::errc::invalid_argument,
                               "GlobalData section type %u extends beyond "
                               "buffer (offset=%" PRIu64 ", size=%" PRIu64
                               ", bufsize=%" PRIu64 ")",
                               static_cast<uint32_t>(Type), FileOffset,
                               FileSize, BufSize);

    switch (Type) {
    case GlobalInfoType::AddrOffsets:
      AddrOffsetsOff = FileOffset;
      AddrOffsetsSize = FileSize;
      break;
    case GlobalInfoType::AddrInfoOffsets:
      AddrInfoOffsetsOff = FileOffset;
      AddrInfoOffsetsSize = FileSize;
      break;
    case GlobalInfoType::StringTable:
      StringTableOff = FileOffset;
      StringTableSize = FileSize;
      break;
    case GlobalInfoType::FileTable:
      FileTableOff = FileOffset;
      FileTableSize = FileSize;
      break;
    case GlobalInfoType::FunctionInfo:
      FuncInfoOff = FileOffset;
      FuncInfoSize = FileSize;
      break;
    case GlobalInfoType::UUID:
      break;
    case GlobalInfoType::EndOfList:
      llvm_unreachable("handled above");
    }
  }
  return createStringError(std::errc::invalid_argument,
                           "GlobalData array not terminated by EndOfList");
}

llvm::Error GsymReaderV2::parse() {
  const StringRef Buf = MemBuffer->getBuffer();
  const uint64_t BufSize = Buf.size();

  if (BufSize < 24)
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM V2 header");

  const auto HostByteOrder = llvm::endianness::native;
  uint32_t Magic;
  memcpy(&Magic, Buf.data(), 4);

  switch (Magic) {
  case GSYM_MAGIC:
    Endian = HostByteOrder;
    break;
  case GSYM_CIGAM:
    Endian = sys::IsBigEndianHost ? llvm::endianness::little
                                  : llvm::endianness::big;
    Swap.reset(new SwappedData);
    break;
  default:
    return createStringError(std::errc::invalid_argument, "not a GSYM file");
  }

  const bool IsLittleEndian = (Endian == llvm::endianness::little);

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

  if (Error Err = Hdr->checkForError())
    return Err;

  // Populate cached header values in the base class.
  CachedBaseAddress = Hdr->BaseAddress;
  CachedNumAddresses = Hdr->NumAddresses;
  CachedAddrOffSize = Hdr->AddrOffSize;

  // Parse GlobalData entries to find section locations.
  uint64_t Offset = 24;
  uint64_t AddrOffsetsOff = 0, AddrOffsetsSize = 0;
  uint64_t AddrInfoOffsetsOff = 0, AddrInfoOffsetsSize = 0;
  uint64_t StringTableOff = 0, StringTableSize = 0;
  uint64_t FileTableOff = 0, FileTableSize = 0;
  uint64_t FuncInfoOff = 0, FuncInfoSize = 0;

  if (auto Err = parseGlobalDataEntries(
          DE, Offset, BufSize, AddrOffsetsOff, AddrOffsetsSize,
          AddrInfoOffsetsOff, AddrInfoOffsetsSize, StringTableOff,
          StringTableSize, FileTableOff, FileTableSize, FuncInfoOff,
          FuncInfoSize))
    return Err;

  if (!AddrOffsetsSize)
    return createStringError(std::errc::invalid_argument,
                             "missing AddrOffsets section");
  if (!AddrInfoOffsetsSize)
    return createStringError(std::errc::invalid_argument,
                             "missing AddrInfoOffsets section");
  if (!StringTableSize)
    return createStringError(std::errc::invalid_argument,
                             "missing StringTable section");
  if (!FileTableSize)
    return createStringError(std::errc::invalid_argument,
                             "missing FileTable section");

  if (AddrOffsetsSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrOffsets section size mismatch");

  if (AddrInfoOffsetsSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrInfoOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrInfoOffsets section size mismatch");

  if (!Swap) {
    AddrOffsets = ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(Buf.data() + AddrOffsetsOff),
        AddrOffsetsSize);

    if (Hdr->AddrInfoOffSize == 4) {
      AddrInfoOffsets = ArrayRef<uint32_t>(
          reinterpret_cast<const uint32_t *>(Buf.data() + AddrInfoOffsetsOff),
          Hdr->NumAddresses);
    } else {
      return createStringError(std::errc::not_supported,
                               "8-byte AddrInfoOffsets not yet supported");
    }

    if (FileTableSize < 4)
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small");
    uint32_t NumFiles;
    memcpy(&NumFiles, Buf.data() + FileTableOff, 4);
    if (FileTableSize < 4 + NumFiles * sizeof(FileEntry))
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small for %u files",
                               NumFiles);
    Files = ArrayRef<FileEntry>(
        reinterpret_cast<const FileEntry *>(Buf.data() + FileTableOff + 4),
        NumFiles);

    StrTab.Data = Buf.substr(StringTableOff, StringTableSize);
  } else {
    uint64_t AOff = AddrOffsetsOff;
    Swap->AddrOffsets.resize(AddrOffsetsSize);
    switch (Hdr->AddrOffSize) {
    case 1:
      if (!DE.getU8(&AOff, Swap->AddrOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 2:
      if (!DE.getU16(&AOff,
                     reinterpret_cast<uint16_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 4:
      if (!DE.getU32(&AOff,
                     reinterpret_cast<uint32_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 8:
      if (!DE.getU64(&AOff,
                     reinterpret_cast<uint64_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);

    if (Hdr->AddrInfoOffSize == 4) {
      uint64_t AIOff = AddrInfoOffsetsOff;
      Swap->AddrInfoOffsets.resize(Hdr->NumAddresses);
      if (!DE.getU32(&AIOff, Swap->AddrInfoOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address info offsets");
      AddrInfoOffsets = ArrayRef<uint32_t>(Swap->AddrInfoOffsets);
    } else {
      return createStringError(std::errc::not_supported,
                               "8-byte AddrInfoOffsets not yet supported");
    }

    uint64_t FTOff = FileTableOff;
    uint32_t NumFiles = DE.getU32(&FTOff);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (!DE.getU32(&FTOff, &Swap->Files[0].Dir, NumFiles * 2))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
      Files = ArrayRef<FileEntry>(Swap->Files);
    }

    StrTab.Data = Buf.substr(StringTableOff, StringTableSize);
  }
  return Error::success();
}

const HeaderV2 &GsymReaderV2::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

void GsymReaderV2::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET";

  switch (CachedAddrOffSize) {
  case 1: OS << "8 "; break;
  case 2: OS << "16"; break;
  case 4: OS << "32"; break;
  case 8: OS << "64"; break;
  default: OS << "??"; break;
  }
  OS << " (ADDRESS)\n";
  OS << "====== =============================== \n";
  for (uint32_t I = 0; I < CachedNumAddresses; ++I) {
    OS << format("[%4u] ", I);
    switch (CachedAddrOffSize) {
    case 1: OS << HEX8(getAddrOffsets<uint8_t>()[I]); break;
    case 2: OS << HEX16(getAddrOffsets<uint16_t>()[I]); break;
    case 4: OS << HEX32(getAddrOffsets<uint32_t>()[I]); break;
    case 8: OS << HEX32(getAddrOffsets<uint64_t>()[I]); break;
    default: break;
    }
    OS << " (" << HEX64(*getAddress(I)) << ")\n";
  }
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < CachedNumAddresses; ++I)
    OS << format("[%4u] ", I) << HEX32(AddrInfoOffsets[I]) << "\n";
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

  for (uint32_t I = 0; I < CachedNumAddresses; ++I) {
    OS << "FunctionInfo @ " << HEX32(AddrInfoOffsets[I]) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
