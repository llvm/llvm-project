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

GsymReaderV1::GsymReaderV1(std::unique_ptr<MemoryBuffer> Buffer)
    : GsymReader(std::move(Buffer)) {}

GsymReaderV1::GsymReaderV1(GsymReaderV1 &&RHS) = default;
GsymReaderV1::~GsymReaderV1() = default;

llvm::Expected<GsymReaderV1> GsymReaderV1::openFile(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  auto Err = BuffOrErr.getError();
  if (Err)
    return llvm::errorCodeToError(Err);
  return create(BuffOrErr.get());
}

llvm::Expected<GsymReaderV1> GsymReaderV1::copyBuffer(StringRef Bytes) {
  auto MB = MemoryBuffer::getMemBufferCopy(Bytes, "GSYM bytes");
  return create(MB);
}

llvm::Expected<GsymReaderV1>
GsymReaderV1::create(std::unique_ptr<MemoryBuffer> &MB) {
  if (!MB)
    return createStringError(std::errc::invalid_argument,
                             "invalid memory buffer");
  GsymReaderV1 GR(std::move(MB));
  if (auto Err = GR.parse())
    return std::move(Err);
  return std::move(GR);
}

llvm::Error GsymReaderV1::parse() {
  BinaryStreamReader FileData(MemBuffer->getBuffer(), llvm::endianness::native);
  // Check for the magic bytes. This file format is designed to be mmap'ed
  // into a process and accessed as read only. This is done for performance
  // and efficiency for symbolicating and parsing GSYM data.
  if (FileData.readObject(Hdr))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM header");

  const auto HostByteOrder = llvm::endianness::native;
  switch (Hdr->Magic) {
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

  bool DataIsLittleEndian = HostByteOrder != llvm::endianness::little;
  // Read a correctly byte swapped header if we need to.
  if (Swap) {
    DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);
    if (auto ExpectedHdr = Header::decode(Data))
      Swap->Hdr = ExpectedHdr.get();
    else
      return ExpectedHdr.takeError();
    Hdr = &Swap->Hdr;
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and a valid UUID size.
  if (Error Err = Hdr->checkForError())
    return Err;

  if (!Swap) {
    if (FileData.padToAlignment(Hdr->AddrOffSize) ||
        FileData.readArray(AddrOffsets, Hdr->NumAddresses * Hdr->AddrOffSize))
      return createStringError(std::errc::invalid_argument,
                               "failed to read address table");

    if (FileData.padToAlignment(4) ||
        FileData.readArray(AddrInfoOffsets, Hdr->NumAddresses))
      return createStringError(std::errc::invalid_argument,
                               "failed to read address info offsets table");

    uint32_t NumFiles = 0;
    if (FileData.readInteger(NumFiles) || FileData.readArray(Files, NumFiles))
      return createStringError(std::errc::invalid_argument,
                               "failed to read file table");

    FileData.setOffset(Hdr->StrtabOffset);
    if (FileData.readFixedString(StrTab.Data, Hdr->StrtabSize))
      return createStringError(std::errc::invalid_argument,
                               "failed to read string table");
  } else {
    DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);

    uint64_t Offset = alignTo(sizeof(Header), Hdr->AddrOffSize);
    Swap->AddrOffsets.resize(Hdr->NumAddresses * Hdr->AddrOffSize);
    switch (Hdr->AddrOffSize) {
    case 1:
      if (!Data.getU8(&Offset, Swap->AddrOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 2:
      if (!Data.getU16(&Offset,
                       reinterpret_cast<uint16_t *>(Swap->AddrOffsets.data()),
                       Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 4:
      if (!Data.getU32(&Offset,
                       reinterpret_cast<uint32_t *>(Swap->AddrOffsets.data()),
                       Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 8:
      if (!Data.getU64(&Offset,
                       reinterpret_cast<uint64_t *>(Swap->AddrOffsets.data()),
                       Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);

    Offset = alignTo(Offset, 4);
    Swap->AddrInfoOffsets.resize(Hdr->NumAddresses);
    if (Data.getU32(&Offset, Swap->AddrInfoOffsets.data(), Hdr->NumAddresses))
      AddrInfoOffsets = ArrayRef<uint32_t>(Swap->AddrInfoOffsets);
    else
      return createStringError(std::errc::invalid_argument,
                               "failed to read address table");

    const uint32_t NumFiles = Data.getU32(&Offset);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (Data.getU32(&Offset, &Swap->Files[0].Dir, NumFiles * 2))
        Files = ArrayRef<FileEntry>(Swap->Files);
      else
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
    }

    StrTab.Data =
        MemBuffer->getBuffer().substr(Hdr->StrtabOffset, Hdr->StrtabSize);
    if (StrTab.Data.empty())
      return createStringError(std::errc::invalid_argument,
                               "failed to read string table");
  }
  return Error::success();
}

const Header &GsymReaderV1::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

std::optional<uint64_t> GsymReaderV1::getAddress(size_t Index) const {
  switch (Hdr->AddrOffSize) {
  case 1:
    return addressForIndex<uint8_t>(Index);
  case 2:
    return addressForIndex<uint16_t>(Index);
  case 4:
    return addressForIndex<uint32_t>(Index);
  case 8:
    return addressForIndex<uint64_t>(Index);
  }
  return std::nullopt;
}

Expected<uint64_t> GsymReaderV1::getAddressIndex(const uint64_t Addr) const {
  if (Addr >= Hdr->BaseAddress) {
    const uint64_t AddrOffset = Addr - Hdr->BaseAddress;
    std::optional<uint64_t> AddrOffsetIndex;
    switch (Hdr->AddrOffSize) {
    case 1:
      AddrOffsetIndex = getAddressOffsetIndex<uint8_t>(AddrOffset);
      break;
    case 2:
      AddrOffsetIndex = getAddressOffsetIndex<uint16_t>(AddrOffset);
      break;
    case 4:
      AddrOffsetIndex = getAddressOffsetIndex<uint32_t>(AddrOffset);
      break;
    case 8:
      AddrOffsetIndex = getAddressOffsetIndex<uint64_t>(AddrOffset);
      break;
    default:
      return createStringError(std::errc::invalid_argument,
                               "unsupported address offset size %u",
                               Hdr->AddrOffSize);
    }
    if (AddrOffsetIndex)
      return *AddrOffsetIndex;
  }
  return createStringError(std::errc::invalid_argument,
                           "address 0x%" PRIx64 " is not in GSYM", Addr);
}

uint64_t GsymReaderV1::getAddressInfoOffset(size_t Index) const {
  return AddrInfoOffsets[Index];
}

void GsymReaderV1::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET";

  switch (getAddressOffsetByteSize()) {
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
    switch (getAddressOffsetByteSize()) {
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
    OS << format("[%4u] ", I) << HEX32(AddrInfoOffsets[I]) << "\n";
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
  OS << "\n" << StrTab << "\n";

  for (uint32_t I = 0; I < getNumAddresses(); ++I) {
    OS << "FunctionInfo @ " << HEX32(AddrInfoOffsets[I]) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}
