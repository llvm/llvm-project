//===- DWARFListTable.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFListTable.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Error DWARFListTableHeader::extract(DWARFDataExtractor Data,
                                    uint64_t *OffsetPtr) {
  HeaderOffset = *OffsetPtr;
  // Read and verify the length field.
  if (!Data.isValidOffsetForDataOfSize(*OffsetPtr, sizeof(uint32_t)))
    return createStringError(errc::invalid_argument,
                       "section is not large enough to contain a "
                       "%s table length at offset 0x%" PRIx64,
                       SectionName.data(), *OffsetPtr);
  Format = dwarf::DwarfFormat::DWARF32;
  uint8_t OffsetByteSize = 4;
  HeaderData.Length = Data.getRelocatedValue(4, OffsetPtr);
  if (HeaderData.Length == dwarf::DW_LENGTH_DWARF64) {
    Format = dwarf::DwarfFormat::DWARF64;
    OffsetByteSize = 8;
    HeaderData.Length = Data.getU64(OffsetPtr);
  } else if (HeaderData.Length >= dwarf::DW_LENGTH_lo_reserved) {
    return createStringError(errc::invalid_argument,
        "%s table at offset 0x%" PRIx64
        " has unsupported reserved unit length of value 0x%8.8" PRIx64,
        SectionName.data(), HeaderOffset, HeaderData.Length);
  }
  uint64_t FullLength =
      HeaderData.Length + dwarf::getUnitLengthFieldByteSize(Format);
  assert(FullLength == length());
  if (FullLength < getHeaderSize(Format))
    return createStringError(errc::invalid_argument,
                       "%s table at offset 0x%" PRIx64
                       " has too small length (0x%" PRIx64
                       ") to contain a complete header",
                       SectionName.data(), HeaderOffset, FullLength);
  uint64_t End = HeaderOffset + FullLength;
  if (!Data.isValidOffsetForDataOfSize(HeaderOffset, FullLength))
    return createStringError(errc::invalid_argument,
                       "section is not large enough to contain a %s table "
                       "of length 0x%" PRIx64 " at offset 0x%" PRIx64,
                       SectionName.data(), FullLength, HeaderOffset);

  HeaderData.Version = Data.getU16(OffsetPtr);
  HeaderData.AddrSize = Data.getU8(OffsetPtr);
  HeaderData.SegSize = Data.getU8(OffsetPtr);
  HeaderData.OffsetEntryCount = Data.getU32(OffsetPtr);

  // Perform basic validation of the remaining header fields.
  if (HeaderData.Version != 5)
    return createStringError(errc::invalid_argument,
                       "unrecognised %s table version %" PRIu16
                       " in table at offset 0x%" PRIx64,
                       SectionName.data(), HeaderData.Version, HeaderOffset);
  if (HeaderData.AddrSize != 4 && HeaderData.AddrSize != 8)
    return createStringError(errc::not_supported,
                       "%s table at offset 0x%" PRIx64
                       " has unsupported address size %" PRIu8,
                       SectionName.data(), HeaderOffset, HeaderData.AddrSize);
  if (HeaderData.SegSize != 0)
    return createStringError(errc::not_supported,
                       "%s table at offset 0x%" PRIx64
                       " has unsupported segment selector size %" PRIu8,
                       SectionName.data(), HeaderOffset, HeaderData.SegSize);
  if (End < HeaderOffset + getHeaderSize(Format) +
                HeaderData.OffsetEntryCount * OffsetByteSize)
    return createStringError(errc::invalid_argument,
        "%s table at offset 0x%" PRIx64 " has more offset entries (%" PRIu32
        ") than there is space for",
        SectionName.data(), HeaderOffset, HeaderData.OffsetEntryCount);
  Data.setAddressSize(HeaderData.AddrSize);
  for (uint32_t I = 0; I < HeaderData.OffsetEntryCount; ++I)
    Offsets.push_back(Data.getRelocatedValue(OffsetByteSize, OffsetPtr));
  return Error::success();
}

void DWARFListTableHeader::dump(raw_ostream &OS, DIDumpOptions DumpOpts) const {
  if (DumpOpts.Verbose)
    OS << format("0x%8.8" PRIx64 ": ", HeaderOffset);
  OS << format(
      "%s list header: length = 0x%8.8" PRIx64 ", version = 0x%4.4" PRIx16 ", "
      "addr_size = 0x%2.2" PRIx8 ", seg_size = 0x%2.2" PRIx8
      ", offset_entry_count = "
      "0x%8.8" PRIx32 "\n",
      ListTypeString.data(), HeaderData.Length, HeaderData.Version,
      HeaderData.AddrSize, HeaderData.SegSize, HeaderData.OffsetEntryCount);

  if (HeaderData.OffsetEntryCount > 0) {
    OS << "offsets: [";
    for (const auto &Off : Offsets) {
      OS << format("\n0x%8.8" PRIx64, Off);
      if (DumpOpts.Verbose)
        OS << format(" => 0x%8.8" PRIx64,
                     Off + HeaderOffset + getHeaderSize(Format));
    }
    OS << "\n]\n";
  }
}

uint64_t DWARFListTableHeader::length() const {
  if (HeaderData.Length == 0)
    return 0;
  return HeaderData.Length + dwarf::getUnitLengthFieldByteSize(Format);
}
