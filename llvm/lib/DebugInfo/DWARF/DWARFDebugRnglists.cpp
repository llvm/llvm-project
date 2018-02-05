//===- DWARFDebugRnglists.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugRnglists.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFDebugRnglists::clear() {
  HeaderData = {};
  Offsets.clear();
  Ranges.clear();
}

template <typename... Ts>
static Error createError(char const *Fmt, const Ts &... Vals) {
  std::string Buffer;
  raw_string_ostream Stream(Buffer);
  Stream << format(Fmt, Vals...);
  return make_error<StringError>(Stream.str(), inconvertibleErrorCode());
}

Error DWARFDebugRnglists::extract(DWARFDataExtractor Data,
                                  uint32_t *OffsetPtr) {
  clear();
  uint32_t TableOffset = *OffsetPtr;

  // Read and verify the length field.
  if (!Data.isValidOffsetForDataOfSize(*OffsetPtr, sizeof(uint32_t)))
    return createError("section is not large enough to contain a "
                       ".debug_rnglists table length at offset 0x%" PRIx32,
                       *OffsetPtr);
  // TODO: Add support for DWARF64.
  HeaderData.Length = Data.getU32(OffsetPtr);
  if (HeaderData.Length + sizeof(uint32_t) < sizeof(Header))
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has too small length (0x%" PRIx32
                       ") to contain a complete header",
                       TableOffset, length());
  uint32_t End = TableOffset + length();
  if (!Data.isValidOffsetForDataOfSize(TableOffset, End - TableOffset))
    return createError(
        "section is not large enough to contain a .debug_rnglists table "
        "of length 0x%" PRIx32 " at offset 0x%" PRIx32,
        length(), TableOffset);

  HeaderData.Version = Data.getU16(OffsetPtr);
  HeaderData.AddrSize = Data.getU8(OffsetPtr);
  HeaderData.SegSize = Data.getU8(OffsetPtr);
  HeaderData.OffsetEntryCount = Data.getU32(OffsetPtr);

  // Perform basic validation of the remaining header fields.
  if (HeaderData.Version != 5)
    return createError("unrecognised .debug_rnglists table version %" PRIu16
                       " in table at offset 0x%" PRIx32,
                       HeaderData.Version, TableOffset);
  if (HeaderData.AddrSize != 4 && HeaderData.AddrSize != 8)
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has unsupported address size %hhu",
                       TableOffset, HeaderData.AddrSize);
  if (HeaderData.SegSize != 0)
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has unsupported segment selector size %" PRIu8,
                       TableOffset, HeaderData.SegSize);
  if (End < TableOffset + sizeof(HeaderData) +
                HeaderData.OffsetEntryCount * sizeof(uint32_t))
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has more offset entries (%" PRIu32
                       ") than there is space for",
                       TableOffset, HeaderData.OffsetEntryCount);

  Data.setAddressSize(HeaderData.AddrSize);

  for (uint32_t I = 0; I < HeaderData.OffsetEntryCount; ++I)
    Offsets.push_back(Data.getU32(OffsetPtr));

  DWARFAddressRangesVector CurrentRanges;
  while (*OffsetPtr < End) {
    uint8_t Encoding = Data.getU8(OffsetPtr);
    switch (Encoding) {
    case dwarf::DW_RLE_end_of_list:
      Ranges.insert(Ranges.end(), CurrentRanges);
      CurrentRanges.clear();
      break;
    // TODO: Support other encodings.
    case dwarf::DW_RLE_base_addressx:
      return createError("unsupported rnglists encoding DW_RLE_base_addressx "
                         "at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    case dwarf::DW_RLE_startx_endx:
      return createError("unsupported rnglists encoding DW_RLE_startx_endx at "
                         "offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    case dwarf::DW_RLE_startx_length:
      return createError("unsupported rnglists encoding DW_RLE_startx_length "
                         "at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    case dwarf::DW_RLE_offset_pair:
      return createError("unsupported rnglists encoding DW_RLE_offset_pair at "
                         "offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    case dwarf::DW_RLE_base_address:
      return createError("unsupported rnglists encoding DW_RLE_base_address at "
                         "offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    case dwarf::DW_RLE_start_end: {
      if ((End - *OffsetPtr) < unsigned(HeaderData.AddrSize * 2))
        return createError("insufficient space remaining in table for "
                           "DW_RLE_start_end encoding "
                           "at offset 0x%" PRIx32,
                           *OffsetPtr - 1);
      uint64_t Start = Data.getAddress(OffsetPtr);
      uint64_t End = Data.getAddress(OffsetPtr);
      CurrentRanges.emplace_back(Start, End);
      break;
    }
    case dwarf::DW_RLE_start_length: {
      uint32_t PreviousOffset = *OffsetPtr - 1;
      uint64_t Start = Data.getAddress(OffsetPtr);
      uint64_t Length = Data.getULEB128(OffsetPtr);
      if (End < *OffsetPtr)
        return createError("read past end of table when reading "
                           "DW_RLE_start_length encoding at offset 0x%" PRIx32,
                           PreviousOffset);
      CurrentRanges.emplace_back(Start, Start + Length);
      break;
    }
    default:
      Ranges.insert(Ranges.end(), CurrentRanges);
      return createError("unknown rnglists encoding 0x%" PRIx32
                         " at offset 0x%" PRIx32,
                         uint32_t(Encoding), *OffsetPtr - 1);
    }
  }

  // If OffsetPtr does not indicate the End offset, then either the above loop
  // terminated prematurely, or we encountered a malformed encoding, but did not
  // report an error when we should have done.
  assert(*OffsetPtr == End &&
         "did not detect malformed data or loop ended unexpectedly");

  // If CurrentRanges is not empty, we have a malformed section, because we did
  // not find a DW_RLE_end_of_list marker at the end of the last list.
  if (!CurrentRanges.empty())
    return createError(
        "no end of list marker detected at end of .debug_rnglists table "
        "starting at offset 0x%" PRIx32,
        TableOffset);
  return Error::success();
}

void DWARFDebugRnglists::dump(raw_ostream &OS) const {
  // TODO: Add verbose printing of the raw encodings.
  OS << format("Range List Header: length = 0x%8.8x, version = 0x%4.4x, "
               "addr_size = 0x%2.2x, seg_size = 0x%2.2x, offset_entry_count = "
               "0x%8.8x\n",
               HeaderData.Length, HeaderData.Version, HeaderData.AddrSize,
               HeaderData.SegSize, HeaderData.OffsetEntryCount);

  if (HeaderData.OffsetEntryCount > 0) {
    OS << "Offsets: [";
    for (const auto &Off : Offsets)
      OS << format("\n0x%8.8x", Off);
    OS << "\n]\n";
  }
  OS << "Ranges:\n";

  const uint32_t HexWidth = HeaderData.AddrSize * 2;
  for (const auto &List : Ranges) {
    for (const auto &Entry : List)
      OS << format("[0x%*.*" PRIx64 ", 0x%*.*" PRIx64 ")\n", HexWidth, HexWidth,
                   Entry.LowPC, HexWidth, HexWidth, Entry.HighPC);
    OS << "<End of list>\n";
  }
}

uint32_t DWARFDebugRnglists::length() const {
  if (HeaderData.Length == 0)
    return 0;
  // TODO: DWARF64 support.
  return HeaderData.Length + sizeof(uint32_t);
}
