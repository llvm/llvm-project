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

void DWARFDebugRnglistTable::clear() {
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

Error DWARFDebugRnglistTable::extractHeaderAndOffsets(DWARFDataExtractor Data,
                                                      uint32_t *OffsetPtr) {
  HeaderOffset = *OffsetPtr;
  // Read and verify the length field.
  if (!Data.isValidOffsetForDataOfSize(*OffsetPtr, sizeof(uint32_t)))
    return createError("section is not large enough to contain a "
                       ".debug_rnglists table length at offset 0x%" PRIx32,
                       *OffsetPtr);
  // TODO: Add support for DWARF64.
  HeaderData.Length = Data.getU32(OffsetPtr);
  if (HeaderData.Length == 0xffffffffu)
    return createError(
        "DWARF64 is not supported in .debug_rnglists at offset 0x%" PRIx32,
        HeaderOffset);
  if (HeaderData.Length + sizeof(uint32_t) < sizeof(Header))
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has too small length (0x%" PRIx32
                       ") to contain a complete header",
                       HeaderOffset, length());
  uint32_t End = HeaderOffset + length();
  if (!Data.isValidOffsetForDataOfSize(HeaderOffset, End - HeaderOffset))
    return createError(
        "section is not large enough to contain a .debug_rnglists table "
        "of length 0x%" PRIx32 " at offset 0x%" PRIx32,
        length(), HeaderOffset);

  HeaderData.Version = Data.getU16(OffsetPtr);
  HeaderData.AddrSize = Data.getU8(OffsetPtr);
  HeaderData.SegSize = Data.getU8(OffsetPtr);
  HeaderData.OffsetEntryCount = Data.getU32(OffsetPtr);

  // Perform basic validation of the remaining header fields.
  if (HeaderData.Version != 5)
    return createError("unrecognised .debug_rnglists table version %" PRIu16
                       " in table at offset 0x%" PRIx32,
                       HeaderData.Version, HeaderOffset);
  if (HeaderData.AddrSize != 4 && HeaderData.AddrSize != 8)
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has unsupported address size %hhu",
                       HeaderOffset, HeaderData.AddrSize);
  if (HeaderData.SegSize != 0)
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has unsupported segment selector size %" PRIu8,
                       HeaderOffset, HeaderData.SegSize);
  if (End < HeaderOffset + sizeof(HeaderData) +
                HeaderData.OffsetEntryCount * sizeof(uint32_t))
    return createError(".debug_rnglists table at offset 0x%" PRIx32
                       " has more offset entries (%" PRIu32
                       ") than there is space for",
                       HeaderOffset, HeaderData.OffsetEntryCount);
  Data.setAddressSize(HeaderData.AddrSize);
  for (uint32_t I = 0; I < HeaderData.OffsetEntryCount; ++I)
    Offsets.push_back(Data.getU32(OffsetPtr));
  return Error::success();
}

Error DWARFDebugRnglist::RangeListEntry::extract(DWARFDataExtractor Data,
                                                 uint32_t End,
                                                 uint32_t *OffsetPtr) {
  Offset = *OffsetPtr;
  // The caller should guarantee that we have at least 1 byte available, so
  // we just assert instead of revalidate.
  assert(*OffsetPtr < End &&
         "not enough space to extract a rangelist encoding");
  uint8_t Encoding = Data.getU8(OffsetPtr);

  switch (Encoding) {
  case dwarf::DW_RLE_end_of_list:
    Value0 = Value1 = 0;
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
  case dwarf::DW_RLE_offset_pair: {
    uint32_t PreviousOffset = *OffsetPtr - 1;
    Value0 = Data.getULEB128(OffsetPtr);
    Value1 = Data.getULEB128(OffsetPtr);
    if (End < *OffsetPtr)
      return createError("read past end of table when reading "
                         "DW_RLE_offset_pair encoding at offset 0x%" PRIx32,
                         PreviousOffset);
    break;
  }
  case dwarf::DW_RLE_base_address: {
    if ((End - *OffsetPtr) < Data.getAddressSize())
      return createError("insufficient space remaining in table for "
                         "DW_RLE_base_address encoding at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    Value0 = Data.getAddress(OffsetPtr);
    break;
  }
  case dwarf::DW_RLE_start_end: {
    if ((End - *OffsetPtr) < unsigned(Data.getAddressSize() * 2))
      return createError("insufficient space remaining in table for "
                         "DW_RLE_start_end encoding "
                         "at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    Value0 = Data.getAddress(OffsetPtr);
    Value1 = Data.getAddress(OffsetPtr);
    break;
  }
  case dwarf::DW_RLE_start_length: {
    uint32_t PreviousOffset = *OffsetPtr - 1;
    Value0 = Data.getAddress(OffsetPtr);
    Value1 = Data.getULEB128(OffsetPtr);
    if (End < *OffsetPtr)
      return createError("read past end of table when reading "
                         "DW_RLE_start_length encoding at offset 0x%" PRIx32,
                         PreviousOffset);
    break;
  }
  default:
    return createError("unknown rnglists encoding 0x%" PRIx32
                       " at offset 0x%" PRIx32,
                       uint32_t(Encoding), *OffsetPtr - 1);
  }

  EntryKind = Encoding;
  return Error::success();
}

Error DWARFDebugRnglist::extract(DWARFDataExtractor Data, uint32_t HeaderOffset,
                                 uint32_t End, uint32_t *OffsetPtr) {
  Entries.clear();
  while (*OffsetPtr < End) {
    RangeListEntry Entry{0, 0, 0, 0};
    if (Error E = Entry.extract(Data, End, OffsetPtr))
      return E;
    Entries.push_back(Entry);
    if (Entry.EntryKind == dwarf::DW_RLE_end_of_list)
      return Error::success();
  }
  return createError(
      "no end of list marker detected at end of .debug_rnglists table "
      "starting at offset 0x%" PRIx32,
      HeaderOffset);
}

Error DWARFDebugRnglistTable::extract(DWARFDataExtractor Data,
                                      uint32_t *OffsetPtr) {
  clear();
  if (Error E = extractHeaderAndOffsets(Data, OffsetPtr))
    return E;

  Data.setAddressSize(HeaderData.AddrSize);
  uint32_t End = HeaderOffset + length();
  while (*OffsetPtr < End) {
    DWARFDebugRnglist CurrentRangeList;
    uint32_t Off = *OffsetPtr;
    if (Error E = CurrentRangeList.extract(Data, HeaderOffset, End, OffsetPtr))
      return E;
    Ranges[Off] = CurrentRangeList;
  }

  assert(*OffsetPtr == End &&
         "mismatch between expected length of .debug_rnglists table and length "
         "of extracted data");
  return Error::success();
}

static void dumpRangeEntry(raw_ostream &OS,
                           DWARFDebugRnglist::RangeListEntry Entry,
                           uint8_t AddrSize, uint8_t MaxEncodingStringLength,
                           uint64_t &CurrentBase, DIDumpOptions DumpOpts) {
  auto PrintRawEntry = [](raw_ostream &OS,
                          DWARFDebugRnglist::RangeListEntry Entry,
                          uint8_t AddrSize, DIDumpOptions DumpOpts) {
    if (DumpOpts.Verbose) {
      DumpOpts.DisplayRawContents = true;
      DWARFAddressRange(Entry.Value0, Entry.Value1)
          .dump(OS, AddrSize, DumpOpts);
      OS << " => ";
    }
  };

  if (DumpOpts.Verbose) {
    // Print the section offset in verbose mode.
    OS << format("0x%8.8" PRIx32 ":", Entry.Offset);
    auto EncodingString = dwarf::RangeListEncodingString(Entry.EntryKind);
    // Unsupported encodings should have been reported during parsing.
    assert(!EncodingString.empty() && "Unknown range entry encoding");
    OS << format(" [%s%*c", EncodingString.data(),
                 MaxEncodingStringLength - EncodingString.size() + 1, ']');
    if (Entry.EntryKind != dwarf::DW_RLE_end_of_list)
      OS << ": ";
  }

  switch (Entry.EntryKind) {
  case dwarf::DW_RLE_end_of_list:
    OS << (DumpOpts.Verbose ? "" : "<End of list>");
    break;
  case dwarf::DW_RLE_base_address:
    // In non-verbose mode we do not print anything for this entry.
    CurrentBase = Entry.Value0;
    if (!DumpOpts.Verbose)
      return;
    OS << format(" 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, Entry.Value0);
    break;
  case dwarf::DW_RLE_start_length:
    PrintRawEntry(OS, Entry, AddrSize, DumpOpts);
    DWARFAddressRange(Entry.Value0, Entry.Value0 + Entry.Value1)
        .dump(OS, AddrSize, DumpOpts);
    break;
  case dwarf::DW_RLE_offset_pair:
    PrintRawEntry(OS, Entry, AddrSize, DumpOpts);
    DWARFAddressRange(Entry.Value0 + CurrentBase, Entry.Value1 + CurrentBase)
        .dump(OS, AddrSize, DumpOpts);
    break;
  case dwarf::DW_RLE_start_end:
    DWARFAddressRange(Entry.Value0, Entry.Value1).dump(OS, AddrSize, DumpOpts);
    break;
  default:
    llvm_unreachable("Unsupported range list encoding");
  }
  OS << "\n";
}

void DWARFDebugRnglistTable::dump(raw_ostream &OS,
                                  DIDumpOptions DumpOpts) const {
  if (DumpOpts.Verbose)
    OS << format("0x%8.8" PRIx32 ": ", HeaderOffset);
  OS << format("Range List Header: length = 0x%8.8" PRIx32
               ", version = 0x%4.4" PRIx16 ", "
               "addr_size = 0x%2.2" PRIx8 ", seg_size = 0x%2.2" PRIx8
               ", offset_entry_count = "
               "0x%8.8" PRIx32 "\n",
               HeaderData.Length, HeaderData.Version, HeaderData.AddrSize,
               HeaderData.SegSize, HeaderData.OffsetEntryCount);

  if (HeaderData.OffsetEntryCount > 0) {
    OS << "Offsets: [";
    for (const auto &Off : Offsets) {
      OS << format("\n0x%8.8" PRIx32, Off);
      if (DumpOpts.Verbose)
        OS << format(" => 0x%8.8" PRIx32,
                     Off + HeaderOffset + sizeof(HeaderData));
    }
    OS << "\n]\n";
  }
  OS << "Ranges:\n";

  // Determine the length of the longest encoding string we have in the table,
  // so we can align the output properly. We only need this in verbose mode.
  size_t MaxEncodingStringLength = 0;
  if (DumpOpts.Verbose) {
    for (const auto &List : Ranges)
      for (const auto &Entry : List.second.getEntries())
        MaxEncodingStringLength =
            std::max(MaxEncodingStringLength,
                     dwarf::RangeListEncodingString(Entry.EntryKind).size());
  }

  uint64_t CurrentBase = 0;
  for (const auto &List : Ranges)
    for (const auto &Entry : List.second.getEntries())
      dumpRangeEntry(OS, Entry, HeaderData.AddrSize, MaxEncodingStringLength,
                     CurrentBase, DumpOpts);
}

uint32_t DWARFDebugRnglistTable::length() const {
  if (HeaderData.Length == 0)
    return 0;
  // TODO: DWARF64 support.
  return HeaderData.Length + sizeof(uint32_t);
}
