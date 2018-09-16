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
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Error RangeListEntry::extract(DWARFDataExtractor Data, uint32_t End,
                              uint16_t Version, StringRef /* SectionName */,
                              uint32_t *OffsetPtr, bool /* isDWO */) {
  Offset = *OffsetPtr;
  SectionIndex = -1ULL;

  assert((Data.getAddressSize() == 4 || Data.getAddressSize() == 8) &&
         "Unsupported address size");

  // We model a DWARF v4 range list entry like DWARF v5 DW_RLE_offset_pair,
  // since it is subject to base adjustment.
  uint8_t Encoding = dwarf::DW_RLE_offset_pair;
  if (Version > 4) {
    // The caller should guarantee that we have at least 1 byte available, so
    // we just assert instead of revalidate.
    assert(*OffsetPtr < End &&
           "not enough space to extract a rangelist encoding");
    Encoding = Data.getU8(OffsetPtr);
  }

  switch (Encoding) {
  case dwarf::DW_RLE_end_of_list:
    Value0 = Value1 = 0;
    break;
  // TODO: Support other encodings.
  case dwarf::DW_RLE_base_addressx:
    return createStringError(errc::not_supported,
                       "unsupported rnglists encoding DW_RLE_base_addressx "
                       "at offset 0x%" PRIx32,
                       *OffsetPtr - 1);
  case dwarf::DW_RLE_startx_endx:
    return createStringError(errc::not_supported,
                       "unsupported rnglists encoding DW_RLE_startx_endx at "
                       "offset 0x%" PRIx32,
                       *OffsetPtr - 1);
  case dwarf::DW_RLE_startx_length:
    return createStringError(errc::not_supported,
                       "unsupported rnglists encoding DW_RLE_startx_length "
                       "at offset 0x%" PRIx32,
                       *OffsetPtr - 1);
  case dwarf::DW_RLE_offset_pair: {
    if (Version < 5) {
      if ((End - *OffsetPtr) < unsigned(Data.getAddressSize() * 2))
        return createStringError(
            errc::illegal_byte_sequence,
            "invalid range list entry at offset 0x%" PRIx32, *OffsetPtr);
      Value0 = Data.getRelocatedAddress(OffsetPtr);
      Value1 = Data.getRelocatedAddress(OffsetPtr, &SectionIndex);
      // Adjust the EntryKind for end-of-list and base_address based on the
      // contents.
      if (Value0 == maxUIntN(Data.getAddressSize() * 8)) {
        Encoding = dwarf::DW_RLE_base_address;
        Value0 = Value1;
        Value1 = 0;
      } else if (Value0 == 0 && Value1 == 0)
        Encoding = dwarf::DW_RLE_end_of_list;
      break;
    }
    uint32_t PreviousOffset = *OffsetPtr - 1;
    Value0 = Data.getULEB128(OffsetPtr);
    Value1 = Data.getULEB128(OffsetPtr);
    if (End < *OffsetPtr)
      return createStringError(errc::invalid_argument,
                         "read past end of table when reading "
                         "DW_RLE_offset_pair encoding at offset 0x%" PRIx32,
                         PreviousOffset);
    break;
  }
  case dwarf::DW_RLE_base_address:
    if ((End - *OffsetPtr) < Data.getAddressSize())
      return createStringError(errc::invalid_argument,
                         "insufficient space remaining in table for "
                         "DW_RLE_base_address encoding at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    Value0 = Data.getRelocatedAddress(OffsetPtr, &SectionIndex);
    break;
  case dwarf::DW_RLE_start_end:
    if ((End - *OffsetPtr) < unsigned(Data.getAddressSize() * 2))
      return createStringError(errc::invalid_argument,
                         "insufficient space remaining in table for "
                         "DW_RLE_start_end encoding "
                         "at offset 0x%" PRIx32,
                         *OffsetPtr - 1);
    Value0 = Data.getRelocatedAddress(OffsetPtr);
    Value1 = Data.getRelocatedAddress(OffsetPtr, &SectionIndex);
    break;
  case dwarf::DW_RLE_start_length: {
    uint32_t PreviousOffset = *OffsetPtr - 1;
    Value0 = Data.getRelocatedAddress(OffsetPtr, &SectionIndex);
    Value1 = Data.getULEB128(OffsetPtr);
    if (End < *OffsetPtr)
      return createStringError(errc::invalid_argument,
                         "read past end of table when reading "
                         "DW_RLE_start_length encoding at offset 0x%" PRIx32,
                         PreviousOffset);
    break;
  }
  default:
    return createStringError(errc::not_supported,
                       "unknown rnglists encoding 0x%" PRIx32
                       " at offset 0x%" PRIx32,
                       uint32_t(Encoding), *OffsetPtr - 1);
  }

  EntryKind = Encoding;
  return Error::success();
}

DWARFAddressRangesVector DWARFDebugRnglist::getAbsoluteRanges(
    llvm::Optional<BaseAddress> BaseAddr) const {
  DWARFAddressRangesVector Res;
  for (const RangeListEntry &RLE : Entries) {
    if (RLE.EntryKind == dwarf::DW_RLE_end_of_list)
      break;
    if (RLE.EntryKind == dwarf::DW_RLE_base_address) {
      BaseAddr = {RLE.Value0, RLE.SectionIndex};
      continue;
    }

    DWARFAddressRange E;
    E.SectionIndex = RLE.SectionIndex;
    if (BaseAddr && E.SectionIndex == -1ULL)
      E.SectionIndex = BaseAddr->SectionIndex;

    switch (RLE.EntryKind) {
    case dwarf::DW_RLE_offset_pair:
      E.LowPC = RLE.Value0;
      E.HighPC = RLE.Value1;
      if (BaseAddr) {
        E.LowPC += BaseAddr->Address;
        E.HighPC += BaseAddr->Address;
      }
      break;
    case dwarf::DW_RLE_start_end:
      E.LowPC = RLE.Value0;
      E.HighPC = RLE.Value1;
      break;
    case dwarf::DW_RLE_start_length:
      E.LowPC = RLE.Value0;
      E.HighPC = E.LowPC + RLE.Value1;
      break;
    default:
      // Unsupported encodings should have been reported during extraction,
      // so we should not run into any here.
      llvm_unreachable("Unsupported range list encoding");
    }
    Res.push_back(E);
  }
  return Res;
}

void RangeListEntry::dump(raw_ostream &OS, DWARFContext *, uint8_t AddrSize,
                          uint64_t &CurrentBase, unsigned Indent,
                          uint16_t Version, uint8_t MaxEncodingStringLength,
                          DIDumpOptions DumpOpts) const {
  auto PrintRawEntry = [](raw_ostream &OS, const RangeListEntry &Entry,
                          uint8_t AddrSize, DIDumpOptions DumpOpts) {
    if (DumpOpts.Verbose) {
      DumpOpts.DisplayRawContents = true;
      DWARFAddressRange(Entry.Value0, Entry.Value1)
          .dump(OS, AddrSize, DumpOpts);
      OS << " => ";
    }
  };

  // Output indentations before we print the actual entry. We only print
  // anything for DW_RLE_base_address when we are in verbose mode.
  if (DumpOpts.Verbose || !isBaseAddressSelectionEntry())
    OS.indent(Indent);

  if (DumpOpts.Verbose) {
    // Print the section offset in verbose mode.
    OS << format("0x%8.8" PRIx32 ":", Offset);
    if (Version > 4) {
      auto EncodingString = dwarf::RangeListEncodingString(EntryKind);
      // Unsupported encodings should have been reported during parsing.
      assert(!EncodingString.empty() && "Unknown range entry encoding");
      OS << format(" [%s%*c", EncodingString.data(),
                   MaxEncodingStringLength - EncodingString.size() + 1, ']');
      if (!isEndOfList())
        OS << ": ";
    }
  }

  switch (EntryKind) {
  case dwarf::DW_RLE_end_of_list:
    if (DumpOpts.Verbose) {
      // For DWARF v4 and earlier, we print the raw entry, i.e. 2 zeros.
      if (Version < 5) {
        OS << format(" 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, Value0);
        OS << format(", 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, Value1);
      }
      break;
    }
    OS << "<End of list>";
    break;
  case dwarf::DW_RLE_base_address:
    // In non-verbose mode, we do not print anything for this entry.
    CurrentBase = Value0;
    if (!DumpOpts.Verbose)
      return;
    if (Version < 5) {
      // Dump the entry in pre-DWARF v5 format, i.e. with a -1 as Value0.
      uint64_t allOnes = maxUIntN(AddrSize * 8);
      OS << format(" 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, allOnes);
      OS << format(", 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, Value0);
      break;
    }
    OS << format(" 0x%*.*" PRIx64, AddrSize * 2, AddrSize * 2, Value0);
    break;
  case dwarf::DW_RLE_start_length:
    PrintRawEntry(OS, *this, AddrSize, DumpOpts);
    DWARFAddressRange(Value0, Value0 + Value1).dump(OS, AddrSize, DumpOpts);
    break;
  case dwarf::DW_RLE_offset_pair:
    PrintRawEntry(OS, *this, AddrSize, DumpOpts);
    DWARFAddressRange(Value0 + CurrentBase, Value1 + CurrentBase)
        .dump(OS, AddrSize, DumpOpts);
    break;
  case dwarf::DW_RLE_start_end:
    DWARFAddressRange(Value0, Value1).dump(OS, AddrSize, DumpOpts);
    break;
  default:
    llvm_unreachable("Unsupported range list encoding");
  }
  OS << "\n";
}
