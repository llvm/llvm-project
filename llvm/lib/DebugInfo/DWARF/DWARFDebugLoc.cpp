//===- DWARFDebugLoc.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cinttypes>
#include <cstdint>

using namespace llvm;

// When directly dumping the .debug_loc without a compile unit, we have to guess
// at the DWARF version. This only affects DW_OP_call_ref, which is a rare
// expression that LLVM doesn't produce. Guessing the wrong version means we
// won't be able to pretty print expressions in DWARF2 binaries produced by
// non-LLVM tools.
static void dumpExpression(raw_ostream &OS, ArrayRef<uint8_t> Data,
                           bool IsLittleEndian, unsigned AddressSize,
                           const MCRegisterInfo *MRI, DWARFUnit *U) {
  DWARFDataExtractor Extractor(toStringRef(Data), IsLittleEndian, AddressSize);
  DWARFExpression(Extractor, dwarf::DWARF_VERSION, AddressSize).print(OS, MRI, U);
}

void DWARFDebugLoc::LocationList::dump(raw_ostream &OS, uint64_t BaseAddress,
                                       bool IsLittleEndian,
                                       unsigned AddressSize,
                                       const MCRegisterInfo *MRI, DWARFUnit *U,
                                       DIDumpOptions DumpOpts,
                                       unsigned Indent) const {
  for (const Entry &E : Entries) {
    OS << '\n';
    OS.indent(Indent);
    OS << format("[0x%*.*" PRIx64 ", ", AddressSize * 2, AddressSize * 2,
                 BaseAddress + E.Begin);
    OS << format(" 0x%*.*" PRIx64 ")", AddressSize * 2, AddressSize * 2,
                 BaseAddress + E.End);
    OS << ": ";

    dumpExpression(OS, E.Loc, IsLittleEndian, AddressSize, MRI, U);
  }
}

DWARFDebugLoc::LocationList const *
DWARFDebugLoc::getLocationListAtOffset(uint64_t Offset) const {
  auto It = partition_point(
      Locations, [=](const LocationList &L) { return L.Offset < Offset; });
  if (It != Locations.end() && It->Offset == Offset)
    return &(*It);
  return nullptr;
}

void DWARFDebugLoc::dump(raw_ostream &OS, const MCRegisterInfo *MRI, DIDumpOptions DumpOpts,
                         Optional<uint64_t> Offset) const {
  auto DumpLocationList = [&](const LocationList &L) {
    OS << format("0x%8.8" PRIx64 ": ", L.Offset);
    L.dump(OS, 0, IsLittleEndian, AddressSize, MRI, nullptr, DumpOpts, 12);
    OS << "\n";
  };

  if (Offset) {
    if (auto *L = getLocationListAtOffset(*Offset))
      DumpLocationList(*L);
    return;
  }

  for (const LocationList &L : Locations) {
    DumpLocationList(L);
    if (&L != &Locations.back())
      OS << '\n';
  }
}

Expected<DWARFDebugLoc::LocationList>
DWARFDebugLoc::parseOneLocationList(const DWARFDataExtractor &Data,
                                    uint64_t *Offset) {
  LocationList LL;
  LL.Offset = *Offset;
  AddressSize = Data.getAddressSize();
  DataExtractor::Cursor C(*Offset);

  // 2.6.2 Location Lists
  // A location list entry consists of:
  while (true) {
    Entry E;

    // 1. A beginning address offset. ...
    E.Begin = Data.getRelocatedAddress(C);

    // 2. An ending address offset. ...
    E.End = Data.getRelocatedAddress(C);

    if (Error Err = C.takeError())
      return std::move(Err);

    // The end of any given location list is marked by an end of list entry,
    // which consists of a 0 for the beginning address offset and a 0 for the
    // ending address offset.
    if (E.Begin == 0 && E.End == 0) {
      *Offset = C.tell();
      return LL;
    }

    if (E.Begin != (AddressSize == 4 ? -1U : -1ULL)) {
      unsigned Bytes = Data.getU16(C);
      // A single location description describing the location of the object...
      Data.getU8(C, E.Loc, Bytes);
    }

    LL.Entries.push_back(std::move(E));
  }
}

void DWARFDebugLoc::parse(const DWARFDataExtractor &data) {
  IsLittleEndian = data.isLittleEndian();
  AddressSize = data.getAddressSize();

  uint64_t Offset = 0;
  while (Offset < data.getData().size()) {
    if (auto LL = parseOneLocationList(data, &Offset))
      Locations.push_back(std::move(*LL));
    else {
      logAllUnhandledErrors(LL.takeError(), WithColor::error());
      break;
    }
  }
}

Expected<DWARFDebugLoclists::LocationList>
DWARFDebugLoclists::parseOneLocationList(const DataExtractor &Data,
                                         uint64_t *Offset, unsigned Version) {
  LocationList LL;
  LL.Offset = *Offset;
  DataExtractor::Cursor C(*Offset);

  // dwarf::DW_LLE_end_of_list_entry is 0 and indicates the end of the list.
  while (auto Kind = Data.getU8(C)) {
    Entry E;
    E.Kind = Kind;
    E.Offset = C.tell() - 1;
    switch (Kind) {
    case dwarf::DW_LLE_base_addressx:
      E.Value0 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_startx_length:
      E.Value0 = Data.getULEB128(C);
      // Pre-DWARF 5 has different interpretation of the length field. We have
      // to support both pre- and standartized styles for the compatibility.
      if (Version < 5)
        E.Value1 = Data.getU32(C);
      else
        E.Value1 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_start_length:
      E.Value0 = Data.getAddress(C);
      E.Value1 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_offset_pair:
      E.Value0 = Data.getULEB128(C);
      E.Value1 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_base_address:
      E.Value0 = Data.getAddress(C);
      break;
    default:
      cantFail(C.takeError());
      return createStringError(errc::illegal_byte_sequence,
                               "LLE of kind %x not supported", (int)Kind);
    }

    if (Kind != dwarf::DW_LLE_base_address &&
        Kind != dwarf::DW_LLE_base_addressx) {
      unsigned Bytes = Version >= 5 ? Data.getULEB128(C) : Data.getU16(C);
      // A single location description describing the location of the object...
      Data.getU8(C, E.Loc, Bytes);
    }

    LL.Entries.push_back(std::move(E));
  }
  if (Error Err = C.takeError())
    return std::move(Err);
  Entry E;
  E.Kind = dwarf::DW_LLE_end_of_list;
  E.Offset = C.tell() - 1;
  LL.Entries.push_back(E);
  *Offset = C.tell();
  return LL;
}

void DWARFDebugLoclists::parse(DataExtractor data, uint64_t Offset, uint64_t EndOffset, uint16_t Version) {
  IsLittleEndian = data.isLittleEndian();
  AddressSize = data.getAddressSize();

  while (Offset < EndOffset) {
    if (auto LL = parseOneLocationList(data, &Offset, Version))
      Locations.push_back(std::move(*LL));
    else {
      logAllUnhandledErrors(LL.takeError(), WithColor::error());
      return;
    }
  }
}

DWARFDebugLoclists::LocationList const *
DWARFDebugLoclists::getLocationListAtOffset(uint64_t Offset) const {
  auto It = partition_point(
      Locations, [=](const LocationList &L) { return L.Offset < Offset; });
  if (It != Locations.end() && It->Offset == Offset)
    return &(*It);
  return nullptr;
}

void DWARFDebugLoclists::Entry::dump(raw_ostream &OS, uint64_t &BaseAddr,
                                     bool IsLittleEndian, unsigned AddressSize,
                                     const MCRegisterInfo *MRI, DWARFUnit *U,
                                     DIDumpOptions DumpOpts, unsigned Indent,
                                     size_t MaxEncodingStringLength) const {
  if (DumpOpts.Verbose) {
    OS << "\n";
    OS.indent(Indent);
    auto EncodingString = dwarf::LocListEncodingString(Kind);
    // Unsupported encodings should have been reported during parsing.
    assert(!EncodingString.empty() && "Unknown loclist entry encoding");
    OS << format("%s%*c", EncodingString.data(),
                 MaxEncodingStringLength - EncodingString.size() + 1, '(');
    switch (Kind) {
    case dwarf::DW_LLE_startx_length:
    case dwarf::DW_LLE_start_length:
    case dwarf::DW_LLE_offset_pair:
      OS << format("0x%*.*" PRIx64 ", 0x%*.*" PRIx64, AddressSize * 2,
                 AddressSize * 2, Value0, AddressSize * 2, AddressSize * 2,
                 Value1);
      break;
    case dwarf::DW_LLE_base_addressx:
    case dwarf::DW_LLE_base_address:
      OS << format("0x%*.*" PRIx64, AddressSize * 2, AddressSize * 2,
                   Value0);
      break;
    case dwarf::DW_LLE_end_of_list:
      break;
    }
    OS << ')';
  }
  auto PrintPrefix = [&] {
    OS << "\n";
    OS.indent(Indent);
    if (DumpOpts.Verbose)
      OS << format("%*s", MaxEncodingStringLength, (const char *)"=> ");
  };
  switch (Kind) {
  case dwarf::DW_LLE_startx_length:
    PrintPrefix();
    OS << "Addr idx " << Value0 << " (w/ length " << Value1 << "): ";
    break;
  case dwarf::DW_LLE_start_length:
    PrintPrefix();
    DWARFAddressRange(Value0, Value0 + Value1)
        .dump(OS, AddressSize, DumpOpts);
    OS << ": ";
    break;
  case dwarf::DW_LLE_offset_pair:
    PrintPrefix();
    DWARFAddressRange(BaseAddr + Value0, BaseAddr + Value1)
        .dump(OS, AddressSize, DumpOpts);
    OS << ": ";
    break;
  case dwarf::DW_LLE_base_addressx:
    if (!DumpOpts.Verbose)
      return;
    break;
  case dwarf::DW_LLE_end_of_list:
    if (!DumpOpts.Verbose)
      return;
    break;
  case dwarf::DW_LLE_base_address:
    BaseAddr = Value0;
    if (!DumpOpts.Verbose)
      return;
    break;
  default:
    llvm_unreachable("unreachable locations list kind");
  }

  dumpExpression(OS, Loc, IsLittleEndian, AddressSize, MRI, U);
}
void DWARFDebugLoclists::LocationList::dump(raw_ostream &OS, uint64_t BaseAddr,
                                            bool IsLittleEndian,
                                            unsigned AddressSize,
                                            const MCRegisterInfo *MRI,
                                            DWARFUnit *U,
                                            DIDumpOptions DumpOpts, 
                                            unsigned Indent) const {
  size_t MaxEncodingStringLength = 0;
  if (DumpOpts.Verbose)
    for (const auto &Entry : Entries)
      MaxEncodingStringLength =
          std::max(MaxEncodingStringLength,
                   dwarf::LocListEncodingString(Entry.Kind).size());

  for (const Entry &E : Entries)
    E.dump(OS, BaseAddr, IsLittleEndian, AddressSize, MRI, U, DumpOpts, Indent,
           MaxEncodingStringLength);
}

void DWARFDebugLoclists::dump(raw_ostream &OS, uint64_t BaseAddr,
                              const MCRegisterInfo *MRI, DIDumpOptions DumpOpts, 
                              Optional<uint64_t> Offset) const {
  auto DumpLocationList = [&](const LocationList &L) {
    OS << format("0x%8.8" PRIx64 ": ", L.Offset);
    L.dump(OS, BaseAddr, IsLittleEndian, AddressSize, MRI, nullptr, DumpOpts,
           /*Indent=*/12);
    OS << "\n";
  };

  if (Offset) {
    if (auto *L = getLocationListAtOffset(*Offset))
      DumpLocationList(*L);
    return;
  }

  for (const LocationList &L : Locations) {
    DumpLocationList(L);
    if (&L != &Locations.back())
      OS << '\n';
  }
}
