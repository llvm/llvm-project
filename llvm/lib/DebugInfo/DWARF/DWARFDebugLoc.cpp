//===- DWARFDebugLoc.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
static void dumpExpression(raw_ostream &OS, ArrayRef<char> Data,
                           bool IsLittleEndian, unsigned AddressSize,
                           const MCRegisterInfo *MRI) {
  DWARFDataExtractor Extractor(StringRef(Data.data(), Data.size()),
                               IsLittleEndian, AddressSize);
  DWARFExpression(Extractor, dwarf::DWARF_VERSION, AddressSize).print(OS, MRI);
}

void DWARFDebugLoc::LocationList::dump(raw_ostream &OS, bool IsLittleEndian,
                                       unsigned AddressSize,
                                       const MCRegisterInfo *MRI,
                                       unsigned Indent) const {
  for (const Entry &E : Entries) {
    OS << '\n';
    OS.indent(Indent);
    OS << format("[0x%*.*" PRIx64 ", ", AddressSize * 2, AddressSize * 2,
                 E.Begin)
       << format(" 0x%*.*" PRIx64 ")", AddressSize * 2, AddressSize * 2, E.End);
    OS << ": ";

    dumpExpression(OS, E.Loc, IsLittleEndian, AddressSize, MRI);
  }
}

DWARFDebugLoc::LocationList const *
DWARFDebugLoc::getLocationListAtOffset(uint64_t Offset) const {
  auto It = std::lower_bound(
      Locations.begin(), Locations.end(), Offset,
      [](const LocationList &L, uint64_t Offset) { return L.Offset < Offset; });
  if (It != Locations.end() && It->Offset == Offset)
    return &(*It);
  return nullptr;
}

void DWARFDebugLoc::dump(raw_ostream &OS, const MCRegisterInfo *MRI,
                         Optional<uint64_t> Offset) const {
  auto DumpLocationList = [&](const LocationList &L) {
    OS << format("0x%8.8x: ", L.Offset);
    L.dump(OS, IsLittleEndian, AddressSize, MRI, 12);
    OS << "\n\n";
  };

  if (Offset) {
    if (auto *L = getLocationListAtOffset(*Offset))
      DumpLocationList(*L);
    return;
  }

  for (const LocationList &L : Locations) {
    DumpLocationList(L);
  }
}

Optional<DWARFDebugLoc::LocationList>
DWARFDebugLoc::parseOneLocationList(DWARFDataExtractor Data, unsigned *Offset) {
  LocationList LL;
  LL.Offset = *Offset;

  // 2.6.2 Location Lists
  // A location list entry consists of:
  while (true) {
    Entry E;
    if (!Data.isValidOffsetForDataOfSize(*Offset, 2 * Data.getAddressSize())) {
      WithColor::error() << "location list overflows the debug_loc section.\n";
      return None;
    }

    // 1. A beginning address offset. ...
    E.Begin = Data.getRelocatedAddress(Offset);

    // 2. An ending address offset. ...
    E.End = Data.getRelocatedAddress(Offset);

    // The end of any given location list is marked by an end of list entry,
    // which consists of a 0 for the beginning address offset and a 0 for the
    // ending address offset.
    if (E.Begin == 0 && E.End == 0)
      return LL;

    if (!Data.isValidOffsetForDataOfSize(*Offset, 2)) {
      WithColor::error() << "location list overflows the debug_loc section.\n";
      return None;
    }

    unsigned Bytes = Data.getU16(Offset);
    if (!Data.isValidOffsetForDataOfSize(*Offset, Bytes)) {
      WithColor::error() << "location list overflows the debug_loc section.\n";
      return None;
    }
    // A single location description describing the location of the object...
    StringRef str = Data.getData().substr(*Offset, Bytes);
    *Offset += Bytes;
    E.Loc.reserve(str.size());
    std::copy(str.begin(), str.end(), std::back_inserter(E.Loc));
    LL.Entries.push_back(std::move(E));
  }
}

void DWARFDebugLoc::parse(const DWARFDataExtractor &data) {
  IsLittleEndian = data.isLittleEndian();
  AddressSize = data.getAddressSize();

  uint32_t Offset = 0;
  while (data.isValidOffset(Offset + data.getAddressSize() - 1)) {
    if (auto LL = parseOneLocationList(data, &Offset))
      Locations.push_back(std::move(*LL));
    else
      break;
  }
  if (data.isValidOffset(Offset))
    WithColor::error() << "failed to consume entire .debug_loc section\n";
}

Optional<DWARFDebugLocDWO::LocationList>
DWARFDebugLocDWO::parseOneLocationList(DataExtractor Data, unsigned *Offset) {
  LocationList LL;
  LL.Offset = *Offset;

  // dwarf::DW_LLE_end_of_list_entry is 0 and indicates the end of the list.
  while (auto Kind =
             static_cast<dwarf::LocationListEntry>(Data.getU8(Offset))) {
    if (Kind != dwarf::DW_LLE_startx_length) {
      WithColor::error() << "dumping support for LLE of kind " << (int)Kind
                         << " not implemented\n";
      return None;
    }

    Entry E;
    E.Start = Data.getULEB128(Offset);
    E.Length = Data.getU32(Offset);

    unsigned Bytes = Data.getU16(Offset);
    // A single location description describing the location of the object...
    StringRef str = Data.getData().substr(*Offset, Bytes);
    *Offset += Bytes;
    E.Loc.resize(str.size());
    std::copy(str.begin(), str.end(), E.Loc.begin());

    LL.Entries.push_back(std::move(E));
  }
  return LL;
}

void DWARFDebugLocDWO::parse(DataExtractor data) {
  IsLittleEndian = data.isLittleEndian();
  AddressSize = data.getAddressSize();

  uint32_t Offset = 0;
  while (data.isValidOffset(Offset)) {
    if (auto LL = parseOneLocationList(data, &Offset))
      Locations.push_back(std::move(*LL));
    else
      return;
  }
}

DWARFDebugLocDWO::LocationList const *
DWARFDebugLocDWO::getLocationListAtOffset(uint64_t Offset) const {
  auto It = std::lower_bound(
      Locations.begin(), Locations.end(), Offset,
      [](const LocationList &L, uint64_t Offset) { return L.Offset < Offset; });
  if (It != Locations.end() && It->Offset == Offset)
    return &(*It);
  return nullptr;
}

void DWARFDebugLocDWO::LocationList::dump(raw_ostream &OS, bool IsLittleEndian,
                                          unsigned AddressSize,
                                          const MCRegisterInfo *MRI,
                                          unsigned Indent) const {
  for (const Entry &E : Entries) {
    OS << '\n';
    OS.indent(Indent);
    OS << "Addr idx " << E.Start << " (w/ length " << E.Length << "): ";
    dumpExpression(OS, E.Loc, IsLittleEndian, AddressSize, MRI);
  }
}

void DWARFDebugLocDWO::dump(raw_ostream &OS, const MCRegisterInfo *MRI,
                            Optional<uint64_t> Offset) const {
  auto DumpLocationList = [&](const LocationList &L) {
    OS << format("0x%8.8x: ", L.Offset);
    L.dump(OS, IsLittleEndian, AddressSize, MRI, /*Indent=*/12);
    OS << "\n\n";
  };

  if (Offset) {
    if (auto *L = getLocationListAtOffset(*Offset))
      DumpLocationList(*L);
    return;
  }

  for (const LocationList &L : Locations) {
    DumpLocationList(L);
  }
}
