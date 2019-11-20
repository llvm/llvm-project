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
using object::SectionedAddress;

namespace {
class DWARFLocationInterpreter {
  Optional<object::SectionedAddress> Base;
  std::function<Optional<object::SectionedAddress>(uint32_t)> LookupAddr;

public:
  DWARFLocationInterpreter(
      Optional<object::SectionedAddress> Base,
      std::function<Optional<object::SectionedAddress>(uint32_t)> LookupAddr)
      : Base(Base), LookupAddr(std::move(LookupAddr)) {}

  Expected<Optional<DWARFLocationExpression>>
  Interpret(const DWARFLocationEntry &E);
};
} // namespace

static Error createResolverError(uint32_t Index, unsigned Kind) {
  return createStringError(errc::invalid_argument,
                           "Unable to resolve indirect address %u for: %s",
                           Index, dwarf::LocListEncodingString(Kind).data());
}

Expected<Optional<DWARFLocationExpression>>
DWARFLocationInterpreter::Interpret(const DWARFLocationEntry &E) {
  switch (E.Kind) {
  case dwarf::DW_LLE_end_of_list:
    return None;
  case dwarf::DW_LLE_base_addressx: {
    Base = LookupAddr(E.Value0);
    if (!Base)
      return createResolverError(E.Value0, E.Kind);
    return None;
  }
  case dwarf::DW_LLE_startx_length: {
    Optional<SectionedAddress> LowPC = LookupAddr(E.Value0);
    if (!LowPC)
      return createResolverError(E.Value0, E.Kind);
    return DWARFLocationExpression{DWARFAddressRange{LowPC->Address,
                                                     LowPC->Address + E.Value1,
                                                     LowPC->SectionIndex},
                                   E.Loc};
  }
  case dwarf::DW_LLE_offset_pair:
    if (!Base) {
      return createStringError(
          inconvertibleErrorCode(),
          "Unable to resolve DW_LLE_offset_pair: base address unknown");
    }
    return DWARFLocationExpression{DWARFAddressRange{Base->Address + E.Value0,
                                                     Base->Address + E.Value1,
                                                     Base->SectionIndex},
                                   E.Loc};
  case dwarf::DW_LLE_base_address:
    Base = SectionedAddress{E.Value0, SectionedAddress::UndefSection};
    return None;
  case dwarf::DW_LLE_start_length:
    return DWARFLocationExpression{
        DWARFAddressRange{E.Value0, E.Value0 + E.Value1,
                          SectionedAddress::UndefSection},
        E.Loc};
  default:
    llvm_unreachable("unreachable locations list kind");
  }
}

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

bool DWARFLocationTable::dumpLocationList(uint64_t *Offset, raw_ostream &OS,
                                          Optional<SectionedAddress> BaseAddr,
                                          const MCRegisterInfo *MRI,
                                          DWARFUnit *U, DIDumpOptions DumpOpts,
                                          unsigned Indent) const {
  DWARFLocationInterpreter Interp(
      BaseAddr, [U](uint32_t Index) -> Optional<SectionedAddress> {
        if (U)
          return U->getAddrOffsetSectionItem(Index);
        return None;
      });
  OS << format("0x%8.8" PRIx64 ": ", *Offset);
  Error E = visitLocationList(Offset, [&](const DWARFLocationEntry &E) {
    Expected<Optional<DWARFLocationExpression>> Loc = Interp.Interpret(E);
    if (!Loc || DumpOpts.DisplayRawContents)
      dumpRawEntry(E, OS, Indent);
    if (Loc && *Loc) {
      OS << "\n";
      OS.indent(Indent);
      if (DumpOpts.DisplayRawContents)
        OS << "          => ";

      DIDumpOptions RangeDumpOpts(DumpOpts);
      RangeDumpOpts.DisplayRawContents = false;
      const DWARFObject *Obj = nullptr;
      if (U)
        Obj = &U->getContext().getDWARFObj();
      Loc.get()->Range->dump(OS, Data.getAddressSize(), RangeDumpOpts, Obj);
    }
    if (!Loc)
      consumeError(Loc.takeError());

    if (E.Kind != dwarf::DW_LLE_base_address &&
        E.Kind != dwarf::DW_LLE_base_addressx &&
        E.Kind != dwarf::DW_LLE_end_of_list) {
      OS << ": ";
      dumpExpression(OS, E.Loc, Data.isLittleEndian(), Data.getAddressSize(),
                     MRI, U);
    }
    return true;
  });
  if (E) {
    OS << "\n";
    OS.indent(Indent);
    OS << "error: " << toString(std::move(E));
    return false;
  }
  return true;
}

DWARFDebugLoc::LocationList const *
DWARFDebugLoc::getLocationListAtOffset(uint64_t Offset) const {
  auto It = partition_point(
      Locations, [=](const LocationList &L) { return L.Offset < Offset; });
  if (It != Locations.end() && It->Offset == Offset)
    return &(*It);
  return nullptr;
}

void DWARFDebugLoc::dump(raw_ostream &OS, const MCRegisterInfo *MRI,
                         DIDumpOptions DumpOpts,
                         Optional<uint64_t> DumpOffset) const {
  auto BaseAddr = None;
  unsigned Indent = 12;
  if (DumpOffset) {
    dumpLocationList(&*DumpOffset, OS, BaseAddr, MRI, nullptr, DumpOpts,
                     Indent);
  } else {
    uint64_t Offset = 0;
    StringRef Separator;
    bool CanContinue = true;
    while (CanContinue && Data.isValidOffset(Offset)) {
      OS << Separator;
      Separator = "\n";

      CanContinue = dumpLocationList(&Offset, OS, BaseAddr, MRI, nullptr,
                                     DumpOpts, Indent);
      OS << '\n';
    }
  }
}

Error DWARFDebugLoc::visitLocationList(
    uint64_t *Offset,
    function_ref<bool(const DWARFLocationEntry &)> Callback) const {
  DataExtractor::Cursor C(*Offset);
  while (true) {
    uint64_t Value0 = Data.getRelocatedAddress(C);
    uint64_t Value1 = Data.getRelocatedAddress(C);

    DWARFLocationEntry E;

    // The end of any given location list is marked by an end of list entry,
    // which consists of a 0 for the beginning address offset and a 0 for the
    // ending address offset. A beginning offset of 0xff...f marks the base
    // address selection entry.
    if (Value0 == 0 && Value1 == 0) {
      E.Kind = dwarf::DW_LLE_end_of_list;
    } else if (Value0 == (Data.getAddressSize() == 4 ? -1U : -1ULL)) {
      E.Kind = dwarf::DW_LLE_base_address;
      E.Value0 = Value1;
    } else {
      E.Kind = dwarf::DW_LLE_offset_pair;
      E.Value0 = Value0;
      E.Value1 = Value1;
      unsigned Bytes = Data.getU16(C);
      // A single location description describing the location of the object...
      Data.getU8(C, E.Loc, Bytes);
    }

    if (!C)
      return C.takeError();
    if (!Callback(E) || E.Kind == dwarf::DW_LLE_end_of_list)
      break;
  }
  *Offset = C.tell();
  return Error::success();
}

Expected<DWARFDebugLoc::LocationList>
DWARFDebugLoc::parseOneLocationList(uint64_t *Offset) {
  LocationList LL;
  LL.Offset = *Offset;

  Error E = visitLocationList(Offset, [&](const DWARFLocationEntry &E) {
    LL.Entries.push_back(E);
    return true;
  });
  if (E)
    return std::move(E);
  return std::move(LL);
}

void DWARFDebugLoc::parse() {
  uint64_t Offset = 0;
  while (Offset < Data.getData().size()) {
    if (auto LL = parseOneLocationList(&Offset))
      Locations.push_back(std::move(*LL));
    else {
      logAllUnhandledErrors(LL.takeError(), WithColor::error());
      break;
    }
  }
}

void DWARFDebugLoc::dumpRawEntry(const DWARFLocationEntry &Entry,
                                 raw_ostream &OS, unsigned Indent) const {
  uint64_t Value0, Value1;
  switch (Entry.Kind) {
  case dwarf::DW_LLE_base_address:
    Value0 = Data.getAddressSize() == 4 ? -1U : -1ULL;
    Value1 = Entry.Value0;
    break;
  case dwarf::DW_LLE_offset_pair:
    Value0 = Entry.Value0;
    Value1 = Entry.Value1;
    break;
  case dwarf::DW_LLE_end_of_list:
    Value0 = Value1 = 0;
    return;
  default:
    llvm_unreachable("Not possible in DWARF4!");
  }
  OS << '\n';
  OS.indent(Indent);
  OS << '(' << format_hex(Value0, 2 + Data.getAddressSize() * 2) << ", "
     << format_hex(Value1, 2 + Data.getAddressSize() * 2) << ')';
}

Error DWARFDebugLoclists::visitLocationList(
    uint64_t *Offset, function_ref<bool(const DWARFLocationEntry &)> F) const {

  DataExtractor::Cursor C(*Offset);
  bool Continue = true;
  while (Continue) {
    DWARFLocationEntry E;
    E.Kind = Data.getU8(C);
    switch (E.Kind) {
    case dwarf::DW_LLE_end_of_list:
      break;
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
    case dwarf::DW_LLE_offset_pair:
      E.Value0 = Data.getULEB128(C);
      E.Value1 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_base_address:
      E.Value0 = Data.getRelocatedAddress(C);
      break;
    case dwarf::DW_LLE_start_length:
      E.Value0 = Data.getRelocatedAddress(C);
      E.Value1 = Data.getULEB128(C);
      break;
    case dwarf::DW_LLE_startx_endx:
    case dwarf::DW_LLE_default_location:
    case dwarf::DW_LLE_start_end:
    default:
      cantFail(C.takeError());
      return createStringError(errc::illegal_byte_sequence,
                               "LLE of kind %x not supported", (int)E.Kind);
    }

    if (E.Kind != dwarf::DW_LLE_base_address &&
        E.Kind != dwarf::DW_LLE_base_addressx &&
        E.Kind != dwarf::DW_LLE_end_of_list) {
      unsigned Bytes = Version >= 5 ? Data.getULEB128(C) : Data.getU16(C);
      // A single location description describing the location of the object...
      Data.getU8(C, E.Loc, Bytes);
    }

    if (!C)
      return C.takeError();
    Continue = F(E) && E.Kind != dwarf::DW_LLE_end_of_list;
  }
  *Offset = C.tell();
  return Error::success();
}

void DWARFDebugLoclists::dumpRawEntry(const DWARFLocationEntry &Entry,
                                      raw_ostream &OS, unsigned Indent) const {
  size_t MaxEncodingStringLength = 0;
#define HANDLE_DW_LLE(ID, NAME)                                                \
  MaxEncodingStringLength = std::max(MaxEncodingStringLength,                  \
                                     dwarf::LocListEncodingString(ID).size());
#include "llvm/BinaryFormat/Dwarf.def"

  OS << "\n";
  OS.indent(Indent);
  StringRef EncodingString = dwarf::LocListEncodingString(Entry.Kind);
  // Unsupported encodings should have been reported during parsing.
  assert(!EncodingString.empty() && "Unknown loclist entry encoding");
  OS << format("%-*s(", MaxEncodingStringLength, EncodingString.data());
  unsigned FieldSize = 2 + 2 * Data.getAddressSize();
  switch (Entry.Kind) {
  case dwarf::DW_LLE_startx_length:
  case dwarf::DW_LLE_start_length:
  case dwarf::DW_LLE_offset_pair:
    OS << format_hex(Entry.Value0, FieldSize) << ", "
       << format_hex(Entry.Value1, FieldSize);
    break;
  case dwarf::DW_LLE_base_addressx:
  case dwarf::DW_LLE_base_address:
    OS << format_hex(Entry.Value0, FieldSize);
    break;
  case dwarf::DW_LLE_end_of_list:
    break;
  }
  OS << ')';
}

void DWARFDebugLoclists::dumpRange(uint64_t StartOffset, uint64_t Size,
                                   raw_ostream &OS, const MCRegisterInfo *MRI,
                                   DIDumpOptions DumpOpts) {
  if (!Data.isValidOffsetForDataOfSize(StartOffset, Size))  {
    OS << "Invalid dump range\n";
    return;
  }
  uint64_t Offset = StartOffset;
  StringRef Separator;
  bool CanContinue = true;
  while (CanContinue && Offset < StartOffset + Size) {
    OS << Separator;
    Separator = "\n";

    CanContinue = dumpLocationList(&Offset, OS, /*BaseAddr=*/None, MRI, nullptr,
                                   DumpOpts, /*Indent=*/12);
    OS << '\n';
  }
}
