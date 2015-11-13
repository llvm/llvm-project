//===-- DWARFUnitIndex.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

bool DWARFUnitIndex::Header::parse(DataExtractor IndexData,
                                   uint32_t *OffsetPtr) {
  if (!IndexData.isValidOffsetForDataOfSize(*OffsetPtr, 16))
    return false;
  Version = IndexData.getU32(OffsetPtr);
  NumColumns = IndexData.getU32(OffsetPtr);
  NumUnits = IndexData.getU32(OffsetPtr);
  NumBuckets = IndexData.getU32(OffsetPtr);
  return Version <= 2;
}

void DWARFUnitIndex::Header::dump(raw_ostream &OS) const {
  OS << format("version = %u slots = %u\n\n", Version, NumBuckets);
}

bool DWARFUnitIndex::parse(DataExtractor IndexData) {
  uint32_t Offset = 0;
  if (!Header.parse(IndexData, &Offset))
    return false;

  if (!IndexData.isValidOffsetForDataOfSize(
          Offset, Header.NumBuckets * (8 + 4) +
                      (2 * Header.NumUnits + 1) * 4 * Header.NumColumns))
    return false;

  Rows = llvm::make_unique<HashRow[]>(Header.NumBuckets);
  auto Contribs =
      llvm::make_unique<HashRow::SectionContribution *[]>(Header.NumUnits);
  ColumnKinds = llvm::make_unique<DwarfSection[]>(Header.NumColumns);

  // Read Hash Table of Signatures
  for (unsigned i = 0; i != Header.NumBuckets; ++i)
    Rows[i].Signature = IndexData.getU64(&Offset);

  // Read Parallel Table of Indexes
  for (unsigned i = 0; i != Header.NumBuckets; ++i) {
    auto Index = IndexData.getU32(&Offset);
    if (!Index)
      continue;
    Rows[i].Contributions =
        llvm::make_unique<HashRow::SectionContribution[]>(Header.NumColumns);
    Contribs[Index - 1] = Rows[i].Contributions.get();
  }

  // Read the Column Headers
  for (unsigned i = 0; i != Header.NumColumns; ++i)
    ColumnKinds[i] = static_cast<DwarfSection>(IndexData.getU32(&Offset));

  // Read Table of Section Offsets
  for (unsigned i = 0; i != Header.NumUnits; ++i) {
    auto *Contrib = Contribs[i];
    for (unsigned i = 0; i != Header.NumColumns; ++i) {
      Contrib[i].Offset = IndexData.getU32(&Offset);
    }
  }

  // Read Table of Section Sizes
  for (unsigned i = 0; i != Header.NumUnits; ++i) {
    auto *Contrib = Contribs[i];
    for (unsigned i = 0; i != Header.NumColumns; ++i) {
      Contrib[i].Size = IndexData.getU32(&Offset);
    }
  }

  return true;
}

StringRef DWARFUnitIndex::getColumnHeader(DwarfSection DS) {
#define CASE(DS)                                                               \
  case DW_SECT_##DS:                                                           \
    return #DS;
  switch (DS) {
    CASE(INFO);
    CASE(TYPES);
    CASE(ABBREV);
    CASE(LINE);
    CASE(LOC);
    CASE(STR_OFFSETS);
    CASE(MACINFO);
    CASE(MACRO);
  }
  llvm_unreachable("unknown DwarfSection");
}

void DWARFUnitIndex::dump(raw_ostream &OS) const {
  Header.dump(OS);
  OS << "Index Signature         ";
  for (unsigned i = 0; i != Header.NumColumns; ++i)
    OS << ' ' << left_justify(getColumnHeader(ColumnKinds[i]), 24);
  OS << "\n----- ------------------";
  for (unsigned i = 0; i != Header.NumColumns; ++i)
    OS << " ------------------------";
  OS << '\n';
  for (unsigned i = 0; i != Header.NumBuckets; ++i) {
    auto &Row = Rows[i];
    if (auto *Contribs = Row.Contributions.get()) {
      OS << format("%5u 0x%016" PRIx64 " ", i, Row.Signature);
      for (unsigned i = 0; i != Header.NumColumns; ++i) {
        auto &Contrib = Contribs[i];
        OS << format("[0x%08u, 0x%08u) ", Contrib.Offset,
                     Contrib.Offset + Contrib.Size);
      }
      OS << '\n';
    }
  }
}
}
