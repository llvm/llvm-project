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
  OS << "Index header:\n" << format("   version: %u\n", Version)
     << format("   columns: %u\n", NumColumns)
     << format("     units: %u\n", NumUnits)
     << format("   buckets: %u\n", NumBuckets);
}

bool DWARFUnitIndex::parse(DataExtractor IndexData) {
  uint32_t Offset = 0;
  if (!Header.parse(IndexData, &Offset))
    return false;

  if (!IndexData.isValidOffsetForDataOfSize(
          Offset, Header.NumBuckets * (8 + 4) +
                      (2 * Header.NumUnits + 1) * 4 * Header.NumColumns))
    return false;

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
}

}
