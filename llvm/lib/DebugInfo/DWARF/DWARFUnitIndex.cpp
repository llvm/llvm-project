//===-- DWARFUnitIndex.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"

namespace llvm {

bool DWARFUnitIndex::Header::parse(DataExtractor IndexData,
                                   uint32_t *OffsetPtr) {
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

  return true;
}

void DWARFUnitIndex::dump(raw_ostream &OS) const { Header.dump(OS); }
}
