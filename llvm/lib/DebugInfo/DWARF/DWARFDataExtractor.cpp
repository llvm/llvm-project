//===- DWARFDataExtractor.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"

using namespace llvm;

uint64_t DWARFDataExtractor::getRelocatedValue(uint32_t Size, uint32_t *Off,
                                               uint64_t *SecNdx) const {
  if (!RelocMap)
    return getUnsigned(Off, Size);
  RelocAddrMap::const_iterator AI = RelocMap->find(*Off);
  if (AI == RelocMap->end())
    return getUnsigned(Off, Size);
  if (SecNdx)
    *SecNdx = AI->second.SectionIndex;
  return getUnsigned(Off, Size) + AI->second.Value;
}
