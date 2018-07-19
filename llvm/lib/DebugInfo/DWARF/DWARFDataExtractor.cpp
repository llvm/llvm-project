//===- DWARFDataExtractor.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

using namespace llvm;

uint64_t DWARFDataExtractor::getRelocatedValue(uint32_t Size, uint32_t *Off,
                                               uint64_t *SecNdx) const {
  if (SecNdx)
    *SecNdx = -1ULL;
  if (!Section)
    return getUnsigned(Off, Size);
  Optional<RelocAddrEntry> Rel = Obj->find(*Section, *Off);
  if (!Rel)
    return getUnsigned(Off, Size);
  if (SecNdx)
    *SecNdx = Rel->SectionIndex;
  return getUnsigned(Off, Size) + Rel->Value;
}
