//===- DWARFDataExtractor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/Support/Errc.h"

using namespace llvm;

uint64_t DWARFDataExtractor::getRelocatedValue(uint32_t Size, uint64_t *Off,
                                               uint64_t *SecNdx,
                                               Error *Err) const {
  if (SecNdx)
    *SecNdx = object::SectionedAddress::UndefSection;
  if (!Section)
    return getUnsigned(Off, Size, Err);

  ErrorAsOutParameter ErrAsOut(Err);
  std::optional<RelocAddrEntry> E = Obj->find(*Section, *Off);
  uint64_t LocData = getUnsigned(Off, Size, Err);
  if (!E || (Err && *Err))
    return LocData;
  if (SecNdx)
    *SecNdx = E->SectionIndex;

  uint64_t R =
      object::resolveRelocation(E->Resolver, E->Reloc, E->SymbolValue, LocData);
  if (E->Reloc2)
    R = object::resolveRelocation(E->Resolver, *E->Reloc2, E->SymbolValue2, R);
  return R;
}
