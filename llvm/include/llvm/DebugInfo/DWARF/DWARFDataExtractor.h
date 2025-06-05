//===- DWARFDataExtractor.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTOR_H
#define LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTOR_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractorSimple.h"
#include "llvm/DebugInfo/DWARF/DWARFObject.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// A DWARFDataExtractor (typically for an in-memory copy of an object-file
/// section) plus a relocation map for that section, if there is one.
class DWARFDataExtractor : public DWARFDataExtractorBase<DWARFDataExtractor> {
  const DWARFObject *Obj = nullptr;
  const DWARFSection *Section = nullptr;

public:
  using DWARFDataExtractorBase::DWARFDataExtractorBase;

  /// Constructor for the normal case of extracting data from a DWARF section.
  /// The DWARFSection's lifetime must be at least as long as the extractor's.
  DWARFDataExtractor(const DWARFObject &Obj, const DWARFSection &Section,
                     bool IsLittleEndian, uint8_t AddressSize)
      : DWARFDataExtractorBase(Section.Data, IsLittleEndian, AddressSize),
        Obj(&Obj), Section(&Section) {}

  /// Truncating constructor
  DWARFDataExtractor(const DWARFDataExtractor &Other, size_t Length)
      : DWARFDataExtractorBase(Other.getData().substr(0, Length),
                               Other.isLittleEndian(), Other.getAddressSize()),
        Obj(Other.Obj), Section(Other.Section) {}

  /// Extracts the DWARF "initial length" field, which can either be a 32-bit
  /// value smaller than 0xfffffff0, or the value 0xffffffff followed by a
  /// 64-bit length. Returns the actual length, and the DWARF format which is
  /// encoded in the field. In case of errors, it returns {0, DWARF32} and
  /// leaves the offset unchanged.
  LLVM_ABI std::pair<uint64_t, dwarf::DwarfFormat>
  getInitialLength(uint64_t *Off, Error *Err = nullptr) const;

  std::pair<uint64_t, dwarf::DwarfFormat> getInitialLength(Cursor &C) const {
    return getInitialLength(&getOffset(C), &getError(C));
  }

  /// Extracts a value and applies a relocation to the result if
  /// one exists for the given offset.
  LLVM_ABI uint64_t getRelocatedValue(uint32_t Size, uint64_t *Off,
                                      uint64_t *SectionIndex = nullptr,
                                      Error *Err = nullptr) const;
  LLVM_ABI uint64_t getRelocatedValue(Cursor &C, uint32_t Size,
                                      uint64_t *SectionIndex = nullptr) const {
    return getRelocatedValue(Size, &getOffset(C), SectionIndex, &getError(C));
  }

  /// Extracts a value and applies a relocation to the result if
  /// one exists for the given offset.
  LLVM_ABI uint64_t getRelocatedValueImpl(uint32_t Size, uint64_t *Off,
                                          uint64_t *SecNdx, Error *Err) const {
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

    uint64_t R = object::resolveRelocation(E->Resolver, E->Reloc,
                                           E->SymbolValue, LocData);
    if (E->Reloc2)
      R = object::resolveRelocation(E->Resolver, *E->Reloc2, E->SymbolValue2,
                                    R);
    return R;
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTOR_H
