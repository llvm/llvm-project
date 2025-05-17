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
#include "llvm/DebugInfo/DWARF/DWARFSection.h"

namespace llvm {
class DWARFObject;

/// A DWARFDataExtractorSimple (typically for an in-memory copy of an
/// object-file section) plus a relocation map for that section, if there is
/// one.
class DWARFDataExtractor : public DWARFDataExtractorSimple {
  const DWARFObject *Obj = nullptr;
  const DWARFSection *Section = nullptr;

public:
  using DWARFDataExtractorSimple::DWARFDataExtractorSimple;

  /// Constructor for the normal case of extracting data from a DWARF section.
  /// The DWARFSection's lifetime must be at least as long as the extractor's.
  DWARFDataExtractor(const DWARFObject &Obj, const DWARFSection &Section,
                     bool IsLittleEndian, uint8_t AddressSize)
      : DWARFDataExtractorSimple(Section.Data, IsLittleEndian, AddressSize),
        Obj(&Obj), Section(&Section) {}

  /// Truncating constructor
  DWARFDataExtractor(const DWARFDataExtractor &Other, size_t Length)
      : DWARFDataExtractorSimple(Other.getData().substr(0, Length),
                                 Other.isLittleEndian(),
                                 Other.getAddressSize()),
        Obj(Other.Obj), Section(Other.Section) {}

  /// Extracts a value and applies a relocation to the result if
  /// one exists for the given offset.
  uint64_t getRelocatedValue(uint32_t Size, uint64_t *Off,
                             uint64_t *SectionIndex = nullptr,
                             Error *Err = nullptr) const override;

  uint64_t getRelocatedValue(Cursor &C, uint32_t Size,
                             uint64_t *SectionIndex = nullptr) const override {
    return getRelocatedValue(Size, &getOffset(C), SectionIndex, &getError(C));
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDATAEXTRACTOR_H
