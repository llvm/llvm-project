//===- DWARFDataExtractor.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDATAEXTRACTOR_H
#define LLVM_DEBUGINFO_DWARFDATAEXTRACTOR_H

#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

/// A DataExtractor (typically for an in-memory copy of an object-file section)
/// plus a relocation map for that section, if there is one.
class DWARFDataExtractor : public DataExtractor {
  const RelocAddrMap *RelocMap = nullptr;
public:
  /// Constructor for the normal case of extracting data from a DWARF section.
  /// The DWARFSection's lifetime must be at least as long as the extractor's.
  DWARFDataExtractor(const DWARFSection &Section, bool IsLittleEndian,
                     uint8_t AddressSize)
    : DataExtractor(Section.Data, IsLittleEndian, AddressSize),
      RelocMap(&Section.Relocs) {}

  /// Constructor for cases when there are no relocations.
  DWARFDataExtractor(StringRef Data, bool IsLittleEndian, uint8_t AddressSize)
    : DataExtractor(Data, IsLittleEndian, AddressSize) {}

  /// Extracts a value and applies a relocation to the result if
  /// one exists for the given offset.
  uint64_t getRelocatedValue(uint32_t Size, uint32_t *Off,
                             uint64_t *SectionIndex = nullptr) const;

  /// Extracts an address-sized value and applies a relocation to the result if
  /// one exists for the given offset.
  uint64_t getRelocatedAddress(uint32_t *Off, uint64_t *SecIx = nullptr) const {
    return getRelocatedValue(getAddressSize(), Off, SecIx);
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFDATAEXTRACTOR_H
