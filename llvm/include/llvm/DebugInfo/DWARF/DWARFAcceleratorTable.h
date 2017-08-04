//===- DWARFAcceleratorTable.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
#define LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include <cstdint>
#include <utility>

namespace llvm {

class raw_ostream;

class DWARFAcceleratorTable {
  struct Header {
    uint32_t Magic;
    uint16_t Version;
    uint16_t HashFunction;
    uint32_t NumBuckets;
    uint32_t NumHashes;
    uint32_t HeaderDataLength;
  };

  struct HeaderData {
    using AtomType = uint16_t;
    using Form = dwarf::Form;

    uint32_t DIEOffsetBase;
    SmallVector<std::pair<AtomType, Form>, 3> Atoms;
  };

  struct Header Hdr;
  struct HeaderData HdrData;
  DWARFDataExtractor AccelSection;
  DataExtractor StringSection;

public:
  DWARFAcceleratorTable(const DWARFDataExtractor &AccelSection,
                        DataExtractor StringSection)
      : AccelSection(AccelSection), StringSection(StringSection) {}

  bool extract();
  uint32_t getNumBuckets();
  uint32_t getNumHashes();
  uint32_t getSizeHdr();
  uint32_t getHeaderDataLength();
  ArrayRef<std::pair<HeaderData::AtomType, HeaderData::Form>> getAtomsDesc();
  bool validateForms();

  /// Return information related to the DWARF DIE we're looking for when
  /// performing a lookup by name.
  ///
  /// \param HashDataOffset an offset into the hash data table
  /// \returns DIEOffset the offset into the .debug_info section for the DIE
  /// related to the input hash data offset. Currently this function returns
  /// only the DIEOffset but it can be modified to return more data regarding
  /// the DIE
  uint32_t readAtoms(uint32_t &HashDataOffset);
  void dump(raw_ostream &OS) const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
