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
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/Support/DataExtractor.h"
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
    typedef uint16_t AtomType;
    typedef dwarf::Form Form;
    uint32_t DIEOffsetBase;
    SmallVector<std::pair<AtomType, Form>, 3> Atoms;
  };

  struct Header Hdr;
  struct HeaderData HdrData;
  DataExtractor AccelSection;
  DataExtractor StringSection;
  const RelocAddrMap& Relocs;

public:
  DWARFAcceleratorTable(DataExtractor AccelSection, DataExtractor StringSection,
                        const RelocAddrMap &Relocs)
    : AccelSection(AccelSection), StringSection(StringSection), Relocs(Relocs) {}

  bool extract();
  uint32_t getNumBuckets();
  uint32_t getNumHashes();
  uint32_t getSizeHdr();
  uint32_t getHeaderDataLength();
  void dump(raw_ostream &OS) const;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFACCELERATORTABLE_H
