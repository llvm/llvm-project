//===- DWARFTypeUnit.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFTYPEUNIT_H
#define LLVM_DEBUGINFO_DWARF_DWARFTYPEUNIT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"
#include "llvm/Support/DataExtractor.h"
#include <cstdint>

namespace llvm {

class DWARFContext;
class DWARFDebugAbbrev;
struct DWARFSection;
class raw_ostream;

class DWARFTypeUnit : public DWARFUnit {
private:
  uint64_t TypeHash;
  uint32_t TypeOffset;

public:
  DWARFTypeUnit(DWARFContext &Context, const DWARFSection &Section,
                const DWARFDebugAbbrev *DA, const DWARFSection *RS,
                StringRef SS, const DWARFSection &SOS, const DWARFSection *AOS,
                const DWARFSection &LS, bool LE, bool IsDWO,
                const DWARFUnitSectionBase &UnitSection,
                const DWARFUnitIndex::Entry *Entry)
      : DWARFUnit(Context, Section, DA, RS, SS, SOS, AOS, LS, LE, IsDWO,
                  UnitSection, Entry) {}

  uint32_t getHeaderSize() const override {
    return DWARFUnit::getHeaderSize() + 12;
  }

  void dump(raw_ostream &OS, bool Brief = false);
  static const DWARFSectionKind Section = DW_SECT_TYPES;

protected:
  bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) override;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFTYPEUNIT_H
