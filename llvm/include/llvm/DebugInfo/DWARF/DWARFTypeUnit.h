//===-- DWARFTypeUnit.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFTYPEUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFTYPEUNIT_H

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

namespace llvm {

class DWARFTypeUnit : public DWARFUnit {
private:
  uint64_t TypeHash;
  uint32_t TypeOffset;
public:
  DWARFTypeUnit(DWARFContext &Context, const DWARFSection &Section,
                const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                StringRef SOS, StringRef AOS, StringRef LS, bool LE, bool IsDWO,
                const DWARFUnitSectionBase &UnitSection,
                const DWARFUnitIndex::Entry *Entry)
      : DWARFUnit(Context, Section, DA, RS, SS, SOS, AOS, LS, LE, IsDWO,
                  UnitSection, Entry) {}
  uint32_t getHeaderSize() const override {
    return DWARFUnit::getHeaderSize() + 12;
  }
  void dump(raw_ostream &OS);
  static const DWARFSectionKind Section = DW_SECT_TYPES;

protected:
  bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) override;
};

}

#endif

