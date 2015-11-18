//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFCOMPILEUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFCOMPILEUNIT_H

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

namespace llvm {

class DWARFCompileUnit : public DWARFUnit {
public:
  DWARFCompileUnit(DWARFContext &Context, const DWARFSection &Section,
                   const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                   StringRef SOS, StringRef AOS, StringRef LS, bool LE,
                   const DWARFUnitSectionBase &UnitSection,
                   const DWARFUnitIndex::Entry *Entry)
      : DWARFUnit(Context, Section, DA, RS, SS, SOS, AOS, LS, LE, UnitSection,
                  Entry) {}
  void dump(raw_ostream &OS);
  static const DWARFSectionKind Section = DW_SECT_INFO;
  // VTable anchor.
  ~DWARFCompileUnit() override;
};

}

#endif
