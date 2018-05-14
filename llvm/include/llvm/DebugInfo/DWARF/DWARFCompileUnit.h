//===- DWARFCompileUnit.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H
#define LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnitIndex.h"

namespace llvm {

class DWARFCompileUnit : public DWARFUnit {
public:
  DWARFCompileUnit(DWARFContext &Context, const DWARFSection &Section,
                   const DWARFUnitHeader &Header,
                   const DWARFDebugAbbrev *DA, const DWARFSection *RS,
                   StringRef SS, const DWARFSection &SOS,
                   const DWARFSection *AOS, const DWARFSection &LS, bool LE,
                   bool IsDWO, const DWARFUnitSectionBase &UnitSection)
      : DWARFUnit(Context, Section, Header, DA, RS, SS, SOS, AOS, LS, LE, IsDWO,
                  UnitSection) {}

  // VTable anchor.
  ~DWARFCompileUnit() override;

  void dump(raw_ostream &OS, DIDumpOptions DumpOpts);

  static const DWARFSectionKind Section = DW_SECT_INFO;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H
