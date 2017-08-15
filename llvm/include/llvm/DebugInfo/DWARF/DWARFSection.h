//===- DWARFSection.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFSECTION_H
#define LLVM_DEBUGINFO_DWARF_DWARFSECTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"

namespace llvm {

struct DWARFSection {
  StringRef Data;
};

struct SectionName {
  StringRef Name;
  bool IsNameUnique;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFSECTION_H
