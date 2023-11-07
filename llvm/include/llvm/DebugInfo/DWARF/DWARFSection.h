//===- DWARFSection.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFSECTION_H
#define LLVM_DEBUGINFO_DWARF_DWARFSECTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

struct LLVM_CLASS_ABI DWARFSection {
  StringRef Data;
  uint64_t Address = 0;
};

struct LLVM_CLASS_ABI SectionName {
  StringRef Name;
  bool IsNameUnique;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFSECTION_H
