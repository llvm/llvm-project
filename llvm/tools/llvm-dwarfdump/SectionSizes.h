//===- SectionSizes.h - Debug section sizes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#ifndef LLVM_TOOLS_SECTION_SIZES_H
#define LLVM_TOOLS_SECTION_SIZES_H

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/WithColor.h"

namespace llvm {

using SectionSizeMap = StringMap<uint64_t>;

/// Holds cumulative section sizes for an object file.
struct SectionSizes {
  /// Map of .debug section names and their sizes across all such-named
  /// sections.
  SectionSizeMap DebugSectionSizes;
  /// Total number of bytes of all sections.
  uint64_t TotalObjectSize = 0;
  /// Total number of bytes of all debug sections.
  uint64_t TotalDebugSectionsSize = 0;
};

/// Calculate the section sizes.
void calculateSectionSizes(const object::ObjectFile &Obj, SectionSizes &Sizes,
                           const Twine &Filename);

} // end namespace llvm

#endif // LLVM_TOOLS_SECTION_SIZES_H
