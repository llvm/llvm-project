//===- COFFEdit.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Edits to the sections of a COFF object that leave every other byte of the
// file as it was, so that the edit can be undone exactly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_COFFEDIT_H
#define LLVM_DEBUGINFO_CVPACK_COFFEDIT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

/// A byte range kept by a section rewrite, used to move relocations.
struct OffsetSpan {
  uint32_t OldOffset = 0;
  uint32_t NewOffset = 0;
  uint32_t Size = 0;
};

struct SectionReplacement {
  /// One-based COFF section index.
  uint32_t SectionIndex = 0;
  std::vector<uint8_t> Data;
  /// Kept spans, ordered by OldOffset. Every relocation of the section must
  /// lie in one of them. Ignored when the section has no relocations.
  std::vector<OffsetSpan> Spans;
};

/// Replace the raw data of some sections, shift every later file offset and
/// move the relocations of the replaced sections through their spans.
LLVM_ABI Expected<std::vector<uint8_t>>
replaceCOFFSections(ArrayRef<uint8_t> Object,
                    ArrayRef<SectionReplacement> Replacements);

/// Append a section without relocations or line numbers after the last one.
/// Its raw data goes just before the symbol table.
LLVM_ABI Expected<std::vector<uint8_t>>
appendCOFFSection(ArrayRef<uint8_t> Object, StringRef Name,
                  uint32_t Characteristics, ArrayRef<uint8_t> Data);

/// Remove the last section, which must have no relocations or line numbers,
/// and its raw data.
LLVM_ABI Expected<std::vector<uint8_t>>
removeLastCOFFSection(ArrayRef<uint8_t> Object);

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_COFFEDIT_H
