//===- COFFSectionTable.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_CVPACK_COFFSECTIONTABLE_H
#define LLVM_LIB_DEBUGINFO_CVPACK_COFFSECTIONTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {
namespace codeview {
namespace cvpack {

/// Where the headers of a regular or bigobj COFF object are, for the code
/// that rewrites them in place.
struct COFFLayout {
  bool IsBigObj = false;
  uint16_t Machine = 0;
  size_t SectionTableOffset = 0;
  uint32_t SectionCount = 0;
  /// Offset of the PointerToSymbolTable field in the file header.
  size_t SymbolTablePointerOffset = 0;
  /// The file offset of the symbol table, or 0.
  uint32_t SymbolTable = 0;
};

/// The sections of a COFF object file, read through object::COFFObjectFile.
/// Sections are numbered from one, as in COFF. The bytes of the object must
/// outlive the table.
class COFFSectionTable {
public:
  static Expected<COFFSectionTable> create(ArrayRef<uint8_t> Object);
  const object::COFFObjectFile &object() const { return *Obj; }
  const COFFLayout &layout() const { return Layout; }
  uint32_t size() const { return Layout.SectionCount; }
  /// The offset of a section's header in the object.
  size_t headerOffset(uint32_t Index) const;
  const object::coff_section &header(uint32_t Index) const;
  /// The name of a section, or an empty name if it cannot be read.
  StringRef name(uint32_t Index) const;
  bool isComdat(uint32_t Index) const;
  /// The raw data of a section: empty if it has none, nullopt if the header
  /// places it outside the object.
  std::optional<ArrayRef<uint8_t>> contents(uint32_t Index) const;
  /// The relocations of a section: nullopt if they lie outside the object,
  /// or if their count overflows into the first entry.
  std::optional<ArrayRef<object::coff_relocation>>
  relocations(uint32_t Index) const;

private:
  std::unique_ptr<object::COFFObjectFile> Obj;
  COFFLayout Layout;
};

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_LIB_DEBUGINFO_CVPACK_COFFSECTIONTABLE_H
