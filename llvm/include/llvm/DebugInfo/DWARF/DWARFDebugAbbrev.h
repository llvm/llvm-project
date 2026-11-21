//===- DWARFDebugAbbrev.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDEBUGABBREV_H
#define LLVM_DEBUGINFO_DWARF_DWARFDEBUGABBREV_H

#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace llvm {

class raw_ostream;

/// Read the next (attribute, form) specification from an abbreviation
/// declaration at \p Offset, advancing \p Offset past it. \p ImplicitConst is
/// set to the inline value of a DW_FORM_implicit_const attribute and to
/// std::nullopt otherwise. Returns false on the terminating (0, 0) pair.
LLVM_ABI bool readAbbrevAttribute(const DataExtractor &AbbrevData,
                                  uint64_t *Offset, dwarf::Attribute &Name,
                                  dwarf::Form &Form,
                                  std::optional<int64_t> &ImplicitConst);

class DWARFAbbreviationDeclarationSet {
  uint64_t Offset;
  /// Code of the first abbreviation, if all abbreviations in the set have
  /// consecutive codes. UINT32_MAX otherwise.
  uint32_t FirstAbbrCode;
  std::vector<DWARFAbbreviationDeclaration> Decls;

  using const_iterator =
      std::vector<DWARFAbbreviationDeclaration>::const_iterator;

public:
  LLVM_ABI DWARFAbbreviationDeclarationSet();

  uint64_t getOffset() const { return Offset; }
  LLVM_ABI void dump(raw_ostream &OS) const;
  LLVM_ABI Error extract(DataExtractor Data, uint64_t *OffsetPtr);

  LLVM_ABI const DWARFAbbreviationDeclaration *
  getAbbreviationDeclaration(uint32_t AbbrCode) const;

  const_iterator begin() const {
    return Decls.begin();
  }

  const_iterator end() const {
    return Decls.end();
  }

  LLVM_ABI std::string getCodeRange() const;

  uint32_t getFirstAbbrCode() const { return FirstAbbrCode; }

private:
  void clear();
};

class DWARFDebugAbbrev {
  using DWARFAbbreviationDeclarationSetMap =
      std::map<uint64_t, DWARFAbbreviationDeclarationSet>;

  mutable DWARFAbbreviationDeclarationSetMap AbbrDeclSets;
  mutable DWARFAbbreviationDeclarationSetMap::const_iterator PrevAbbrOffsetPos;
  mutable std::optional<DataExtractor> Data;

public:
  LLVM_ABI DWARFDebugAbbrev(DataExtractor Data);

  LLVM_ABI Expected<const DWARFAbbreviationDeclarationSet *>
  getAbbreviationDeclarationSet(uint64_t CUAbbrOffset) const;

  LLVM_ABI void dump(raw_ostream &OS) const;
  LLVM_ABI Error parse() const;

  DWARFAbbreviationDeclarationSetMap::const_iterator begin() const {
    assert(!Data && "Must call parse before iterating over DWARFDebugAbbrev");
    return AbbrDeclSets.begin();
  }

  DWARFAbbreviationDeclarationSetMap::const_iterator end() const {
    return AbbrDeclSets.end();
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDEBUGABBREV_H
