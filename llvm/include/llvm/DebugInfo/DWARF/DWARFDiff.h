//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFDIFF_H
#define LLVM_DEBUGINFO_DWARF_DWARFDIFF_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/Support/Compiler.h"
#include <string>
#include <utility>

namespace llvm {

class DWARFContext;
class raw_ostream;

struct DiffOptions {
  bool IgnoreLines = false;
};

struct DiffInput {
  DWARFContext &Context;
  StringRef Filename;
};

/// Lightweight identifier for a DIE, storing its offset, tag, and qualified
/// name. Used in diff results so the comparison engine does not retain
/// DWARFDie handles.
struct DiffDIERef {
  uint64_t Offset;
  dwarf::Tag Tag;
  std::string QualifiedName;
};

/// A pair of DiffDIERefs identifying a difference between matched DIEs.
struct DiffEntry {
  DiffDIERef LHS, RHS;
};

/// Result of a semantic DWARF comparison.
struct DiffResult {
  unsigned NumMatched = 0;

  SmallVector<DiffDIERef> OnlyInLHS;
  SmallVector<DiffDIERef> OnlyInRHS;
  SmallVector<DiffEntry> Different;

  bool hasDifferences() const {
    return !OnlyInLHS.empty() || !OnlyInRHS.empty() || !Different.empty();
  }
};

/// Identity key for matching DIEs across files.
struct DIEIdentity {
  std::string QualifiedName;
  dwarf::Tag Tag = dwarf::DW_TAG_null;

  bool operator==(const DIEIdentity &O) const {
    return Tag == O.Tag && QualifiedName == O.QualifiedName;
  }
  bool isValid() const { return Tag != dwarf::DW_TAG_null; }
};

} // namespace llvm

namespace llvm {
template <> struct DenseMapInfo<DIEIdentity> {
  static DIEIdentity getEmptyKey() {
    return {{}, static_cast<dwarf::Tag>(~0U)};
  }
  static DIEIdentity getTombstoneKey() {
    return {{}, static_cast<dwarf::Tag>(~0U - 1)};
  }
  static unsigned getHashValue(const DIEIdentity &V) {
    return hash_combine(hash_value(V.QualifiedName), hash_value(V.Tag));
  }
  static bool isEqual(const DIEIdentity &LHS, const DIEIdentity &RHS) {
    return LHS.Tag == RHS.Tag && LHS.QualifiedName == RHS.QualifiedName;
  }
};

using DIEIndexMap = DenseMap<DIEIdentity, SmallVector<DWARFDie, 2>>;
using CUMap = DenseMap<DIEIdentity, DWARFDie>;

/// Semantic DWARF diff engine. Compares per-CU, handles cross-CU type
/// moves, forward declarations, and identity-based references.
class LLVM_ABI DWARFDiff {
public:
  explicit DWARFDiff(const DiffOptions &Opts) : Opts(Opts) {}

  /// Compare two DWARF inputs and return the result.
  DiffResult diff(const DiffInput &LHS, const DiffInput &RHS);

private:
  const DiffOptions Opts;

  bool isSkippedAttribute(dwarf::Attribute Attr) const;
  std::string getTypeKey(DWARFDie Die, DenseSet<uint64_t> &Visited,
                         unsigned Depth = 0);
  std::string getQualifiedName(DWARFDie Die);
  DIEIdentity getIdentity(DWARFDie Die);
  bool compareTypes(DWARFDie LHS, DWARFDie RHS,
                    DenseSet<std::pair<uint64_t, uint64_t>> &Visited);
  bool compareDIEs(DWARFDie LHS, DWARFDie RHS);
  CUMap collectCUs(DWARFContext &Ctx);
  void indexDIE(DWARFDie Die, DIEIndexMap &Index);
  DIEIndexMap buildCUIndex(DWARFDie UnitDie);
  DiffDIERef makeDIERef(DWARFDie Die);
};

/// Formats and prints DiffResult as a structured table.
LLVM_ABI void printDiffResult(raw_ostream &OS, const DiffResult &Result);

} // namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFDIFF_H
