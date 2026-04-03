//===- FileEntry.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_FILEENTRY_H
#define LLVM_DEBUGINFO_GSYM_FILEENTRY_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include <functional>
#include <stdint.h>

namespace llvm {
namespace gsym {

/// Files in GSYM are contained in FileEntry structs where we split the
/// directory and basename into two different strings in the string
/// table. This allows paths to shared commont directory and filename
/// strings and saves space.
template <typename GSYM_STRP_T>
struct FileEntry {

  /// Offsets in the string table.
  /// @{
  GSYM_STRP_T Dir = 0;
  GSYM_STRP_T Base = 0;
  /// @}

  FileEntry() = default;
  FileEntry(GSYM_STRP_T D, GSYM_STRP_T B) : Dir(D), Base(B) {}

  // Implement operator== so that FileEntry can be used as key in
  // unordered containers.
  bool operator==(const FileEntry &RHS) const {
    return Base == RHS.Base && Dir == RHS.Dir;
  };
  bool operator!=(const FileEntry &RHS) const {
    return Base != RHS.Base || Dir != RHS.Dir;
  };
};

} // namespace gsym

template <typename GSYM_STRP_T> struct DenseMapInfo<gsym::FileEntry<GSYM_STRP_T>> {
  static inline gsym::FileEntry<GSYM_STRP_T> getEmptyKey() {
    GSYM_STRP_T key = DenseMapInfo<GSYM_STRP_T>::getEmptyKey();
    return gsym::FileEntry<GSYM_STRP_T>(key, key);
  }
  static inline gsym::FileEntry<GSYM_STRP_T> getTombstoneKey() {
    GSYM_STRP_T key = DenseMapInfo<GSYM_STRP_T>::getTombstoneKey();
    return gsym::FileEntry<GSYM_STRP_T>(key, key);
  }
  static unsigned getHashValue(const gsym::FileEntry<GSYM_STRP_T> &Val) {
    return llvm::hash_combine(DenseMapInfo<GSYM_STRP_T>::getHashValue(Val.Dir),
                              DenseMapInfo<GSYM_STRP_T>::getHashValue(Val.Base));
  }
  static bool isEqual(const gsym::FileEntry<GSYM_STRP_T> &LHS,
                      const gsym::FileEntry<GSYM_STRP_T> &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm
#endif // LLVM_DEBUGINFO_GSYM_FILEENTRY_H
