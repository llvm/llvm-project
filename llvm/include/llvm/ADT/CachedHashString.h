//===- llvm/ADT/CachedHashString.h - Prehashed string/StringRef -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines CachedHashString and CachedHashStringRef.  These are like
// std::string and StringRef, except they store their hash in addition to their
// string data.
//
// Unlike std::string, CachedHashString can be used in DenseSet/DenseMap
// (because, unlike std::string, CachedHashString lets us have empty and
// tombstone values).
//
// TODO: Add CachedHashString.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_CACHED_HASH_STRING_H
#define LLVM_ADT_CACHED_HASH_STRING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

/// A container which contains a StringRef plus a precomputed hash.
class CachedHashStringRef {
  const char *P;
  uint32_t Size;
  uint32_t Hash;

public:
  // Explicit because hashing a string isn't free.
  explicit CachedHashStringRef(StringRef S)
      : CachedHashStringRef(S, DenseMapInfo<StringRef>::getHashValue(S)) {}

  CachedHashStringRef(StringRef S, uint32_t Hash)
      : P(S.data()), Size(S.size()), Hash(Hash) {
    assert(S.size() <= std::numeric_limits<uint32_t>::max());
  }

  StringRef val() const { return StringRef(P, Size); }
  uint32_t size() const { return Size; }
  uint32_t hash() const { return Hash; }
};

template <> struct DenseMapInfo<CachedHashStringRef> {
  static CachedHashStringRef getEmptyKey() {
    return CachedHashStringRef(DenseMapInfo<StringRef>::getEmptyKey(), 0);
  }
  static CachedHashStringRef getTombstoneKey() {
    return CachedHashStringRef(DenseMapInfo<StringRef>::getTombstoneKey(), 1);
  }
  static unsigned getHashValue(const CachedHashStringRef &S) {
    assert(!isEqual(S, getEmptyKey()) && "Cannot hash the empty key!");
    assert(!isEqual(S, getTombstoneKey()) && "Cannot hash the tombstone key!");
    return S.hash();
  }
  static bool isEqual(const CachedHashStringRef &LHS,
                      const CachedHashStringRef &RHS) {
    return DenseMapInfo<StringRef>::isEqual(LHS.val(), RHS.val());
  }
};

} // namespace llvm

#endif
