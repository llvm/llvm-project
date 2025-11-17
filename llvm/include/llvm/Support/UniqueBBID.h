//===- llvm/Support/UniqueBBID.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unique fixed ID assigned to basic blocks upon their creation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_UNIQUEBBID_H
#define LLVM_SUPPORT_UNIQUEBBID_H

#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {

// This structure represents the information for a basic block pertaining to
// the basic block sections profile.
struct UniqueBBID {
  unsigned BaseID;
  unsigned CloneID;
};

// Provides DenseMapInfo for UniqueBBID.
template <> struct DenseMapInfo<UniqueBBID> {
  static inline UniqueBBID getEmptyKey() {
    unsigned EmptyKey = DenseMapInfo<unsigned>::getEmptyKey();
    return UniqueBBID{EmptyKey, EmptyKey};
  }
  static inline UniqueBBID getTombstoneKey() {
    unsigned TombstoneKey = DenseMapInfo<unsigned>::getTombstoneKey();
    return UniqueBBID{TombstoneKey, TombstoneKey};
  }
  static unsigned getHashValue(const UniqueBBID &Val) {
    std::pair<unsigned, unsigned> PairVal =
        std::make_pair(Val.BaseID, Val.CloneID);
    return DenseMapInfo<std::pair<unsigned, unsigned>>::getHashValue(PairVal);
  }
  static bool isEqual(const UniqueBBID &LHS, const UniqueBBID &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.BaseID, RHS.BaseID) &&
           DenseMapInfo<unsigned>::isEqual(LHS.CloneID, RHS.CloneID);
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_UNIQUEBBID_H
