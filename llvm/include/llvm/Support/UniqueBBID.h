//===- llvm/Support/UniqueBBID.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a structure that uniquely identifies a basic block within
// a function.
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

// The prefetch symbol is emitted immediately after the call of the given index,
// in block `BBID` (First call has an index of 1). Zero callsite index means the
// start of the block.
struct CallsiteID {
  UniqueBBID BBID;
  unsigned CallsiteIndex;
};

// This represents a prefetch hint to be injected at site `SiteID`, targetting
// `TargetID` in function `TargetFunction`.
struct PrefetchHint {
  CallsiteID SiteID;
  StringRef TargetFunction;
  CallsiteID TargetID;
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
    return DenseMapInfo<unsigned>::getHashValue(Val.BaseID) ^
           DenseMapInfo<unsigned>::getHashValue(Val.CloneID);
  }

  static bool isEqual(const UniqueBBID &LHS, const UniqueBBID &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.BaseID, RHS.BaseID) &&
           DenseMapInfo<unsigned>::isEqual(LHS.CloneID, RHS.CloneID);
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_UNIQUEBBID_H
