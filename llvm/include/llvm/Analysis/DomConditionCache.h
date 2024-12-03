//===- llvm/Analysis/DomConditionCache.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cache for branch conditions that affect a certain value for use by
// ValueTracking. Unlike AssumptionCache, this class does not perform any
// automatic analysis or invalidation. The caller is responsible for registering
// all relevant branches (and re-registering them if they change), and for
// removing invalidated values from the cache.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMCONDITIONCACHE_H
#define LLVM_ANALYSIS_DOMCONDITIONCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace llvm {

class Value;
class BranchInst;

enum class DomConditionFlag : uint8_t {
  None = 0,
  KnownBits = 1 << 0,
  KnownFPClass = 1 << 1,
  PowerOfTwo = 1 << 2,
  ICmp = 1 << 3,
};

LLVM_DECLARE_ENUM_AS_BITMASK(
    DomConditionFlag,
    /*LargestValue=*/static_cast<uint8_t>(DomConditionFlag::ICmp));

class DomConditionCache {
private:
  /// A map of values about which a branch might be providing information.
  using AffectedValuesMap =
      DenseMap<Value *,
               SmallVector<std::pair<BranchInst *, DomConditionFlag>, 1>>;
  AffectedValuesMap AffectedValues;

public:
  /// Add a branch condition to the cache.
  void registerBranch(BranchInst *BI);

  /// Remove a value from the cache, e.g. because it will be erased.
  void removeValue(Value *V) { AffectedValues.erase(V); }

  /// Access the list of branches which affect this value.
  ArrayRef<std::pair<BranchInst *, DomConditionFlag>>
  conditionsFor(const Value *V) const {
    auto AVI = AffectedValues.find_as(const_cast<Value *>(V));
    if (AVI == AffectedValues.end())
      return {};

    return AVI->second;
  }
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_DOMCONDITIONCACHE_H
