//===- DomConditionCache.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomConditionCache.h"
#include "llvm/Analysis/ValueTracking.h"
using namespace llvm;

static void findAffectedValues(
    Value *Cond,
    SmallVectorImpl<std::pair<Value *, DomConditionFlag>> &Affected) {
  auto InsertAffected = [&Affected](Value *V, DomConditionFlag Flags) {
    Affected.push_back({V, Flags});
  };
  findValuesAffectedByCondition(Cond, /*IsAssume=*/false, InsertAffected);
}

void DomConditionCache::registerBranch(BranchInst *BI) {
  assert(BI->isConditional() && "Must be conditional branch");
  SmallVector<std::pair<Value *, DomConditionFlag>, 16> Affected;
  findAffectedValues(BI->getCondition(), Affected);
  for (auto [V, Flags] : Affected) {
    auto &AV = AffectedValues[V];
    bool Exist = false;
    for (auto &[OtherBI, OtherFlags] : AV) {
      if (OtherBI == BI) {
        OtherFlags |= Flags;
        Exist = true;
        break;
      }
    }
    if (!Exist)
      AV.push_back({BI, Flags});
  }
}
