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
    uint32_t Underlying = to_underlying(Flags);
    while (Underlying) {
      uint32_t LSB = Underlying & -Underlying;
      auto &AV = AffectedValues[countr_zero(LSB)][V];
      if (llvm::none_of(AV, [&](BranchInst *Elem) { return Elem == BI; }))
        AV.push_back(BI);
      Underlying -= LSB;
    }
  }
}
