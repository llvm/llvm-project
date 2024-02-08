//===- DomConditionCache.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomConditionCache.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;
using namespace llvm::PatternMatch;

// TODO: This code is very similar to findAffectedValues() in
// AssumptionCache, but currently specialized to just the patterns that
// computeKnownBits() supports, and without the notion of result elem indices
// that are AC specific. Deduplicate this code once we have a clearer picture
// of how much they can be shared.
static void findAffectedValues(Value *Cond,
                               SmallVectorImpl<Value *> &Affected) {
  auto AddAffected = [&Affected](Value *V) {
    if (isa<Argument>(V) || isa<GlobalValue>(V)) {
      Affected.push_back(V);
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      Affected.push_back(I);

      // Peek through unary operators to find the source of the condition.
      Value *Op;
      if (match(I, m_PtrToInt(m_Value(Op)))) {
        if (isa<Instruction>(Op) || isa<Argument>(Op))
          Affected.push_back(Op);
      }
    }
  };

  bool TopLevelIsAnd = match(Cond, m_LogicalAnd());
  SmallVector<Value *, 8> Worklist;
  SmallPtrSet<Value *, 8> Visited;
  Worklist.push_back(Cond);
  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    ICmpInst::Predicate Pred;
    Value *A, *B;
    // Only recurse into and/or if it matches the top-level and/or type.
    if (TopLevelIsAnd ? match(V, m_LogicalAnd(m_Value(A), m_Value(B)))
                      : match(V, m_LogicalOr(m_Value(A), m_Value(B)))) {
      Worklist.push_back(A);
      Worklist.push_back(B);
    } else if (match(V, m_ICmp(Pred, m_Value(A), m_Constant()))) {
      AddAffected(A);

      if (ICmpInst::isEquality(Pred)) {
        Value *X;
        // (X & C) or (X | C) or (X ^ C).
        // (X << C) or (X >>_s C) or (X >>_u C).
        if (match(A, m_BitwiseLogic(m_Value(X), m_ConstantInt())) ||
            match(A, m_Shift(m_Value(X), m_ConstantInt())))
          AddAffected(X);
      } else {
        Value *X;
        // Handle (A + C1) u< C2, which is the canonical form of
        // A > C3 && A < C4.
        if (match(A, m_Add(m_Value(X), m_ConstantInt())))
          AddAffected(X);
      }
    }
  }
}

void DomConditionCache::registerBranch(BranchInst *BI) {
  assert(BI->isConditional() && "Must be conditional branch");
  SmallVector<Value *, 16> Affected;
  findAffectedValues(BI->getCondition(), Affected);
  for (Value *V : Affected) {
    auto &AV = AffectedValues[V];
    if (!is_contained(AV, BI))
      AV.push_back(BI);
  }
}
