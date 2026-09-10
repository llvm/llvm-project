//===- SLPReductionUtils.cpp - SLP reduction match helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPReductionUtils.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"

#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

static bool matchRdxBop(Instruction *I, Value *&V0, Value *&V1) {
  if (match(I, m_BinOp(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMaxNum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMinNum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMaximum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_FMinimum(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::smin>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umax>(m_Value(V0), m_Value(V1))))
    return true;
  if (match(I, m_Intrinsic<Intrinsic::umin>(m_Value(V0), m_Value(V1))))
    return true;
  return false;
}

Instruction *getNonPhiOperand(Instruction *I, PHINode *Phi) {
  Value *Op0 = nullptr;
  Value *Op1 = nullptr;
  if (!matchRdxBop(I, Op0, Op1))
    return nullptr;
  return dyn_cast<Instruction>(Op0 == Phi ? Op1 : Op0);
}

bool isReductionCandidate(Instruction *I) {
  bool IsSelect = match(I, m_Select(m_Value(), m_Value(), m_Value()));
  Value *B0 = nullptr, *B1 = nullptr;
  bool IsBinop = matchRdxBop(I, B0, B1);
  return IsBinop || IsSelect;
}

std::optional<unsigned> getAggregateSize(Instruction *InsertInst) {
  if (auto *IE = dyn_cast<InsertElementInst>(InsertInst))
    return cast<FixedVectorType>(IE->getType())->getNumElements();

  unsigned AggregateSize = 1;
  auto *IV = cast<InsertValueInst>(InsertInst);
  Type *CurrentType = IV->getType();
  do {
    if (auto *ST = dyn_cast<StructType>(CurrentType)) {
      for (auto *Elt : ST->elements())
        if (Elt != ST->getElementType(0)) // check homogeneity
          return std::nullopt;
      AggregateSize *= ST->getNumElements();
      CurrentType = ST->getElementType(0);
    } else if (auto *AT = dyn_cast<ArrayType>(CurrentType)) {
      AggregateSize *= AT->getNumElements();
      CurrentType = AT->getElementType();
    } else if (auto *VT = dyn_cast<FixedVectorType>(CurrentType)) {
      AggregateSize *= VT->getNumElements();
      return AggregateSize;
    } else if (CurrentType->isSingleValueType()) {
      return AggregateSize;
    } else {
      return std::nullopt;
    }
  } while (true);
}

Instruction *getReductionInstr(const DominatorTree *DT, PHINode *P,
                               BasicBlock *ParentBB, LoopInfo *LI) {
  // There are situations where the reduction value is not dominated by the
  // reduction phi. Vectorizing such cases has been reported to cause
  // miscompiles. See PR25787.
  auto DominatedReduxValue = [&](Value *R) {
    return isa<Instruction>(R) &&
           DT->dominates(P->getParent(), cast<Instruction>(R)->getParent());
  };

  Instruction *Rdx = nullptr;

  // Return the incoming value if it comes from the same BB as the phi node.
  if (P->getIncomingBlock(0) == ParentBB) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(0));
  } else if (P->getIncomingBlock(1) == ParentBB) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(1));
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  // Otherwise, check whether we have a loop latch to look at.
  Loop *BBL = LI->getLoopFor(ParentBB);
  if (!BBL)
    return nullptr;
  BasicBlock *BBLatch = BBL->getLoopLatch();
  if (!BBLatch)
    return nullptr;

  // There is a loop latch, return the incoming value if it comes from
  // that. This reduction pattern occasionally turns up.
  if (P->getIncomingBlock(0) == BBLatch) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(0));
  } else if (P->getIncomingBlock(1) == BBLatch) {
    Rdx = dyn_cast<Instruction>(P->getIncomingValue(1));
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  return nullptr;
}

} // namespace llvm::slpvectorizer
