//===- SLPReductionUtils.cpp - SLP reduction match helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPReductionUtils.h"

#include "SLPCostAnalysis.h"

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"

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

Type *getBoolReduxWideTy(RecurKind RdxKind, Type *RootTy, Type *LeafTy) {
  if ((RdxKind == RecurKind::And || RdxKind == RecurKind::Or) &&
      RootTy->isIntegerTy(1) && LeafTy->isIntegerTy() &&
      !LeafTy->isIntegerTy(1))
    return LeafTy;
  return nullptr;
}

Value *tryEmitBoolReduxBitcastCmp(IRBuilderBase &Builder,
                                  const TargetTransformInfo &TTI,
                                  RecurKind RdxKind, Value *Vec,
                                  const Value *Root, FastMathFlags FMF,
                                  const TTI::TargetCostKind CostKind) {
  auto *VecTy = cast<FixedVectorType>(Vec->getType());
  unsigned VF = VecTy->getNumElements();
  auto *I1VecTy = FixedVectorType::get(Builder.getInt1Ty(), VF);
  DebugLoc DL = Builder.getCurrentDebugLocation();
  Builder.SetCurrentDebugLocation(cast<Instruction>(Root)->getDebugLoc());
  Value *T = Builder.CreateTrunc(Vec, I1VecTy);
  Value *BC = Builder.CreateBitCast(T, Builder.getIntNTy(VF));
  CmpInst::Predicate Pred =
      RdxKind == RecurKind::And ? CmpInst::ICMP_EQ : CmpInst::ICMP_NE;
  Constant *RHS = RdxKind == RecurKind::And
                      ? Constant::getAllOnesValue(BC->getType())
                      : Constant::getNullValue(BC->getType());
  Value *Res = Builder.CreateICmp(Pred, BC, RHS);
  // The costs are evaluated from the emitted instructions; they are dropped
  // if the wide reduction form is cheaper.
  auto CastCost = [&](Value *V, unsigned Opcode, Type *SrcTy) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return InstructionCost(0);
    return TTI.getCastInstrCost(Opcode, I->getType(), SrcTy,
                                TTI.getCastContextHint(I), CostKind, I);
  };
  InstructionCost BitcastCmpCost = CastCost(T, Instruction::Trunc, VecTy) +
                                   CastCost(BC, Instruction::BitCast, I1VecTy);
  if (auto *Cmp = dyn_cast<Instruction>(Res))
    BitcastCmpCost += TTI.getCmpSelInstrCost(
        Instruction::ICmp, BC->getType(), /*CondTy=*/nullptr, Pred, CostKind,
        TTI.getOperandInfo(BC), TTI.getOperandInfo(RHS), Cmp);
  if (BitcastCmpCost >=
      getBoolReduxWideRdxCost(TTI, RdxKind, VecTy, Root, FMF, CostKind)) {
    for (Value *V : {Res, BC, T})
      if (auto *I = dyn_cast<Instruction>(V))
        I->eraseFromParent();
    Builder.SetCurrentDebugLocation(DL);
    return nullptr;
  }
  Builder.SetCurrentDebugLocation(DL);
  return Res;
}

} // namespace llvm::slpvectorizer
