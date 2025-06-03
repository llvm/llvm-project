//===- LoopUnrollAnalyzer.cpp - Unrolling Effect Estimation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements UnrolledInstAnalyzer class. It's used for predicting
// potential effects that loop unrolling might have, such as enabling constant
// propagation and other optimizations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

/// Try to simplify instruction \param I using its SCEV expression.
///
/// The idea is that some AddRec expressions become constants, which then
/// could trigger folding of other instructions. However, that only happens
/// for expressions whose start value is also constant, which isn't always the
/// case. In another common and important case the start value is just some
/// address (i.e. SCEVUnknown) - in this case we compute the offset and save
/// it along with the base address instead.
bool UnrolledInstAnalyzer::simplifyInstWithSCEV(Instruction *I) {
  if (!SE.isSCEVable(I->getType()))
    return false;

  const SCEV *S = SE.getSCEV(I);
  if (auto *SC = dyn_cast<SCEVConstant>(S)) {
    SimplifiedValues[I] = SC->getValue();
    return true;
  }

  // If we have a loop invariant computation, we only need to compute it once.
  // Given that, all but the first occurance are free.
  if (!IterationNumber->isZero() && SE.isLoopInvariant(S, L))
    return true;

  auto *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR || AR->getLoop() != L)
    return false;

  const SCEV *ValueAtIteration = AR->evaluateAtIteration(IterationNumber, SE);
  // Check if the AddRec expression becomes a constant.
  if (auto *SC = dyn_cast<SCEVConstant>(ValueAtIteration)) {
    SimplifiedValues[I] = SC->getValue();
    return true;
  }

  // Check if the offset from the base address becomes a constant.
  auto *Base = dyn_cast<SCEVUnknown>(SE.getPointerBase(S));
  if (!Base)
    return false;
  std::optional<APInt> Offset =
      SE.computeConstantDifference(ValueAtIteration, Base);
  if (!Offset)
    return false;
  SimplifiedAddress Address;
  Address.Base = Base->getValue();
  Address.Offset = *Offset;
  SimplifiedAddresses[I] = Address;
  return false;
}

/// Try to simplify binary operator I.
///
/// TODO: Probably it's worth to hoist the code for estimating the
/// simplifications effects to a separate class, since we have a very similar
/// code in InlineCost already.
bool UnrolledInstAnalyzer::visitBinaryOperator(BinaryOperator &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  if (!isa<Constant>(LHS))
    if (Value *SimpleLHS = SimplifiedValues.lookup(LHS))
      LHS = SimpleLHS;
  if (!isa<Constant>(RHS))
    if (Value *SimpleRHS = SimplifiedValues.lookup(RHS))
      RHS = SimpleRHS;

  Value *SimpleV = nullptr;
  const DataLayout &DL = I.getDataLayout();
  if (auto FI = dyn_cast<FPMathOperator>(&I))
    SimpleV =
        simplifyBinOp(I.getOpcode(), LHS, RHS, FI->getFastMathFlags(), DL);
  else
    SimpleV = simplifyBinOp(I.getOpcode(), LHS, RHS, DL);

  if (SimpleV) {
    SimplifiedValues[&I] = SimpleV;
    return true;
  }
  return Base::visitBinaryOperator(I);
}

/// Try to fold load I.
bool UnrolledInstAnalyzer::visitLoad(LoadInst &I) {
  Value *AddrOp = I.getPointerOperand();

  auto AddressIt = SimplifiedAddresses.find(AddrOp);
  if (AddressIt == SimplifiedAddresses.end())
    return false;

  auto *GV = dyn_cast<GlobalVariable>(AddressIt->second.Base);
  // We're only interested in loads that can be completely folded to a
  // constant.
  if (!GV || !GV->hasDefinitiveInitializer() || !GV->isConstant())
    return false;

  Constant *Res =
      ConstantFoldLoadFromConst(GV->getInitializer(), I.getType(),
                                AddressIt->second.Offset, I.getDataLayout());
  if (!Res)
    return false;

  SimplifiedValues[&I] = Res;
  return true;
}

/// Try to simplify cast instruction.
bool UnrolledInstAnalyzer::visitCastInst(CastInst &I) {
  Value *Op = I.getOperand(0);
  if (Value *Simplified = SimplifiedValues.lookup(Op))
    Op = Simplified;

  // The cast can be invalid, because SimplifiedValues contains results of SCEV
  // analysis, which operates on integers (and, e.g., might convert i8* null to
  // i32 0).
  if (CastInst::castIsValid(I.getOpcode(), Op, I.getType())) {
    const DataLayout &DL = I.getDataLayout();
    if (Value *V = simplifyCastInst(I.getOpcode(), Op, I.getType(), DL)) {
      SimplifiedValues[&I] = V;
      return true;
    }
  }

  return Base::visitCastInst(I);
}

/// Try to simplify cmp instruction.
bool UnrolledInstAnalyzer::visitCmpInst(CmpInst &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  // First try to handle simplified comparisons.
  if (!isa<Constant>(LHS))
    if (Value *SimpleLHS = SimplifiedValues.lookup(LHS))
      LHS = SimpleLHS;
  if (!isa<Constant>(RHS))
    if (Value *SimpleRHS = SimplifiedValues.lookup(RHS))
      RHS = SimpleRHS;

  if (!isa<Constant>(LHS) && !isa<Constant>(RHS) && !I.isSigned()) {
    auto SimplifiedLHS = SimplifiedAddresses.find(LHS);
    if (SimplifiedLHS != SimplifiedAddresses.end()) {
      auto SimplifiedRHS = SimplifiedAddresses.find(RHS);
      if (SimplifiedRHS != SimplifiedAddresses.end()) {
        SimplifiedAddress &LHSAddr = SimplifiedLHS->second;
        SimplifiedAddress &RHSAddr = SimplifiedRHS->second;
        if (LHSAddr.Base == RHSAddr.Base) {
          // FIXME: This is only correct for equality predicates. For
          // unsigned predicates, this only holds if we have nowrap flags,
          // which we don't track (for nuw it's valid as-is, for nusw it
          // requires converting the predicated to signed). As this is used only
          // for cost modelling, this is not a correctness issue.
          bool Res = ICmpInst::compare(LHSAddr.Offset, RHSAddr.Offset,
                                       I.getPredicate());
          SimplifiedValues[&I] = ConstantInt::getBool(I.getType(), Res);
          return true;
        }
      }
    }
  }

  const DataLayout &DL = I.getDataLayout();
  if (Value *V = simplifyCmpInst(I.getPredicate(), LHS, RHS, DL)) {
    SimplifiedValues[&I] = V;
    return true;
  }

  return Base::visitCmpInst(I);
}

bool UnrolledInstAnalyzer::visitPHINode(PHINode &PN) {
  // Run base visitor first. This way we can gather some useful for later
  // analysis information.
  if (Base::visitPHINode(PN))
    return true;

  // The loop induction PHI nodes are definitionally free.
  return PN.getParent() == L->getHeader();
}

bool UnrolledInstAnalyzer::visitInstruction(Instruction &I) {
  return simplifyInstWithSCEV(&I);
}
