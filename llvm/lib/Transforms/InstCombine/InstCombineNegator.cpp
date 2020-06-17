//===- InstCombineNegator.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sinking of negation into expression trees,
// as long as that can be done without increasing instruction count.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <tuple>
#include <utility>

namespace llvm {
class AssumptionCache;
class DataLayout;
class DominatorTree;
class LLVMContext;
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "instcombine"

STATISTIC(NegatorTotalNegationsAttempted,
          "Negator: Number of negations attempted to be sinked");
STATISTIC(NegatorNumTreesNegated,
          "Negator: Number of negations successfully sinked");
STATISTIC(NegatorMaxDepthVisited, "Negator: Maximal traversal depth ever "
                                  "reached while attempting to sink negation");
STATISTIC(NegatorTimesDepthLimitReached,
          "Negator: How many times did the traversal depth limit was reached "
          "during sinking");
STATISTIC(
    NegatorNumValuesVisited,
    "Negator: Total number of values visited during attempts to sink negation");
STATISTIC(NegatorMaxTotalValuesVisited,
          "Negator: Maximal number of values ever visited while attempting to "
          "sink negation");
STATISTIC(NegatorNumInstructionsCreatedTotal,
          "Negator: Number of new negated instructions created, total");
STATISTIC(NegatorMaxInstructionsCreated,
          "Negator: Maximal number of new instructions created during negation "
          "attempt");
STATISTIC(NegatorNumInstructionsNegatedSuccess,
          "Negator: Number of new negated instructions created in successful "
          "negation sinking attempts");

DEBUG_COUNTER(NegatorCounter, "instcombine-negator",
              "Controls Negator transformations in InstCombine pass");

static cl::opt<bool>
    NegatorEnabled("instcombine-negator-enabled", cl::init(true),
                   cl::desc("Should we attempt to sink negations?"));

static cl::opt<unsigned>
    NegatorMaxDepth("instcombine-negator-max-depth",
                    cl::init(NegatorDefaultMaxDepth),
                    cl::desc("What is the maximal lookup depth when trying to "
                             "check for viability of negation sinking."));

Negator::Negator(LLVMContext &C, const DataLayout &DL_, AssumptionCache &AC_,
                 const DominatorTree &DT_, bool IsTrulyNegation_)
    : Builder(C, TargetFolder(DL_),
              IRBuilderCallbackInserter([&](Instruction *I) {
                ++NegatorNumInstructionsCreatedTotal;
                NewInstructions.push_back(I);
              })),
      DL(DL_), AC(AC_), DT(DT_), IsTrulyNegation(IsTrulyNegation_) {}

#if LLVM_ENABLE_STATS
Negator::~Negator() {
  NegatorMaxTotalValuesVisited.updateMax(NumValuesVisitedInThisNegator);
}
#endif

// FIXME: can this be reworked into a worklist-based algorithm while preserving
// the depth-first, early bailout traversal?
LLVM_NODISCARD Value *Negator::visit(Value *V, unsigned Depth) {
  NegatorMaxDepthVisited.updateMax(Depth);
  ++NegatorNumValuesVisited;

#if LLVM_ENABLE_STATS
  ++NumValuesVisitedInThisNegator;
#endif

  // -(undef) -> undef.
  if (match(V, m_Undef()))
    return V;

  // In i1, negation can simply be ignored.
  if (V->getType()->isIntOrIntVectorTy(1))
    return V;

  Value *X;

  // -(-(X)) -> X.
  if (match(V, m_Neg(m_Value(X))))
    return X;

  // Integral constants can be freely negated.
  if (match(V, m_AnyIntegralConstant()))
    return ConstantExpr::getNeg(cast<Constant>(V), /*HasNUW=*/false,
                                /*HasNSW=*/false);

  // If we have a non-instruction, then give up.
  if (!isa<Instruction>(V))
    return nullptr;

  // If we have started with a true negation (i.e. `sub 0, %y`), then if we've
  // got instruction that does not require recursive reasoning, we can still
  // negate it even if it has other uses, without increasing instruction count.
  if (!V->hasOneUse() && !IsTrulyNegation)
    return nullptr;

  auto *I = cast<Instruction>(V);
  unsigned BitWidth = I->getType()->getScalarSizeInBits();

  // We must preserve the insertion point and debug info that is set in the
  // builder at the time this function is called.
  InstCombiner::BuilderTy::InsertPointGuard Guard(Builder);
  // And since we are trying to negate instruction I, that tells us about the
  // insertion point and the debug info that we need to keep.
  Builder.SetInsertPoint(I);

  // In some cases we can give the answer without further recursion.
  switch (I->getOpcode()) {
  case Instruction::Add:
    // `inc` is always negatible.
    if (match(I->getOperand(1), m_One()))
      return Builder.CreateNot(I->getOperand(0), I->getName() + ".neg");
    break;
  case Instruction::Xor:
    // `not` is always negatible.
    if (match(I, m_Not(m_Value(X))))
      return Builder.CreateAdd(X, ConstantInt::get(X->getType(), 1),
                               I->getName() + ".neg");
    break;
  case Instruction::AShr:
  case Instruction::LShr: {
    // Right-shift sign bit smear is negatible.
    const APInt *Op1Val;
    if (match(I->getOperand(1), m_APInt(Op1Val)) && *Op1Val == BitWidth - 1) {
      Value *BO = I->getOpcode() == Instruction::AShr
                      ? Builder.CreateLShr(I->getOperand(0), I->getOperand(1))
                      : Builder.CreateAShr(I->getOperand(0), I->getOperand(1));
      if (auto *NewInstr = dyn_cast<Instruction>(BO)) {
        NewInstr->copyIRFlags(I);
        NewInstr->setName(I->getName() + ".neg");
      }
      return BO;
    }
    break;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
    // `*ext` of i1 is always negatible
    if (I->getOperand(0)->getType()->isIntOrIntVectorTy(1))
      return I->getOpcode() == Instruction::SExt
                 ? Builder.CreateZExt(I->getOperand(0), I->getType(),
                                      I->getName() + ".neg")
                 : Builder.CreateSExt(I->getOperand(0), I->getType(),
                                      I->getName() + ".neg");
    break;
  default:
    break; // Other instructions require recursive reasoning.
  }

  // Some other cases, while still don't require recursion,
  // are restricted to the one-use case.
  if (!V->hasOneUse())
    return nullptr;

  switch (I->getOpcode()) {
  case Instruction::Sub:
    // `sub` is always negatible.
    // But if the old `sub` sticks around, even thought we don't increase
    // instruction count, this is a likely regression since we increased
    // live-range of *both* of the operands, which might lead to more spilling.
    return Builder.CreateSub(I->getOperand(1), I->getOperand(0),
                             I->getName() + ".neg");
  case Instruction::SDiv:
    // `sdiv` is negatible if divisor is not undef/INT_MIN/1.
    // While this is normally not behind a use-check,
    // let's consider division to be special since it's costly.
    if (auto *Op1C = dyn_cast<Constant>(I->getOperand(1))) {
      if (!Op1C->containsUndefElement() && Op1C->isNotMinSignedValue() &&
          Op1C->isNotOneValue()) {
        Value *BO =
            Builder.CreateSDiv(I->getOperand(0), ConstantExpr::getNeg(Op1C),
                               I->getName() + ".neg");
        if (auto *NewInstr = dyn_cast<Instruction>(BO))
          NewInstr->setIsExact(I->isExact());
        return BO;
      }
    }
    break;
  }

  // Rest of the logic is recursive, so if it's time to give up then it's time.
  if (Depth > NegatorMaxDepth) {
    LLVM_DEBUG(dbgs() << "Negator: reached maximal allowed traversal depth in "
                      << *V << ". Giving up.\n");
    ++NegatorTimesDepthLimitReached;
    return nullptr;
  }

  switch (I->getOpcode()) {
  case Instruction::PHI: {
    // `phi` is negatible if all the incoming values are negatible.
    auto *PHI = cast<PHINode>(I);
    SmallVector<Value *, 4> NegatedIncomingValues(PHI->getNumOperands());
    for (auto I : zip(PHI->incoming_values(), NegatedIncomingValues)) {
      if (!(std::get<1>(I) = visit(std::get<0>(I), Depth + 1))) // Early return.
        return nullptr;
    }
    // All incoming values are indeed negatible. Create negated PHI node.
    PHINode *NegatedPHI = Builder.CreatePHI(
        PHI->getType(), PHI->getNumOperands(), PHI->getName() + ".neg");
    for (auto I : zip(NegatedIncomingValues, PHI->blocks()))
      NegatedPHI->addIncoming(std::get<0>(I), std::get<1>(I));
    return NegatedPHI;
  }
  case Instruction::Select: {
    {
      // `abs`/`nabs` is always negatible.
      Value *LHS, *RHS;
      SelectPatternFlavor SPF =
          matchSelectPattern(I, LHS, RHS, /*CastOp=*/nullptr, Depth).Flavor;
      if (SPF == SPF_ABS || SPF == SPF_NABS) {
        auto *NewSelect = cast<SelectInst>(I->clone());
        // Just swap the operands of the select.
        NewSelect->swapValues();
        // Don't swap prof metadata, we didn't change the branch behavior.
        NewSelect->setName(I->getName() + ".neg");
        Builder.Insert(NewSelect);
        return NewSelect;
      }
    }
    // `select` is negatible if both hands of `select` are negatible.
    Value *NegOp1 = visit(I->getOperand(1), Depth + 1);
    if (!NegOp1) // Early return.
      return nullptr;
    Value *NegOp2 = visit(I->getOperand(2), Depth + 1);
    if (!NegOp2)
      return nullptr;
    // Do preserve the metadata!
    return Builder.CreateSelect(I->getOperand(0), NegOp1, NegOp2,
                                I->getName() + ".neg", /*MDFrom=*/I);
  }
  case Instruction::ShuffleVector: {
    // `shufflevector` is negatible if both operands are negatible.
    auto *Shuf = cast<ShuffleVectorInst>(I);
    Value *NegOp0 = visit(I->getOperand(0), Depth + 1);
    if (!NegOp0) // Early return.
      return nullptr;
    Value *NegOp1 = visit(I->getOperand(1), Depth + 1);
    if (!NegOp1)
      return nullptr;
    return Builder.CreateShuffleVector(NegOp0, NegOp1, Shuf->getShuffleMask(),
                                       I->getName() + ".neg");
  }
  case Instruction::ExtractElement: {
    // `extractelement` is negatible if source operand is negatible.
    auto *EEI = cast<ExtractElementInst>(I);
    Value *NegVector = visit(EEI->getVectorOperand(), Depth + 1);
    if (!NegVector) // Early return.
      return nullptr;
    return Builder.CreateExtractElement(NegVector, EEI->getIndexOperand(),
                                        I->getName() + ".neg");
  }
  case Instruction::InsertElement: {
    // `insertelement` is negatible if both the source vector and
    // element-to-be-inserted are negatible.
    auto *IEI = cast<InsertElementInst>(I);
    Value *NegVector = visit(IEI->getOperand(0), Depth + 1);
    if (!NegVector) // Early return.
      return nullptr;
    Value *NegNewElt = visit(IEI->getOperand(1), Depth + 1);
    if (!NegNewElt) // Early return.
      return nullptr;
    return Builder.CreateInsertElement(NegVector, NegNewElt, IEI->getOperand(2),
                                       I->getName() + ".neg");
  }
  case Instruction::Trunc: {
    // `trunc` is negatible if its operand is negatible.
    Value *NegOp = visit(I->getOperand(0), Depth + 1);
    if (!NegOp) // Early return.
      return nullptr;
    return Builder.CreateTrunc(NegOp, I->getType(), I->getName() + ".neg");
  }
  case Instruction::Shl: {
    // `shl` is negatible if the first operand is negatible.
    Value *NegOp0 = visit(I->getOperand(0), Depth + 1);
    if (!NegOp0) // Early return.
      return nullptr;
    return Builder.CreateShl(NegOp0, I->getOperand(1), I->getName() + ".neg");
  }
  case Instruction::Or:
    if (!haveNoCommonBitsSet(I->getOperand(0), I->getOperand(1), DL, &AC, I,
                             &DT))
      return nullptr; // Don't know how to handle `or` in general.
    // `or`/`add` are interchangeable when operands have no common bits set.
    // `inc` is always negatible.
    if (match(I->getOperand(1), m_One()))
      return Builder.CreateNot(I->getOperand(0), I->getName() + ".neg");
    // Else, just defer to Instruction::Add handling.
    LLVM_FALLTHROUGH;
  case Instruction::Add: {
    // `add` is negatible if both of its operands are negatible.
    Value *NegOp0 = visit(I->getOperand(0), Depth + 1);
    if (!NegOp0) // Early return.
      return nullptr;
    Value *NegOp1 = visit(I->getOperand(1), Depth + 1);
    if (!NegOp1)
      return nullptr;
    return Builder.CreateAdd(NegOp0, NegOp1, I->getName() + ".neg");
  }
  case Instruction::Xor:
    // `xor` is negatible if one of its operands is invertible.
    // FIXME: InstCombineInverter? But how to connect Inverter and Negator?
    if (auto *C = dyn_cast<Constant>(I->getOperand(1))) {
      Value *Xor = Builder.CreateXor(I->getOperand(0), ConstantExpr::getNot(C));
      return Builder.CreateAdd(Xor, ConstantInt::get(Xor->getType(), 1),
                               I->getName() + ".neg");
    }
    return nullptr;
  case Instruction::Mul: {
    // `mul` is negatible if one of its operands is negatible.
    Value *NegatedOp, *OtherOp;
    // First try the second operand, in case it's a constant it will be best to
    // just invert it instead of sinking the `neg` deeper.
    if (Value *NegOp1 = visit(I->getOperand(1), Depth + 1)) {
      NegatedOp = NegOp1;
      OtherOp = I->getOperand(0);
    } else if (Value *NegOp0 = visit(I->getOperand(0), Depth + 1)) {
      NegatedOp = NegOp0;
      OtherOp = I->getOperand(1);
    } else
      // Can't negate either of them.
      return nullptr;
    return Builder.CreateMul(NegatedOp, OtherOp, I->getName() + ".neg");
  }
  default:
    return nullptr; // Don't know, likely not negatible for free.
  }

  llvm_unreachable("Can't get here. We always return from switch.");
}

LLVM_NODISCARD Optional<Negator::Result> Negator::run(Value *Root) {
  Value *Negated = visit(Root, /*Depth=*/0);
  if (!Negated) {
    // We must cleanup newly-inserted instructions, to avoid any potential
    // endless combine looping.
    llvm::for_each(llvm::reverse(NewInstructions),
                   [&](Instruction *I) { I->eraseFromParent(); });
    return llvm::None;
  }
  return std::make_pair(ArrayRef<Instruction *>(NewInstructions), Negated);
}

LLVM_NODISCARD Value *Negator::Negate(bool LHSIsZero, Value *Root,
                                      InstCombiner &IC) {
  ++NegatorTotalNegationsAttempted;
  LLVM_DEBUG(dbgs() << "Negator: attempting to sink negation into " << *Root
                    << "\n");

  if (!NegatorEnabled || !DebugCounter::shouldExecute(NegatorCounter))
    return nullptr;

  Negator N(Root->getContext(), IC.getDataLayout(), IC.getAssumptionCache(),
            IC.getDominatorTree(), LHSIsZero);
  Optional<Result> Res = N.run(Root);
  if (!Res) { // Negation failed.
    LLVM_DEBUG(dbgs() << "Negator: failed to sink negation into " << *Root
                      << "\n");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "Negator: successfully sunk negation into " << *Root
                    << "\n         NEW: " << *Res->second << "\n");
  ++NegatorNumTreesNegated;

  // We must temporarily unset the 'current' insertion point and DebugLoc of the
  // InstCombine's IRBuilder so that it won't interfere with the ones we have
  // already specified when producing negated instructions.
  InstCombiner::BuilderTy::InsertPointGuard Guard(IC.Builder);
  IC.Builder.ClearInsertionPoint();
  IC.Builder.SetCurrentDebugLocation(DebugLoc());

  // And finally, we must add newly-created instructions into the InstCombine's
  // worklist (in a proper order!) so it can attempt to combine them.
  LLVM_DEBUG(dbgs() << "Negator: Propagating " << Res->first.size()
                    << " instrs to InstCombine\n");
  NegatorMaxInstructionsCreated.updateMax(Res->first.size());
  NegatorNumInstructionsNegatedSuccess += Res->first.size();

  // They are in def-use order, so nothing fancy, just insert them in order.
  llvm::for_each(Res->first,
                 [&](Instruction *I) { IC.Builder.Insert(I, I->getName()); });

  // And return the new root.
  return Res->second;
}
