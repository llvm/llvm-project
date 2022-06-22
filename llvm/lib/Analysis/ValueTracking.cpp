//===- ValueTracking.cpp - Walk computations to compute properties --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// computations have.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

// Controls the number of uses of the value searched for possible
// dominating comparisons.
static cl::opt<unsigned> DomConditionsMaxUses("dom-conditions-max-uses",
                                              cl::Hidden, cl::init(20));

// According to the LangRef, branching on a poison condition is absolutely
// immediate full UB.  However, historically we haven't implemented that
// consistently as we had an important transformation (non-trivial unswitch)
// which introduced instances of branch on poison/undef to otherwise well
// defined programs.  This issue has since been fixed, but the flag is
// temporarily retained to easily diagnose potential regressions.
static cl::opt<bool> BranchOnPoisonAsUB("branch-on-poison-as-ub",
                                        cl::Hidden, cl::init(true));


/// Returns the bitwidth of the given scalar or pointer type. For vector types,
/// returns the element type's bitwidth.
static unsigned getBitWidth(Type *Ty, const DataLayout &DL) {
  if (unsigned BitWidth = Ty->getScalarSizeInBits())
    return BitWidth;

  return DL.getPointerTypeSizeInBits(Ty);
}

namespace {

// Simplifying using an assume can only be done in a particular control-flow
// context (the context instruction provides that context). If an assume and
// the context instruction are not in the same block then the DT helps in
// figuring out if we can use it.
struct Query {
  const DataLayout &DL;
  AssumptionCache *AC;
  const Instruction *CxtI;
  const DominatorTree *DT;

  // Unlike the other analyses, this may be a nullptr because not all clients
  // provide it currently.
  OptimizationRemarkEmitter *ORE;

  /// If true, it is safe to use metadata during simplification.
  InstrInfoQuery IIQ;

  Query(const DataLayout &DL, AssumptionCache *AC, const Instruction *CxtI,
        const DominatorTree *DT, bool UseInstrInfo,
        OptimizationRemarkEmitter *ORE = nullptr)
      : DL(DL), AC(AC), CxtI(CxtI), DT(DT), ORE(ORE), IIQ(UseInstrInfo) {}
};

} // end anonymous namespace

// Given the provided Value and, potentially, a context instruction, return
// the preferred context instruction (if any).
static const Instruction *safeCxtI(const Value *V, const Instruction *CxtI) {
  // If we've been provided with a context instruction, then use that (provided
  // it has been inserted).
  if (CxtI && CxtI->getParent())
    return CxtI;

  // If the value is really an already-inserted instruction, then use that.
  CxtI = dyn_cast<Instruction>(V);
  if (CxtI && CxtI->getParent())
    return CxtI;

  return nullptr;
}

static const Instruction *safeCxtI(const Value *V1, const Value *V2, const Instruction *CxtI) {
  // If we've been provided with a context instruction, then use that (provided
  // it has been inserted).
  if (CxtI && CxtI->getParent())
    return CxtI;

  // If the value is really an already-inserted instruction, then use that.
  CxtI = dyn_cast<Instruction>(V1);
  if (CxtI && CxtI->getParent())
    return CxtI;

  CxtI = dyn_cast<Instruction>(V2);
  if (CxtI && CxtI->getParent())
    return CxtI;

  return nullptr;
}

static bool getShuffleDemandedElts(const ShuffleVectorInst *Shuf,
                                   const APInt &DemandedElts,
                                   APInt &DemandedLHS, APInt &DemandedRHS) {
  // The length of scalable vectors is unknown at compile time, thus we
  // cannot check their values
  if (isa<ScalableVectorType>(Shuf->getType()))
    return false;

  int NumElts =
      cast<FixedVectorType>(Shuf->getOperand(0)->getType())->getNumElements();
  int NumMaskElts = cast<FixedVectorType>(Shuf->getType())->getNumElements();
  DemandedLHS = DemandedRHS = APInt::getZero(NumElts);
  if (DemandedElts.isZero())
    return true;
  // Simple case of a shuffle with zeroinitializer.
  if (all_of(Shuf->getShuffleMask(), [](int Elt) { return Elt == 0; })) {
    DemandedLHS.setBit(0);
    return true;
  }
  for (int i = 0; i != NumMaskElts; ++i) {
    if (!DemandedElts[i])
      continue;
    int M = Shuf->getMaskValue(i);
    assert(M < (NumElts * 2) && "Invalid shuffle mask constant");

    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    if (M == -1)
      return false;
    if (M < NumElts)
      DemandedLHS.setBit(M % NumElts);
    else
      DemandedRHS.setBit(M % NumElts);
  }

  return true;
}

static void computeKnownBits(const Value *V, const APInt &DemandedElts,
                             KnownBits &Known, unsigned Depth, const Query &Q);

static void computeKnownBits(const Value *V, KnownBits &Known, unsigned Depth,
                             const Query &Q) {
  // FIXME: We currently have no way to represent the DemandedElts of a scalable
  // vector
  if (isa<ScalableVectorType>(V->getType())) {
    Known.resetAll();
    return;
  }

  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  computeKnownBits(V, DemandedElts, Known, Depth, Q);
}

void llvm::computeKnownBits(const Value *V, KnownBits &Known,
                            const DataLayout &DL, unsigned Depth,
                            AssumptionCache *AC, const Instruction *CxtI,
                            const DominatorTree *DT,
                            OptimizationRemarkEmitter *ORE, bool UseInstrInfo) {
  ::computeKnownBits(V, Known, Depth,
                     Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo, ORE));
}

void llvm::computeKnownBits(const Value *V, const APInt &DemandedElts,
                            KnownBits &Known, const DataLayout &DL,
                            unsigned Depth, AssumptionCache *AC,
                            const Instruction *CxtI, const DominatorTree *DT,
                            OptimizationRemarkEmitter *ORE, bool UseInstrInfo) {
  ::computeKnownBits(V, DemandedElts, Known, Depth,
                     Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo, ORE));
}

static KnownBits computeKnownBits(const Value *V, const APInt &DemandedElts,
                                  unsigned Depth, const Query &Q);

static KnownBits computeKnownBits(const Value *V, unsigned Depth,
                                  const Query &Q);

KnownBits llvm::computeKnownBits(const Value *V, const DataLayout &DL,
                                 unsigned Depth, AssumptionCache *AC,
                                 const Instruction *CxtI,
                                 const DominatorTree *DT,
                                 OptimizationRemarkEmitter *ORE,
                                 bool UseInstrInfo) {
  return ::computeKnownBits(
      V, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo, ORE));
}

KnownBits llvm::computeKnownBits(const Value *V, const APInt &DemandedElts,
                                 const DataLayout &DL, unsigned Depth,
                                 AssumptionCache *AC, const Instruction *CxtI,
                                 const DominatorTree *DT,
                                 OptimizationRemarkEmitter *ORE,
                                 bool UseInstrInfo) {
  return ::computeKnownBits(
      V, DemandedElts, Depth,
      Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo, ORE));
}

bool llvm::haveNoCommonBitsSet(const Value *LHS, const Value *RHS,
                               const DataLayout &DL, AssumptionCache *AC,
                               const Instruction *CxtI, const DominatorTree *DT,
                               bool UseInstrInfo) {
  assert(LHS->getType() == RHS->getType() &&
         "LHS and RHS should have the same type");
  assert(LHS->getType()->isIntOrIntVectorTy() &&
         "LHS and RHS should be integers");
  // Look for an inverted mask: (X & ~M) op (Y & M).
  {
    Value *M;
    if (match(LHS, m_c_And(m_Not(m_Value(M)), m_Value())) &&
        match(RHS, m_c_And(m_Specific(M), m_Value())))
      return true;
    if (match(RHS, m_c_And(m_Not(m_Value(M)), m_Value())) &&
        match(LHS, m_c_And(m_Specific(M), m_Value())))
      return true;
  }

  // X op (Y & ~X)
  if (match(RHS, m_c_And(m_Not(m_Specific(LHS)), m_Value())) ||
      match(LHS, m_c_And(m_Not(m_Specific(RHS)), m_Value())))
    return true;

  // X op ((X & Y) ^ Y) -- this is the canonical form of the previous pattern
  // for constant Y.
  Value *Y;
  if (match(RHS,
            m_c_Xor(m_c_And(m_Specific(LHS), m_Value(Y)), m_Deferred(Y))) ||
      match(LHS, m_c_Xor(m_c_And(m_Specific(RHS), m_Value(Y)), m_Deferred(Y))))
    return true;

  // Look for: (A & B) op ~(A | B)
  {
    Value *A, *B;
    if (match(LHS, m_And(m_Value(A), m_Value(B))) &&
        match(RHS, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))))
      return true;
    if (match(RHS, m_And(m_Value(A), m_Value(B))) &&
        match(LHS, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))))
      return true;
  }
  IntegerType *IT = cast<IntegerType>(LHS->getType()->getScalarType());
  KnownBits LHSKnown(IT->getBitWidth());
  KnownBits RHSKnown(IT->getBitWidth());
  computeKnownBits(LHS, LHSKnown, DL, 0, AC, CxtI, DT, nullptr, UseInstrInfo);
  computeKnownBits(RHS, RHSKnown, DL, 0, AC, CxtI, DT, nullptr, UseInstrInfo);
  return KnownBits::haveNoCommonBitsSet(LHSKnown, RHSKnown);
}

bool llvm::isOnlyUsedInZeroEqualityComparison(const Instruction *I) {
  return !I->user_empty() && all_of(I->users(), [](const User *U) {
    ICmpInst::Predicate P;
    return match(U, m_ICmp(P, m_Value(), m_Zero())) && ICmpInst::isEquality(P);
  });
}

static bool isKnownToBeAPowerOfTwo(const Value *V, bool OrZero, unsigned Depth,
                                   const Query &Q);

bool llvm::isKnownToBeAPowerOfTwo(const Value *V, const DataLayout &DL,
                                  bool OrZero, unsigned Depth,
                                  AssumptionCache *AC, const Instruction *CxtI,
                                  const DominatorTree *DT, bool UseInstrInfo) {
  return ::isKnownToBeAPowerOfTwo(
      V, OrZero, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo));
}

static bool isKnownNonZero(const Value *V, const APInt &DemandedElts,
                           unsigned Depth, const Query &Q);

static bool isKnownNonZero(const Value *V, unsigned Depth, const Query &Q);

bool llvm::isKnownNonZero(const Value *V, const DataLayout &DL, unsigned Depth,
                          AssumptionCache *AC, const Instruction *CxtI,
                          const DominatorTree *DT, bool UseInstrInfo) {
  return ::isKnownNonZero(V, Depth,
                          Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo));
}

bool llvm::isKnownNonNegative(const Value *V, const DataLayout &DL,
                              unsigned Depth, AssumptionCache *AC,
                              const Instruction *CxtI, const DominatorTree *DT,
                              bool UseInstrInfo) {
  KnownBits Known =
      computeKnownBits(V, DL, Depth, AC, CxtI, DT, nullptr, UseInstrInfo);
  return Known.isNonNegative();
}

bool llvm::isKnownPositive(const Value *V, const DataLayout &DL, unsigned Depth,
                           AssumptionCache *AC, const Instruction *CxtI,
                           const DominatorTree *DT, bool UseInstrInfo) {
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->getValue().isStrictlyPositive();

  // TODO: We'd doing two recursive queries here.  We should factor this such
  // that only a single query is needed.
  return isKnownNonNegative(V, DL, Depth, AC, CxtI, DT, UseInstrInfo) &&
         isKnownNonZero(V, DL, Depth, AC, CxtI, DT, UseInstrInfo);
}

bool llvm::isKnownNegative(const Value *V, const DataLayout &DL, unsigned Depth,
                           AssumptionCache *AC, const Instruction *CxtI,
                           const DominatorTree *DT, bool UseInstrInfo) {
  KnownBits Known =
      computeKnownBits(V, DL, Depth, AC, CxtI, DT, nullptr, UseInstrInfo);
  return Known.isNegative();
}

static bool isKnownNonEqual(const Value *V1, const Value *V2, unsigned Depth,
                            const Query &Q);

bool llvm::isKnownNonEqual(const Value *V1, const Value *V2,
                           const DataLayout &DL, AssumptionCache *AC,
                           const Instruction *CxtI, const DominatorTree *DT,
                           bool UseInstrInfo) {
  return ::isKnownNonEqual(V1, V2, 0,
                           Query(DL, AC, safeCxtI(V2, V1, CxtI), DT,
                                 UseInstrInfo, /*ORE=*/nullptr));
}

static bool MaskedValueIsZero(const Value *V, const APInt &Mask, unsigned Depth,
                              const Query &Q);

bool llvm::MaskedValueIsZero(const Value *V, const APInt &Mask,
                             const DataLayout &DL, unsigned Depth,
                             AssumptionCache *AC, const Instruction *CxtI,
                             const DominatorTree *DT, bool UseInstrInfo) {
  return ::MaskedValueIsZero(
      V, Mask, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo));
}

static unsigned ComputeNumSignBits(const Value *V, const APInt &DemandedElts,
                                   unsigned Depth, const Query &Q);

static unsigned ComputeNumSignBits(const Value *V, unsigned Depth,
                                   const Query &Q) {
  // FIXME: We currently have no way to represent the DemandedElts of a scalable
  // vector
  if (isa<ScalableVectorType>(V->getType()))
    return 1;

  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return ComputeNumSignBits(V, DemandedElts, Depth, Q);
}

unsigned llvm::ComputeNumSignBits(const Value *V, const DataLayout &DL,
                                  unsigned Depth, AssumptionCache *AC,
                                  const Instruction *CxtI,
                                  const DominatorTree *DT, bool UseInstrInfo) {
  return ::ComputeNumSignBits(
      V, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT, UseInstrInfo));
}

unsigned llvm::ComputeMaxSignificantBits(const Value *V, const DataLayout &DL,
                                         unsigned Depth, AssumptionCache *AC,
                                         const Instruction *CxtI,
                                         const DominatorTree *DT) {
  unsigned SignBits = ComputeNumSignBits(V, DL, Depth, AC, CxtI, DT);
  return V->getType()->getScalarSizeInBits() - SignBits + 1;
}

static void computeKnownBitsAddSub(bool Add, const Value *Op0, const Value *Op1,
                                   bool NSW, const APInt &DemandedElts,
                                   KnownBits &KnownOut, KnownBits &Known2,
                                   unsigned Depth, const Query &Q) {
  computeKnownBits(Op1, DemandedElts, KnownOut, Depth + 1, Q);

  // If one operand is unknown and we have no nowrap information,
  // the result will be unknown independently of the second operand.
  if (KnownOut.isUnknown() && !NSW)
    return;

  computeKnownBits(Op0, DemandedElts, Known2, Depth + 1, Q);
  KnownOut = KnownBits::computeForAddSub(Add, NSW, Known2, KnownOut);
}

static void computeKnownBitsMul(const Value *Op0, const Value *Op1, bool NSW,
                                const APInt &DemandedElts, KnownBits &Known,
                                KnownBits &Known2, unsigned Depth,
                                const Query &Q) {
  computeKnownBits(Op1, DemandedElts, Known, Depth + 1, Q);
  computeKnownBits(Op0, DemandedElts, Known2, Depth + 1, Q);

  bool isKnownNegative = false;
  bool isKnownNonNegative = false;
  // If the multiplication is known not to overflow, compute the sign bit.
  if (NSW) {
    if (Op0 == Op1) {
      // The product of a number with itself is non-negative.
      isKnownNonNegative = true;
    } else {
      bool isKnownNonNegativeOp1 = Known.isNonNegative();
      bool isKnownNonNegativeOp0 = Known2.isNonNegative();
      bool isKnownNegativeOp1 = Known.isNegative();
      bool isKnownNegativeOp0 = Known2.isNegative();
      // The product of two numbers with the same sign is non-negative.
      isKnownNonNegative = (isKnownNegativeOp1 && isKnownNegativeOp0) ||
                           (isKnownNonNegativeOp1 && isKnownNonNegativeOp0);
      // The product of a negative number and a non-negative number is either
      // negative or zero.
      if (!isKnownNonNegative)
        isKnownNegative =
            (isKnownNegativeOp1 && isKnownNonNegativeOp0 &&
             Known2.isNonZero()) ||
            (isKnownNegativeOp0 && isKnownNonNegativeOp1 && Known.isNonZero());
    }
  }

  bool SelfMultiply = Op0 == Op1;
  // TODO: SelfMultiply can be poison, but not undef.
  if (SelfMultiply)
    SelfMultiply &=
        isGuaranteedNotToBeUndefOrPoison(Op0, Q.AC, Q.CxtI, Q.DT, Depth + 1);
  Known = KnownBits::mul(Known, Known2, SelfMultiply);

  // Only make use of no-wrap flags if we failed to compute the sign bit
  // directly.  This matters if the multiplication always overflows, in
  // which case we prefer to follow the result of the direct computation,
  // though as the program is invoking undefined behaviour we can choose
  // whatever we like here.
  if (isKnownNonNegative && !Known.isNegative())
    Known.makeNonNegative();
  else if (isKnownNegative && !Known.isNonNegative())
    Known.makeNegative();
}

void llvm::computeKnownBitsFromRangeMetadata(const MDNode &Ranges,
                                             KnownBits &Known) {
  unsigned BitWidth = Known.getBitWidth();
  unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1);

  Known.Zero.setAllBits();
  Known.One.setAllBits();

  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());

    // The first CommonPrefixBits of all values in Range are equal.
    unsigned CommonPrefixBits =
        (Range.getUnsignedMax() ^ Range.getUnsignedMin()).countLeadingZeros();
    APInt Mask = APInt::getHighBitsSet(BitWidth, CommonPrefixBits);
    APInt UnsignedMax = Range.getUnsignedMax().zextOrTrunc(BitWidth);
    Known.One &= UnsignedMax & Mask;
    Known.Zero &= ~UnsignedMax & Mask;
  }
}

static bool isEphemeralValueOf(const Instruction *I, const Value *E) {
  SmallVector<const Value *, 16> WorkSet(1, I);
  SmallPtrSet<const Value *, 32> Visited;
  SmallPtrSet<const Value *, 16> EphValues;

  // The instruction defining an assumption's condition itself is always
  // considered ephemeral to that assumption (even if it has other
  // non-ephemeral users). See r246696's test case for an example.
  if (is_contained(I->operands(), E))
    return true;

  while (!WorkSet.empty()) {
    const Value *V = WorkSet.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    // If all uses of this value are ephemeral, then so is this value.
    if (llvm::all_of(V->users(), [&](const User *U) {
                                   return EphValues.count(U);
                                 })) {
      if (V == E)
        return true;

      if (V == I || (isa<Instruction>(V) &&
                     !cast<Instruction>(V)->mayHaveSideEffects() &&
                     !cast<Instruction>(V)->isTerminator())) {
       EphValues.insert(V);
       if (const User *U = dyn_cast<User>(V))
         append_range(WorkSet, U->operands());
      }
    }
  }

  return false;
}

// Is this an intrinsic that cannot be speculated but also cannot trap?
bool llvm::isAssumeLikeIntrinsic(const Instruction *I) {
  if (const IntrinsicInst *CI = dyn_cast<IntrinsicInst>(I))
    return CI->isAssumeLikeIntrinsic();

  return false;
}

bool llvm::isValidAssumeForContext(const Instruction *Inv,
                                   const Instruction *CxtI,
                                   const DominatorTree *DT) {
  // There are two restrictions on the use of an assume:
  //  1. The assume must dominate the context (or the control flow must
  //     reach the assume whenever it reaches the context).
  //  2. The context must not be in the assume's set of ephemeral values
  //     (otherwise we will use the assume to prove that the condition
  //     feeding the assume is trivially true, thus causing the removal of
  //     the assume).

  if (Inv->getParent() == CxtI->getParent()) {
    // If Inv and CtxI are in the same block, check if the assume (Inv) is first
    // in the BB.
    if (Inv->comesBefore(CxtI))
      return true;

    // Don't let an assume affect itself - this would cause the problems
    // `isEphemeralValueOf` is trying to prevent, and it would also make
    // the loop below go out of bounds.
    if (Inv == CxtI)
      return false;

    // The context comes first, but they're both in the same block.
    // Make sure there is nothing in between that might interrupt
    // the control flow, not even CxtI itself.
    // We limit the scan distance between the assume and its context instruction
    // to avoid a compile-time explosion. This limit is chosen arbitrarily, so
    // it can be adjusted if needed (could be turned into a cl::opt).
    auto Range = make_range(CxtI->getIterator(), Inv->getIterator());
    if (!isGuaranteedToTransferExecutionToSuccessor(Range, 15))
      return false;

    return !isEphemeralValueOf(Inv, CxtI);
  }

  // Inv and CxtI are in different blocks.
  if (DT) {
    if (DT->dominates(Inv, CxtI))
      return true;
  } else if (Inv->getParent() == CxtI->getParent()->getSinglePredecessor()) {
    // We don't have a DT, but this trivially dominates.
    return true;
  }

  return false;
}

static bool cmpExcludesZero(CmpInst::Predicate Pred, const Value *RHS) {
  // v u> y implies v != 0.
  if (Pred == ICmpInst::ICMP_UGT)
    return true;

  // Special-case v != 0 to also handle v != null.
  if (Pred == ICmpInst::ICMP_NE)
    return match(RHS, m_Zero());

  // All other predicates - rely on generic ConstantRange handling.
  const APInt *C;
  if (!match(RHS, m_APInt(C)))
    return false;

  ConstantRange TrueValues = ConstantRange::makeExactICmpRegion(Pred, *C);
  return !TrueValues.contains(APInt::getZero(C->getBitWidth()));
}

static bool isKnownNonZeroFromAssume(const Value *V, const Query &Q) {
  // Use of assumptions is context-sensitive. If we don't have a context, we
  // cannot use them!
  if (!Q.AC || !Q.CxtI)
    return false;

  if (Q.CxtI && V->getType()->isPointerTy()) {
    SmallVector<Attribute::AttrKind, 2> AttrKinds{Attribute::NonNull};
    if (!NullPointerIsDefined(Q.CxtI->getFunction(),
                              V->getType()->getPointerAddressSpace()))
      AttrKinds.push_back(Attribute::Dereferenceable);

    if (getKnowledgeValidInContext(V, AttrKinds, Q.CxtI, Q.DT, Q.AC))
      return true;
  }

  for (auto &AssumeVH : Q.AC->assumptionsFor(V)) {
    if (!AssumeVH)
      continue;
    CallInst *I = cast<CallInst>(AssumeVH);
    assert(I->getFunction() == Q.CxtI->getFunction() &&
           "Got assumption for the wrong function!");

    // Warning: This loop can end up being somewhat performance sensitive.
    // We're running this loop for once for each value queried resulting in a
    // runtime of ~O(#assumes * #values).

    assert(I->getCalledFunction()->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    Value *RHS;
    CmpInst::Predicate Pred;
    auto m_V = m_CombineOr(m_Specific(V), m_PtrToInt(m_Specific(V)));
    if (!match(I->getArgOperand(0), m_c_ICmp(Pred, m_V, m_Value(RHS))))
      return false;

    if (cmpExcludesZero(Pred, RHS) && isValidAssumeForContext(I, Q.CxtI, Q.DT))
      return true;
  }

  return false;
}

static void computeKnownBitsFromAssume(const Value *V, KnownBits &Known,
                                       unsigned Depth, const Query &Q) {
  // Use of assumptions is context-sensitive. If we don't have a context, we
  // cannot use them!
  if (!Q.AC || !Q.CxtI)
    return;

  unsigned BitWidth = Known.getBitWidth();

  // Refine Known set if the pointer alignment is set by assume bundles.
  if (V->getType()->isPointerTy()) {
    if (RetainedKnowledge RK = getKnowledgeValidInContext(
            V, {Attribute::Alignment}, Q.CxtI, Q.DT, Q.AC)) {
      if (isPowerOf2_64(RK.ArgValue))
        Known.Zero.setLowBits(Log2_64(RK.ArgValue));
    }
  }

  // Note that the patterns below need to be kept in sync with the code
  // in AssumptionCache::updateAffectedValues.

  for (auto &AssumeVH : Q.AC->assumptionsFor(V)) {
    if (!AssumeVH)
      continue;
    CallInst *I = cast<CallInst>(AssumeVH);
    assert(I->getParent()->getParent() == Q.CxtI->getParent()->getParent() &&
           "Got assumption for the wrong function!");

    // Warning: This loop can end up being somewhat performance sensitive.
    // We're running this loop for once for each value queried resulting in a
    // runtime of ~O(#assumes * #values).

    assert(I->getCalledFunction()->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    Value *Arg = I->getArgOperand(0);

    if (Arg == V && isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      assert(BitWidth == 1 && "assume operand is not i1?");
      Known.setAllOnes();
      return;
    }
    if (match(Arg, m_Not(m_Specific(V))) &&
        isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      assert(BitWidth == 1 && "assume operand is not i1?");
      Known.setAllZero();
      return;
    }

    // The remaining tests are all recursive, so bail out if we hit the limit.
    if (Depth == MaxAnalysisRecursionDepth)
      continue;

    ICmpInst *Cmp = dyn_cast<ICmpInst>(Arg);
    if (!Cmp)
      continue;

    // We are attempting to compute known bits for the operands of an assume.
    // Do not try to use other assumptions for those recursive calls because
    // that can lead to mutual recursion and a compile-time explosion.
    // An example of the mutual recursion: computeKnownBits can call
    // isKnownNonZero which calls computeKnownBitsFromAssume (this function)
    // and so on.
    Query QueryNoAC = Q;
    QueryNoAC.AC = nullptr;

    // Note that ptrtoint may change the bitwidth.
    Value *A, *B;
    auto m_V = m_CombineOr(m_Specific(V), m_PtrToInt(m_Specific(V)));

    CmpInst::Predicate Pred;
    uint64_t C;
    switch (Cmp->getPredicate()) {
    default:
      break;
    case ICmpInst::ICMP_EQ:
      // assume(v = a)
      if (match(Cmp, m_c_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        Known.Zero |= RHSKnown.Zero;
        Known.One  |= RHSKnown.One;
      // assume(v & b = a)
      } else if (match(Cmp,
                       m_c_ICmp(Pred, m_c_And(m_V, m_Value(B)), m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits MaskKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in the mask that are known to be one, we can propagate
        // known bits from the RHS to V.
        Known.Zero |= RHSKnown.Zero & MaskKnown.One;
        Known.One  |= RHSKnown.One  & MaskKnown.One;
      // assume(~(v & b) = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Not(m_c_And(m_V, m_Value(B))),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits MaskKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in the mask that are known to be one, we can propagate
        // inverted known bits from the RHS to V.
        Known.Zero |= RHSKnown.One  & MaskKnown.One;
        Known.One  |= RHSKnown.Zero & MaskKnown.One;
      // assume(v | b = a)
      } else if (match(Cmp,
                       m_c_ICmp(Pred, m_c_Or(m_V, m_Value(B)), m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits BKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in B that are known to be zero, we can propagate known
        // bits from the RHS to V.
        Known.Zero |= RHSKnown.Zero & BKnown.Zero;
        Known.One  |= RHSKnown.One  & BKnown.Zero;
      // assume(~(v | b) = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Not(m_c_Or(m_V, m_Value(B))),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits BKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in B that are known to be zero, we can propagate
        // inverted known bits from the RHS to V.
        Known.Zero |= RHSKnown.One  & BKnown.Zero;
        Known.One  |= RHSKnown.Zero & BKnown.Zero;
      // assume(v ^ b = a)
      } else if (match(Cmp,
                       m_c_ICmp(Pred, m_c_Xor(m_V, m_Value(B)), m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits BKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in B that are known to be zero, we can propagate known
        // bits from the RHS to V. For those bits in B that are known to be one,
        // we can propagate inverted known bits from the RHS to V.
        Known.Zero |= RHSKnown.Zero & BKnown.Zero;
        Known.One  |= RHSKnown.One  & BKnown.Zero;
        Known.Zero |= RHSKnown.One  & BKnown.One;
        Known.One  |= RHSKnown.Zero & BKnown.One;
      // assume(~(v ^ b) = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Not(m_c_Xor(m_V, m_Value(B))),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        KnownBits BKnown =
            computeKnownBits(B, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in B that are known to be zero, we can propagate
        // inverted known bits from the RHS to V. For those bits in B that are
        // known to be one, we can propagate known bits from the RHS to V.
        Known.Zero |= RHSKnown.One  & BKnown.Zero;
        Known.One  |= RHSKnown.Zero & BKnown.Zero;
        Known.Zero |= RHSKnown.Zero & BKnown.One;
        Known.One  |= RHSKnown.One  & BKnown.One;
      // assume(v << c = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Shl(m_V, m_ConstantInt(C)),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT) && C < BitWidth) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // For those bits in RHS that are known, we can propagate them to known
        // bits in V shifted to the right by C.
        RHSKnown.Zero.lshrInPlace(C);
        Known.Zero |= RHSKnown.Zero;
        RHSKnown.One.lshrInPlace(C);
        Known.One  |= RHSKnown.One;
      // assume(~(v << c) = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Not(m_Shl(m_V, m_ConstantInt(C))),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT) && C < BitWidth) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        // For those bits in RHS that are known, we can propagate them inverted
        // to known bits in V shifted to the right by C.
        RHSKnown.One.lshrInPlace(C);
        Known.Zero |= RHSKnown.One;
        RHSKnown.Zero.lshrInPlace(C);
        Known.One  |= RHSKnown.Zero;
      // assume(v >> c = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Shr(m_V, m_ConstantInt(C)),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT) && C < BitWidth) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        // For those bits in RHS that are known, we can propagate them to known
        // bits in V shifted to the right by C.
        Known.Zero |= RHSKnown.Zero << C;
        Known.One  |= RHSKnown.One  << C;
      // assume(~(v >> c) = a)
      } else if (match(Cmp, m_c_ICmp(Pred, m_Not(m_Shr(m_V, m_ConstantInt(C))),
                                     m_Value(A))) &&
                 isValidAssumeForContext(I, Q.CxtI, Q.DT) && C < BitWidth) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);
        // For those bits in RHS that are known, we can propagate them inverted
        // to known bits in V shifted to the right by C.
        Known.Zero |= RHSKnown.One  << C;
        Known.One  |= RHSKnown.Zero << C;
      }
      break;
    case ICmpInst::ICMP_SGE:
      // assume(v >=_s c) where c is non-negative
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth + 1, QueryNoAC).anyextOrTrunc(BitWidth);

        if (RHSKnown.isNonNegative()) {
          // We know that the sign bit is zero.
          Known.makeNonNegative();
        }
      }
      break;
    case ICmpInst::ICMP_SGT:
      // assume(v >_s c) where c is at least -1.
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth + 1, QueryNoAC).anyextOrTrunc(BitWidth);

        if (RHSKnown.isAllOnes() || RHSKnown.isNonNegative()) {
          // We know that the sign bit is zero.
          Known.makeNonNegative();
        }
      }
      break;
    case ICmpInst::ICMP_SLE:
      // assume(v <=_s c) where c is negative
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth + 1, QueryNoAC).anyextOrTrunc(BitWidth);

        if (RHSKnown.isNegative()) {
          // We know that the sign bit is one.
          Known.makeNegative();
        }
      }
      break;
    case ICmpInst::ICMP_SLT:
      // assume(v <_s c) where c is non-positive
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        if (RHSKnown.isZero() || RHSKnown.isNegative()) {
          // We know that the sign bit is one.
          Known.makeNegative();
        }
      }
      break;
    case ICmpInst::ICMP_ULE:
      // assume(v <=_u c)
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // Whatever high bits in c are zero are known to be zero.
        Known.Zero.setHighBits(RHSKnown.countMinLeadingZeros());
      }
      break;
    case ICmpInst::ICMP_ULT:
      // assume(v <_u c)
      if (match(Cmp, m_ICmp(Pred, m_V, m_Value(A))) &&
          isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
        KnownBits RHSKnown =
            computeKnownBits(A, Depth+1, QueryNoAC).anyextOrTrunc(BitWidth);

        // If the RHS is known zero, then this assumption must be wrong (nothing
        // is unsigned less than zero). Signal a conflict and get out of here.
        if (RHSKnown.isZero()) {
          Known.Zero.setAllBits();
          Known.One.setAllBits();
          break;
        }

        // Whatever high bits in c are zero are known to be zero (if c is a power
        // of 2, then one more).
        if (isKnownToBeAPowerOfTwo(A, false, Depth + 1, QueryNoAC))
          Known.Zero.setHighBits(RHSKnown.countMinLeadingZeros() + 1);
        else
          Known.Zero.setHighBits(RHSKnown.countMinLeadingZeros());
      }
      break;
    }
  }

  // If assumptions conflict with each other or previous known bits, then we
  // have a logical fallacy. It's possible that the assumption is not reachable,
  // so this isn't a real bug. On the other hand, the program may have undefined
  // behavior, or we might have a bug in the compiler. We can't assert/crash, so
  // clear out the known bits, try to warn the user, and hope for the best.
  if (Known.Zero.intersects(Known.One)) {
    Known.resetAll();

    if (Q.ORE)
      Q.ORE->emit([&]() {
        auto *CxtI = const_cast<Instruction *>(Q.CxtI);
        return OptimizationRemarkAnalysis("value-tracking", "BadAssumption",
                                          CxtI)
               << "Detected conflicting code assumptions. Program may "
                  "have undefined behavior, or compiler may have "
                  "internal error.";
      });
  }
}

/// Compute known bits from a shift operator, including those with a
/// non-constant shift amount. Known is the output of this function. Known2 is a
/// pre-allocated temporary with the same bit width as Known and on return
/// contains the known bit of the shift value source. KF is an
/// operator-specific function that, given the known-bits and a shift amount,
/// compute the implied known-bits of the shift operator's result respectively
/// for that shift amount. The results from calling KF are conservatively
/// combined for all permitted shift amounts.
static void computeKnownBitsFromShiftOperator(
    const Operator *I, const APInt &DemandedElts, KnownBits &Known,
    KnownBits &Known2, unsigned Depth, const Query &Q,
    function_ref<KnownBits(const KnownBits &, const KnownBits &)> KF) {
  unsigned BitWidth = Known.getBitWidth();
  computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);
  computeKnownBits(I->getOperand(1), DemandedElts, Known, Depth + 1, Q);

  // Note: We cannot use Known.Zero.getLimitedValue() here, because if
  // BitWidth > 64 and any upper bits are known, we'll end up returning the
  // limit value (which implies all bits are known).
  uint64_t ShiftAmtKZ = Known.Zero.zextOrTrunc(64).getZExtValue();
  uint64_t ShiftAmtKO = Known.One.zextOrTrunc(64).getZExtValue();
  bool ShiftAmtIsConstant = Known.isConstant();
  bool MaxShiftAmtIsOutOfRange = Known.getMaxValue().uge(BitWidth);

  if (ShiftAmtIsConstant) {
    Known = KF(Known2, Known);

    // If the known bits conflict, this must be an overflowing left shift, so
    // the shift result is poison. We can return anything we want. Choose 0 for
    // the best folding opportunity.
    if (Known.hasConflict())
      Known.setAllZero();

    return;
  }

  // If the shift amount could be greater than or equal to the bit-width of the
  // LHS, the value could be poison, but bail out because the check below is
  // expensive.
  // TODO: Should we just carry on?
  if (MaxShiftAmtIsOutOfRange) {
    Known.resetAll();
    return;
  }

  // It would be more-clearly correct to use the two temporaries for this
  // calculation. Reusing the APInts here to prevent unnecessary allocations.
  Known.resetAll();

  // If we know the shifter operand is nonzero, we can sometimes infer more
  // known bits. However this is expensive to compute, so be lazy about it and
  // only compute it when absolutely necessary.
  Optional<bool> ShifterOperandIsNonZero;

  // Early exit if we can't constrain any well-defined shift amount.
  if (!(ShiftAmtKZ & (PowerOf2Ceil(BitWidth) - 1)) &&
      !(ShiftAmtKO & (PowerOf2Ceil(BitWidth) - 1))) {
    ShifterOperandIsNonZero =
        isKnownNonZero(I->getOperand(1), DemandedElts, Depth + 1, Q);
    if (!*ShifterOperandIsNonZero)
      return;
  }

  Known.Zero.setAllBits();
  Known.One.setAllBits();
  for (unsigned ShiftAmt = 0; ShiftAmt < BitWidth; ++ShiftAmt) {
    // Combine the shifted known input bits only for those shift amounts
    // compatible with its known constraints.
    if ((ShiftAmt & ~ShiftAmtKZ) != ShiftAmt)
      continue;
    if ((ShiftAmt | ShiftAmtKO) != ShiftAmt)
      continue;
    // If we know the shifter is nonzero, we may be able to infer more known
    // bits. This check is sunk down as far as possible to avoid the expensive
    // call to isKnownNonZero if the cheaper checks above fail.
    if (ShiftAmt == 0) {
      if (!ShifterOperandIsNonZero)
        ShifterOperandIsNonZero =
            isKnownNonZero(I->getOperand(1), DemandedElts, Depth + 1, Q);
      if (*ShifterOperandIsNonZero)
        continue;
    }

    Known = KnownBits::commonBits(
        Known, KF(Known2, KnownBits::makeConstant(APInt(32, ShiftAmt))));
  }

  // If the known bits conflict, the result is poison. Return a 0 and hope the
  // caller can further optimize that.
  if (Known.hasConflict())
    Known.setAllZero();
}

static void computeKnownBitsFromOperator(const Operator *I,
                                         const APInt &DemandedElts,
                                         KnownBits &Known, unsigned Depth,
                                         const Query &Q) {
  unsigned BitWidth = Known.getBitWidth();

  KnownBits Known2(BitWidth);
  switch (I->getOpcode()) {
  default: break;
  case Instruction::Load:
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<LoadInst>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);
    break;
  case Instruction::And: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);

    Known &= Known2;

    // and(x, add (x, -1)) is a common idiom that always clears the low bit;
    // here we handle the more general case of adding any odd number by
    // matching the form add(x, add(x, y)) where y is odd.
    // TODO: This could be generalized to clearing any bit set in y where the
    // following bit is known to be unset in y.
    Value *X = nullptr, *Y = nullptr;
    if (!Known.Zero[0] && !Known.One[0] &&
        match(I, m_c_BinOp(m_Value(X), m_Add(m_Deferred(X), m_Value(Y))))) {
      Known2.resetAll();
      computeKnownBits(Y, DemandedElts, Known2, Depth + 1, Q);
      if (Known2.countMinTrailingOnes() > 0)
        Known.Zero.setBit(0);
    }
    break;
  }
  case Instruction::Or:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);

    Known |= Known2;
    break;
  case Instruction::Xor:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);

    Known ^= Known2;
    break;
  case Instruction::Mul: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsMul(I->getOperand(0), I->getOperand(1), NSW, DemandedElts,
                        Known, Known2, Depth, Q);
    break;
  }
  case Instruction::UDiv: {
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
    Known = KnownBits::udiv(Known, Known2);
    break;
  }
  case Instruction::Select: {
    const Value *LHS = nullptr, *RHS = nullptr;
    SelectPatternFlavor SPF = matchSelectPattern(I, LHS, RHS).Flavor;
    if (SelectPatternResult::isMinOrMax(SPF)) {
      computeKnownBits(RHS, Known, Depth + 1, Q);
      computeKnownBits(LHS, Known2, Depth + 1, Q);
      switch (SPF) {
      default:
        llvm_unreachable("Unhandled select pattern flavor!");
      case SPF_SMAX:
        Known = KnownBits::smax(Known, Known2);
        break;
      case SPF_SMIN:
        Known = KnownBits::smin(Known, Known2);
        break;
      case SPF_UMAX:
        Known = KnownBits::umax(Known, Known2);
        break;
      case SPF_UMIN:
        Known = KnownBits::umin(Known, Known2);
        break;
      }
      break;
    }

    computeKnownBits(I->getOperand(2), Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);

    // Only known if known in both the LHS and RHS.
    Known = KnownBits::commonBits(Known, Known2);

    if (SPF == SPF_ABS) {
      // RHS from matchSelectPattern returns the negation part of abs pattern.
      // If the negate has an NSW flag we can assume the sign bit of the result
      // will be 0 because that makes abs(INT_MIN) undefined.
      if (match(RHS, m_Neg(m_Specific(LHS))) &&
          Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(RHS)))
        Known.Zero.setSignBit();
    }

    break;
  }
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    break; // Can't work with floating point.
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    // Fall through and handle them the same as zext/trunc.
    LLVM_FALLTHROUGH;
  case Instruction::ZExt:
  case Instruction::Trunc: {
    Type *SrcTy = I->getOperand(0)->getType();

    unsigned SrcBitWidth;
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    Type *ScalarTy = SrcTy->getScalarType();
    SrcBitWidth = ScalarTy->isPointerTy() ?
      Q.DL.getPointerTypeSizeInBits(ScalarTy) :
      Q.DL.getTypeSizeInBits(ScalarTy);

    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    Known = Known.anyextOrTrunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    Known = Known.zextOrTrunc(BitWidth);
    break;
  }
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    if (SrcTy->isIntOrPtrTy() &&
        // TODO: For now, not handling conversions like:
        // (bitcast i64 %x to <2 x i32>)
        !I->getType()->isVectorTy()) {
      computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
      break;
    }

    // Handle cast from vector integer type to scalar or vector integer.
    auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcTy);
    if (!SrcVecTy || !SrcVecTy->getElementType()->isIntegerTy() ||
        !I->getType()->isIntOrIntVectorTy())
      break;

    // Look through a cast from narrow vector elements to wider type.
    // Examples: v4i32 -> v2i64, v3i8 -> v24
    unsigned SubBitWidth = SrcVecTy->getScalarSizeInBits();
    if (BitWidth % SubBitWidth == 0) {
      // Known bits are automatically intersected across demanded elements of a
      // vector. So for example, if a bit is computed as known zero, it must be
      // zero across all demanded elements of the vector.
      //
      // For this bitcast, each demanded element of the output is sub-divided
      // across a set of smaller vector elements in the source vector. To get
      // the known bits for an entire element of the output, compute the known
      // bits for each sub-element sequentially. This is done by shifting the
      // one-set-bit demanded elements parameter across the sub-elements for
      // consecutive calls to computeKnownBits. We are using the demanded
      // elements parameter as a mask operator.
      //
      // The known bits of each sub-element are then inserted into place
      // (dependent on endian) to form the full result of known bits.
      unsigned NumElts = DemandedElts.getBitWidth();
      unsigned SubScale = BitWidth / SubBitWidth;
      APInt SubDemandedElts = APInt::getZero(NumElts * SubScale);
      for (unsigned i = 0; i != NumElts; ++i) {
        if (DemandedElts[i])
          SubDemandedElts.setBit(i * SubScale);
      }

      KnownBits KnownSrc(SubBitWidth);
      for (unsigned i = 0; i != SubScale; ++i) {
        computeKnownBits(I->getOperand(0), SubDemandedElts.shl(i), KnownSrc,
                         Depth + 1, Q);
        unsigned ShiftElt = Q.DL.isLittleEndian() ? i : SubScale - 1 - i;
        Known.insertBits(KnownSrc, ShiftElt * SubBitWidth);
      }
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    Known = Known.trunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    Known = Known.sext(BitWidth);
    break;
  }
  case Instruction::Shl: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    auto KF = [NSW](const KnownBits &KnownVal, const KnownBits &KnownAmt) {
      KnownBits Result = KnownBits::shl(KnownVal, KnownAmt);
      // If this shift has "nsw" keyword, then the result is either a poison
      // value or has the same sign bit as the first operand.
      if (NSW) {
        if (KnownVal.Zero.isSignBitSet())
          Result.Zero.setSignBit();
        if (KnownVal.One.isSignBitSet())
          Result.One.setSignBit();
      }
      return Result;
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Depth, Q,
                                      KF);
    // Trailing zeros of a right-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setLowBits(C->countTrailingZeros());
    break;
  }
  case Instruction::LShr: {
    auto KF = [](const KnownBits &KnownVal, const KnownBits &KnownAmt) {
      return KnownBits::lshr(KnownVal, KnownAmt);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Depth, Q,
                                      KF);
    // Leading zeros of a left-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setHighBits(C->countLeadingZeros());
    break;
  }
  case Instruction::AShr: {
    auto KF = [](const KnownBits &KnownVal, const KnownBits &KnownAmt) {
      return KnownBits::ashr(KnownVal, KnownAmt);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Depth, Q,
                                      KF);
    break;
  }
  case Instruction::Sub: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(false, I->getOperand(0), I->getOperand(1), NSW,
                           DemandedElts, Known, Known2, Depth, Q);
    break;
  }
  case Instruction::Add: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(true, I->getOperand(0), I->getOperand(1), NSW,
                           DemandedElts, Known, Known2, Depth, Q);
    break;
  }
  case Instruction::SRem:
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
    Known = KnownBits::srem(Known, Known2);
    break;

  case Instruction::URem:
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
    Known = KnownBits::urem(Known, Known2);
    break;
  case Instruction::Alloca:
    Known.Zero.setLowBits(Log2(cast<AllocaInst>(I)->getAlign()));
    break;
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    // Accumulate the constant indices in a separate variable
    // to minimize the number of calls to computeForAddSub.
    APInt AccConstIndices(BitWidth, 0, /*IsSigned*/ true);

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      // TrailZ can only become smaller, short-circuit if we hit zero.
      if (Known.isUnknown())
        break;

      Value *Index = I->getOperand(i);

      // Handle case when index is zero.
      Constant *CIndex = dyn_cast<Constant>(Index);
      if (CIndex && CIndex->isZeroValue())
        continue;

      if (StructType *STy = GTI.getStructTypeOrNull()) {
        // Handle struct member offset arithmetic.

        assert(CIndex &&
               "Access to structure field must be known at compile time");

        if (CIndex->getType()->isVectorTy())
          Index = CIndex->getSplatValue();

        unsigned Idx = cast<ConstantInt>(Index)->getZExtValue();
        const StructLayout *SL = Q.DL.getStructLayout(STy);
        uint64_t Offset = SL->getElementOffset(Idx);
        AccConstIndices += Offset;
        continue;
      }

      // Handle array index arithmetic.
      Type *IndexedTy = GTI.getIndexedType();
      if (!IndexedTy->isSized()) {
        Known.resetAll();
        break;
      }

      unsigned IndexBitWidth = Index->getType()->getScalarSizeInBits();
      KnownBits IndexBits(IndexBitWidth);
      computeKnownBits(Index, IndexBits, Depth + 1, Q);
      TypeSize IndexTypeSize = Q.DL.getTypeAllocSize(IndexedTy);
      uint64_t TypeSizeInBytes = IndexTypeSize.getKnownMinSize();
      KnownBits ScalingFactor(IndexBitWidth);
      // Multiply by current sizeof type.
      // &A[i] == A + i * sizeof(*A[i]).
      if (IndexTypeSize.isScalable()) {
        // For scalable types the only thing we know about sizeof is
        // that this is a multiple of the minimum size.
        ScalingFactor.Zero.setLowBits(countTrailingZeros(TypeSizeInBytes));
      } else if (IndexBits.isConstant()) {
        APInt IndexConst = IndexBits.getConstant();
        APInt ScalingFactor(IndexBitWidth, TypeSizeInBytes);
        IndexConst *= ScalingFactor;
        AccConstIndices += IndexConst.sextOrTrunc(BitWidth);
        continue;
      } else {
        ScalingFactor =
            KnownBits::makeConstant(APInt(IndexBitWidth, TypeSizeInBytes));
      }
      IndexBits = KnownBits::mul(IndexBits, ScalingFactor);

      // If the offsets have a different width from the pointer, according
      // to the language reference we need to sign-extend or truncate them
      // to the width of the pointer.
      IndexBits = IndexBits.sextOrTrunc(BitWidth);

      // Note that inbounds does *not* guarantee nsw for the addition, as only
      // the offset is signed, while the base address is unsigned.
      Known = KnownBits::computeForAddSub(
          /*Add=*/true, /*NSW=*/false, Known, IndexBits);
    }
    if (!Known.isUnknown() && !AccConstIndices.isZero()) {
      KnownBits Index = KnownBits::makeConstant(AccConstIndices);
      Known = KnownBits::computeForAddSub(
          /*Add=*/true, /*NSW=*/false, Known, Index);
    }
    break;
  }
  case Instruction::PHI: {
    const PHINode *P = cast<PHINode>(I);
    BinaryOperator *BO = nullptr;
    Value *R = nullptr, *L = nullptr;
    if (matchSimpleRecurrence(P, BO, R, L)) {
      // Handle the case of a simple two-predecessor recurrence PHI.
      // There's a lot more that could theoretically be done here, but
      // this is sufficient to catch some interesting cases.
      unsigned Opcode = BO->getOpcode();

      // If this is a shift recurrence, we know the bits being shifted in.
      // We can combine that with information about the start value of the
      // recurrence to conclude facts about the result.
      if ((Opcode == Instruction::LShr || Opcode == Instruction::AShr ||
           Opcode == Instruction::Shl) &&
          BO->getOperand(0) == I) {

        // We have matched a recurrence of the form:
        // %iv = [R, %entry], [%iv.next, %backedge]
        // %iv.next = shift_op %iv, L

        // Recurse with the phi context to avoid concern about whether facts
        // inferred hold at original context instruction.  TODO: It may be
        // correct to use the original context.  IF warranted, explore and
        // add sufficient tests to cover.
        Query RecQ = Q;
        RecQ.CxtI = P;
        computeKnownBits(R, DemandedElts, Known2, Depth + 1, RecQ);
        switch (Opcode) {
        case Instruction::Shl:
          // A shl recurrence will only increase the tailing zeros
          Known.Zero.setLowBits(Known2.countMinTrailingZeros());
          break;
        case Instruction::LShr:
          // A lshr recurrence will preserve the leading zeros of the
          // start value
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          break;
        case Instruction::AShr:
          // An ashr recurrence will extend the initial sign bit
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          Known.One.setHighBits(Known2.countMinLeadingOnes());
          break;
        };
      }

      // Check for operations that have the property that if
      // both their operands have low zero bits, the result
      // will have low zero bits.
      if (Opcode == Instruction::Add ||
          Opcode == Instruction::Sub ||
          Opcode == Instruction::And ||
          Opcode == Instruction::Or ||
          Opcode == Instruction::Mul) {
        // Change the context instruction to the "edge" that flows into the
        // phi. This is important because that is where the value is actually
        // "evaluated" even though it is used later somewhere else. (see also
        // D69571).
        Query RecQ = Q;

        unsigned OpNum = P->getOperand(0) == R ? 0 : 1;
        Instruction *RInst = P->getIncomingBlock(OpNum)->getTerminator();
        Instruction *LInst = P->getIncomingBlock(1-OpNum)->getTerminator();

        // Ok, we have a PHI of the form L op= R. Check for low
        // zero bits.
        RecQ.CxtI = RInst;
        computeKnownBits(R, Known2, Depth + 1, RecQ);

        // We need to take the minimum number of known bits
        KnownBits Known3(BitWidth);
        RecQ.CxtI = LInst;
        computeKnownBits(L, Known3, Depth + 1, RecQ);

        Known.Zero.setLowBits(std::min(Known2.countMinTrailingZeros(),
                                       Known3.countMinTrailingZeros()));

        auto *OverflowOp = dyn_cast<OverflowingBinaryOperator>(BO);
        if (OverflowOp && Q.IIQ.hasNoSignedWrap(OverflowOp)) {
          // If initial value of recurrence is nonnegative, and we are adding
          // a nonnegative number with nsw, the result can only be nonnegative
          // or poison value regardless of the number of times we execute the
          // add in phi recurrence. If initial value is negative and we are
          // adding a negative number with nsw, the result can only be
          // negative or poison value. Similar arguments apply to sub and mul.
          //
          // (add non-negative, non-negative) --> non-negative
          // (add negative, negative) --> negative
          if (Opcode == Instruction::Add) {
            if (Known2.isNonNegative() && Known3.isNonNegative())
              Known.makeNonNegative();
            else if (Known2.isNegative() && Known3.isNegative())
              Known.makeNegative();
          }

          // (sub nsw non-negative, negative) --> non-negative
          // (sub nsw negative, non-negative) --> negative
          else if (Opcode == Instruction::Sub && BO->getOperand(0) == I) {
            if (Known2.isNonNegative() && Known3.isNegative())
              Known.makeNonNegative();
            else if (Known2.isNegative() && Known3.isNonNegative())
              Known.makeNegative();
          }

          // (mul nsw non-negative, non-negative) --> non-negative
          else if (Opcode == Instruction::Mul && Known2.isNonNegative() &&
                   Known3.isNonNegative())
            Known.makeNonNegative();
        }

        break;
      }
    }

    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      break;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    if (Depth < MaxAnalysisRecursionDepth - 1 && !Known.Zero && !Known.One) {
      // Skip if every incoming value references to ourself.
      if (isa_and_nonnull<UndefValue>(P->hasConstantValue()))
        break;

      Known.Zero.setAllBits();
      Known.One.setAllBits();
      for (unsigned u = 0, e = P->getNumIncomingValues(); u < e; ++u) {
        Value *IncValue = P->getIncomingValue(u);
        // Skip direct self references.
        if (IncValue == P) continue;

        // Change the context instruction to the "edge" that flows into the
        // phi. This is important because that is where the value is actually
        // "evaluated" even though it is used later somewhere else. (see also
        // D69571).
        Query RecQ = Q;
        RecQ.CxtI = P->getIncomingBlock(u)->getTerminator();

        Known2 = KnownBits(BitWidth);
        // Recurse, but cap the recursion to one level, because we don't
        // want to waste time spinning around in loops.
        computeKnownBits(IncValue, Known2, MaxAnalysisRecursionDepth - 1, RecQ);
        Known = KnownBits::commonBits(Known, Known2);
        // If all bits have been ruled out, there's no need to check
        // more operands.
        if (Known.isUnknown())
          break;
      }
    }
    break;
  }
  case Instruction::Call:
  case Instruction::Invoke:
    // If range metadata is attached to this call, set known bits from that,
    // and then intersect with known bits based on other properties of the
    // function.
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<Instruction>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);
    if (const Value *RV = cast<CallBase>(I)->getReturnedArgOperand()) {
      computeKnownBits(RV, Known2, Depth + 1, Q);
      Known.Zero |= Known2.Zero;
      Known.One |= Known2.One;
    }
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::abs: {
        computeKnownBits(I->getOperand(0), Known2, Depth + 1, Q);
        bool IntMinIsPoison = match(II->getArgOperand(1), m_One());
        Known = Known2.abs(IntMinIsPoison);
        break;
      }
      case Intrinsic::bitreverse:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);
        Known.Zero |= Known2.Zero.reverseBits();
        Known.One |= Known2.One.reverseBits();
        break;
      case Intrinsic::bswap:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Depth + 1, Q);
        Known.Zero |= Known2.Zero.byteSwap();
        Known.One |= Known2.One.byteSwap();
        break;
      case Intrinsic::ctlz: {
        computeKnownBits(I->getOperand(0), Known2, Depth + 1, Q);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleLZ = Known2.countMaxLeadingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleLZ = std::min(PossibleLZ, BitWidth - 1);
        unsigned LowBits = Log2_32(PossibleLZ)+1;
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::cttz: {
        computeKnownBits(I->getOperand(0), Known2, Depth + 1, Q);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleTZ = Known2.countMaxTrailingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleTZ = std::min(PossibleTZ, BitWidth - 1);
        unsigned LowBits = Log2_32(PossibleTZ)+1;
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::ctpop: {
        computeKnownBits(I->getOperand(0), Known2, Depth + 1, Q);
        // We can bound the space the count needs.  Also, bits known to be zero
        // can't contribute to the population.
        unsigned BitsPossiblySet = Known2.countMaxPopulation();
        unsigned LowBits = Log2_32(BitsPossiblySet)+1;
        Known.Zero.setBitsFrom(LowBits);
        // TODO: we could bound KnownOne using the lower bound on the number
        // of bits which might be set provided by popcnt KnownOne2.
        break;
      }
      case Intrinsic::fshr:
      case Intrinsic::fshl: {
        const APInt *SA;
        if (!match(I->getOperand(2), m_APInt(SA)))
          break;

        // Normalize to funnel shift left.
        uint64_t ShiftAmt = SA->urem(BitWidth);
        if (II->getIntrinsicID() == Intrinsic::fshr)
          ShiftAmt = BitWidth - ShiftAmt;

        KnownBits Known3(BitWidth);
        computeKnownBits(I->getOperand(0), Known2, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known3, Depth + 1, Q);

        Known.Zero =
            Known2.Zero.shl(ShiftAmt) | Known3.Zero.lshr(BitWidth - ShiftAmt);
        Known.One =
            Known2.One.shl(ShiftAmt) | Known3.One.lshr(BitWidth - ShiftAmt);
        break;
      }
      case Intrinsic::uadd_sat:
      case Intrinsic::usub_sat: {
        bool IsAdd = II->getIntrinsicID() == Intrinsic::uadd_sat;
        computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);

        // Add: Leading ones of either operand are preserved.
        // Sub: Leading zeros of LHS and leading ones of RHS are preserved
        // as leading zeros in the result.
        unsigned LeadingKnown;
        if (IsAdd)
          LeadingKnown = std::max(Known.countMinLeadingOnes(),
                                  Known2.countMinLeadingOnes());
        else
          LeadingKnown = std::max(Known.countMinLeadingZeros(),
                                  Known2.countMinLeadingOnes());

        Known = KnownBits::computeForAddSub(
            IsAdd, /* NSW */ false, Known, Known2);

        // We select between the operation result and all-ones/zero
        // respectively, so we can preserve known ones/zeros.
        if (IsAdd) {
          Known.One.setHighBits(LeadingKnown);
          Known.Zero.clearAllBits();
        } else {
          Known.Zero.setHighBits(LeadingKnown);
          Known.One.clearAllBits();
        }
        break;
      }
      case Intrinsic::umin:
        computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
        Known = KnownBits::umin(Known, Known2);
        break;
      case Intrinsic::umax:
        computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
        Known = KnownBits::umax(Known, Known2);
        break;
      case Intrinsic::smin:
        computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
        Known = KnownBits::smin(Known, Known2);
        break;
      case Intrinsic::smax:
        computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
        computeKnownBits(I->getOperand(1), Known2, Depth + 1, Q);
        Known = KnownBits::smax(Known, Known2);
        break;
      case Intrinsic::x86_sse42_crc32_64_64:
        Known.Zero.setBitsFrom(32);
        break;
      case Intrinsic::riscv_vsetvli:
      case Intrinsic::riscv_vsetvlimax:
        // Assume that VL output is positive and would fit in an int32_t.
        // TODO: VLEN might be capped at 16 bits in a future V spec update.
        if (BitWidth >= 32)
          Known.Zero.setBitsFrom(31);
        break;
      case Intrinsic::vscale: {
        if (!II->getParent() || !II->getFunction() ||
            !II->getFunction()->hasFnAttribute(Attribute::VScaleRange))
          break;

        auto Attr = II->getFunction()->getFnAttribute(Attribute::VScaleRange);
        Optional<unsigned> VScaleMax = Attr.getVScaleRangeMax();

        if (!VScaleMax)
          break;

        unsigned VScaleMin = Attr.getVScaleRangeMin();

        // If vscale min = max then we know the exact value at compile time
        // and hence we know the exact bits.
        if (VScaleMin == VScaleMax) {
          Known.One = VScaleMin;
          Known.Zero = VScaleMin;
          Known.Zero.flipAllBits();
          break;
        }

        unsigned FirstZeroHighBit = 32 - countLeadingZeros(*VScaleMax);
        if (FirstZeroHighBit < BitWidth)
          Known.Zero.setBitsFrom(FirstZeroHighBit);

        break;
      }
      }
    }
    break;
  case Instruction::ShuffleVector: {
    auto *Shuf = dyn_cast<ShuffleVectorInst>(I);
    // FIXME: Do we need to handle ConstantExpr involving shufflevectors?
    if (!Shuf) {
      Known.resetAll();
      return;
    }
    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    APInt DemandedLHS, DemandedRHS;
    if (!getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS)) {
      Known.resetAll();
      return;
    }
    Known.One.setAllBits();
    Known.Zero.setAllBits();
    if (!!DemandedLHS) {
      const Value *LHS = Shuf->getOperand(0);
      computeKnownBits(LHS, DemandedLHS, Known, Depth + 1, Q);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    if (!!DemandedRHS) {
      const Value *RHS = Shuf->getOperand(1);
      computeKnownBits(RHS, DemandedRHS, Known2, Depth + 1, Q);
      Known = KnownBits::commonBits(Known, Known2);
    }
    break;
  }
  case Instruction::InsertElement: {
    const Value *Vec = I->getOperand(0);
    const Value *Elt = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(I->getOperand(2));
    // Early out if the index is non-constant or out-of-range.
    unsigned NumElts = DemandedElts.getBitWidth();
    if (!CIdx || CIdx->getValue().uge(NumElts)) {
      Known.resetAll();
      return;
    }
    Known.One.setAllBits();
    Known.Zero.setAllBits();
    unsigned EltIdx = CIdx->getZExtValue();
    // Do we demand the inserted element?
    if (DemandedElts[EltIdx]) {
      computeKnownBits(Elt, Known, Depth + 1, Q);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    // We don't need the base vector element that has been inserted.
    APInt DemandedVecElts = DemandedElts;
    DemandedVecElts.clearBit(EltIdx);
    if (!!DemandedVecElts) {
      computeKnownBits(Vec, DemandedVecElts, Known2, Depth + 1, Q);
      Known = KnownBits::commonBits(Known, Known2);
    }
    break;
  }
  case Instruction::ExtractElement: {
    // Look through extract element. If the index is non-constant or
    // out-of-range demand all elements, otherwise just the extracted element.
    const Value *Vec = I->getOperand(0);
    const Value *Idx = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(Idx);
    if (isa<ScalableVectorType>(Vec->getType())) {
      // FIXME: there's probably *something* we can do with scalable vectors
      Known.resetAll();
      break;
    }
    unsigned NumElts = cast<FixedVectorType>(Vec->getType())->getNumElements();
    APInt DemandedVecElts = APInt::getAllOnes(NumElts);
    if (CIdx && CIdx->getValue().ult(NumElts))
      DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
    computeKnownBits(Vec, DemandedVecElts, Known, Depth + 1, Q);
    break;
  }
  case Instruction::ExtractValue:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->getOperand(0))) {
      const ExtractValueInst *EVI = cast<ExtractValueInst>(I);
      if (EVI->getNumIndices() != 1) break;
      if (EVI->getIndices()[0] == 0) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          computeKnownBitsAddSub(true, II->getArgOperand(0),
                                 II->getArgOperand(1), false, DemandedElts,
                                 Known, Known2, Depth, Q);
          break;
        case Intrinsic::usub_with_overflow:
        case Intrinsic::ssub_with_overflow:
          computeKnownBitsAddSub(false, II->getArgOperand(0),
                                 II->getArgOperand(1), false, DemandedElts,
                                 Known, Known2, Depth, Q);
          break;
        case Intrinsic::umul_with_overflow:
        case Intrinsic::smul_with_overflow:
          computeKnownBitsMul(II->getArgOperand(0), II->getArgOperand(1), false,
                              DemandedElts, Known, Known2, Depth, Q);
          break;
        }
      }
    }
    break;
  case Instruction::Freeze:
    if (isGuaranteedNotToBePoison(I->getOperand(0), Q.AC, Q.CxtI, Q.DT,
                                  Depth + 1))
      computeKnownBits(I->getOperand(0), Known, Depth + 1, Q);
    break;
  }
}

/// Determine which bits of V are known to be either zero or one and return
/// them.
KnownBits computeKnownBits(const Value *V, const APInt &DemandedElts,
                           unsigned Depth, const Query &Q) {
  KnownBits Known(getBitWidth(V->getType(), Q.DL));
  computeKnownBits(V, DemandedElts, Known, Depth, Q);
  return Known;
}

/// Determine which bits of V are known to be either zero or one and return
/// them.
KnownBits computeKnownBits(const Value *V, unsigned Depth, const Query &Q) {
  KnownBits Known(getBitWidth(V->getType(), Q.DL));
  computeKnownBits(V, Known, Depth, Q);
  return Known;
}

/// Determine which bits of V are known to be either zero or one and return
/// them in the Known bit set.
///
/// NOTE: we cannot consider 'undef' to be "IsZero" here.  The problem is that
/// we cannot optimize based on the assumption that it is zero without changing
/// it to be an explicit zero.  If we don't change it to zero, other code could
/// optimized based on the contradictory assumption that it is non-zero.
/// Because instcombine aggressively folds operations with undef args anyway,
/// this won't lose us code quality.
///
/// This function is defined on values with integer type, values with pointer
/// type, and vectors of integers.  In the case
/// where V is a vector, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the demanded elements in the vector specified by DemandedElts.
void computeKnownBits(const Value *V, const APInt &DemandedElts,
                      KnownBits &Known, unsigned Depth, const Query &Q) {
  if (!DemandedElts || isa<ScalableVectorType>(V->getType())) {
    // No demanded elts or V is a scalable vector, better to assume we don't
    // know anything.
    Known.resetAll();
    return;
  }

  assert(V && "No Value?");
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

#ifndef NDEBUG
  Type *Ty = V->getType();
  unsigned BitWidth = Known.getBitWidth();

  assert((Ty->isIntOrIntVectorTy(BitWidth) || Ty->isPtrOrPtrVectorTy()) &&
         "Not integer or pointer type!");

  if (auto *FVTy = dyn_cast<FixedVectorType>(Ty)) {
    assert(
        FVTy->getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars");
  }

  Type *ScalarTy = Ty->getScalarType();
  if (ScalarTy->isPointerTy()) {
    assert(BitWidth == Q.DL.getPointerTypeSizeInBits(ScalarTy) &&
           "V and Known should have same BitWidth");
  } else {
    assert(BitWidth == Q.DL.getTypeSizeInBits(ScalarTy) &&
           "V and Known should have same BitWidth");
  }
#endif

  const APInt *C;
  if (match(V, m_APInt(C))) {
    // We know all of the bits for a scalar constant or a splat vector constant!
    Known = KnownBits::makeConstant(*C);
    return;
  }
  // Null and aggregate-zero are all-zeros.
  if (isa<ConstantPointerNull>(V) || isa<ConstantAggregateZero>(V)) {
    Known.setAllZero();
    return;
  }
  // Handle a constant vector by taking the intersection of the known bits of
  // each element.
  if (const ConstantDataVector *CDV = dyn_cast<ConstantDataVector>(V)) {
    // We know that CDV must be a vector of integers. Take the intersection of
    // each element.
    Known.Zero.setAllBits(); Known.One.setAllBits();
    for (unsigned i = 0, e = CDV->getNumElements(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      APInt Elt = CDV->getElementAsAPInt(i);
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    return;
  }

  if (const auto *CV = dyn_cast<ConstantVector>(V)) {
    // We know that CV must be a vector of integers. Take the intersection of
    // each element.
    Known.Zero.setAllBits(); Known.One.setAllBits();
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      Constant *Element = CV->getAggregateElement(i);
      auto *ElementCI = dyn_cast_or_null<ConstantInt>(Element);
      if (!ElementCI) {
        Known.resetAll();
        return;
      }
      const APInt &Elt = ElementCI->getValue();
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    return;
  }

  // Start out not knowing anything.
  Known.resetAll();

  // We can't imply anything about undefs.
  if (isa<UndefValue>(V))
    return;

  // There's no point in looking through other users of ConstantData for
  // assumptions.  Confirm that we've handled them all.
  assert(!isa<ConstantData>(V) && "Unhandled constant data!");

  // All recursive calls that increase depth must come after this.
  if (Depth == MaxAnalysisRecursionDepth)
    return;

  // A weak GlobalAlias is totally unknown. A non-weak GlobalAlias has
  // the bits of its aliasee.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
    if (!GA->isInterposable())
      computeKnownBits(GA->getAliasee(), Known, Depth + 1, Q);
    return;
  }

  if (const Operator *I = dyn_cast<Operator>(V))
    computeKnownBitsFromOperator(I, DemandedElts, Known, Depth, Q);

  // Aligned pointers have trailing zeros - refine Known.Zero set
  if (isa<PointerType>(V->getType())) {
    Align Alignment = V->getPointerAlignment(Q.DL);
    Known.Zero.setLowBits(Log2(Alignment));
  }

  // computeKnownBitsFromAssume strictly refines Known.
  // Therefore, we run them after computeKnownBitsFromOperator.

  // Check whether a nearby assume intrinsic can determine some known bits.
  computeKnownBitsFromAssume(V, Known, Depth, Q);

  assert((Known.Zero & Known.One) == 0 && "Bits known to be one AND zero?");
}

/// Try to detect a recurrence that the value of the induction variable is
/// always a power of two (or zero).
static bool isPowerOfTwoRecurrence(const PHINode *PN, bool OrZero,
                                   unsigned Depth, Query &Q) {
  BinaryOperator *BO = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  if (!matchSimpleRecurrence(PN, BO, Start, Step))
    return false;

  // Initial value must be a power of two.
  for (const Use &U : PN->operands()) {
    if (U.get() == Start) {
      // Initial value comes from a different BB, need to adjust context
      // instruction for analysis.
      Q.CxtI = PN->getIncomingBlock(U)->getTerminator();
      if (!isKnownToBeAPowerOfTwo(Start, OrZero, Depth, Q))
        return false;
    }
  }

  // Except for Mul, the induction variable must be on the left side of the
  // increment expression, otherwise its value can be arbitrary.
  if (BO->getOpcode() != Instruction::Mul && BO->getOperand(1) != Step)
    return false;

  Q.CxtI = BO->getParent()->getTerminator();
  switch (BO->getOpcode()) {
  case Instruction::Mul:
    // Power of two is closed under multiplication.
    return (OrZero || Q.IIQ.hasNoUnsignedWrap(BO) ||
            Q.IIQ.hasNoSignedWrap(BO)) &&
           isKnownToBeAPowerOfTwo(Step, OrZero, Depth, Q);
  case Instruction::SDiv:
    // Start value must not be signmask for signed division, so simply being a
    // power of two is not sufficient, and it has to be a constant.
    if (!match(Start, m_Power2()) || match(Start, m_SignMask()))
      return false;
    LLVM_FALLTHROUGH;
  case Instruction::UDiv:
    // Divisor must be a power of two.
    // If OrZero is false, cannot guarantee induction variable is non-zero after
    // division, same for Shr, unless it is exact division.
    return (OrZero || Q.IIQ.isExact(BO)) &&
           isKnownToBeAPowerOfTwo(Step, false, Depth, Q);
  case Instruction::Shl:
    return OrZero || Q.IIQ.hasNoUnsignedWrap(BO) || Q.IIQ.hasNoSignedWrap(BO);
  case Instruction::AShr:
    if (!match(Start, m_Power2()) || match(Start, m_SignMask()))
      return false;
    LLVM_FALLTHROUGH;
  case Instruction::LShr:
    return OrZero || Q.IIQ.isExact(BO);
  default:
    return false;
  }
}

/// Return true if the given value is known to have exactly one
/// bit set when defined. For vectors return true if every element is known to
/// be a power of two when defined. Supports values with integer or pointer
/// types and vectors of integers.
bool isKnownToBeAPowerOfTwo(const Value *V, bool OrZero, unsigned Depth,
                            const Query &Q) {
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  // Attempt to match against constants.
  if (OrZero && match(V, m_Power2OrZero()))
      return true;
  if (match(V, m_Power2()))
      return true;

  // 1 << X is clearly a power of two if the one is not shifted off the end.  If
  // it is shifted off the end then the result is undefined.
  if (match(V, m_Shl(m_One(), m_Value())))
    return true;

  // (signmask) >>l X is clearly a power of two if the one is not shifted off
  // the bottom.  If it is shifted off the bottom then the result is undefined.
  if (match(V, m_LShr(m_SignMask(), m_Value())))
    return true;

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxAnalysisRecursionDepth)
    return false;

  Value *X = nullptr, *Y = nullptr;
  // A shift left or a logical shift right of a power of two is a power of two
  // or zero.
  if (OrZero && (match(V, m_Shl(m_Value(X), m_Value())) ||
                 match(V, m_LShr(m_Value(X), m_Value()))))
    return isKnownToBeAPowerOfTwo(X, /*OrZero*/ true, Depth, Q);

  if (const ZExtInst *ZI = dyn_cast<ZExtInst>(V))
    return isKnownToBeAPowerOfTwo(ZI->getOperand(0), OrZero, Depth, Q);

  if (const SelectInst *SI = dyn_cast<SelectInst>(V))
    return isKnownToBeAPowerOfTwo(SI->getTrueValue(), OrZero, Depth, Q) &&
           isKnownToBeAPowerOfTwo(SI->getFalseValue(), OrZero, Depth, Q);

  // Peek through min/max.
  if (match(V, m_MaxOrMin(m_Value(X), m_Value(Y)))) {
    return isKnownToBeAPowerOfTwo(X, OrZero, Depth, Q) &&
           isKnownToBeAPowerOfTwo(Y, OrZero, Depth, Q);
  }

  if (OrZero && match(V, m_And(m_Value(X), m_Value(Y)))) {
    // A power of two and'd with anything is a power of two or zero.
    if (isKnownToBeAPowerOfTwo(X, /*OrZero*/ true, Depth, Q) ||
        isKnownToBeAPowerOfTwo(Y, /*OrZero*/ true, Depth, Q))
      return true;
    // X & (-X) is always a power of two or zero.
    if (match(X, m_Neg(m_Specific(Y))) || match(Y, m_Neg(m_Specific(X))))
      return true;
    return false;
  }

  // Adding a power-of-two or zero to the same power-of-two or zero yields
  // either the original power-of-two, a larger power-of-two or zero.
  if (match(V, m_Add(m_Value(X), m_Value(Y)))) {
    const OverflowingBinaryOperator *VOBO = cast<OverflowingBinaryOperator>(V);
    if (OrZero || Q.IIQ.hasNoUnsignedWrap(VOBO) ||
        Q.IIQ.hasNoSignedWrap(VOBO)) {
      if (match(X, m_And(m_Specific(Y), m_Value())) ||
          match(X, m_And(m_Value(), m_Specific(Y))))
        if (isKnownToBeAPowerOfTwo(Y, OrZero, Depth, Q))
          return true;
      if (match(Y, m_And(m_Specific(X), m_Value())) ||
          match(Y, m_And(m_Value(), m_Specific(X))))
        if (isKnownToBeAPowerOfTwo(X, OrZero, Depth, Q))
          return true;

      unsigned BitWidth = V->getType()->getScalarSizeInBits();
      KnownBits LHSBits(BitWidth);
      computeKnownBits(X, LHSBits, Depth, Q);

      KnownBits RHSBits(BitWidth);
      computeKnownBits(Y, RHSBits, Depth, Q);
      // If i8 V is a power of two or zero:
      //  ZeroBits: 1 1 1 0 1 1 1 1
      // ~ZeroBits: 0 0 0 1 0 0 0 0
      if ((~(LHSBits.Zero & RHSBits.Zero)).isPowerOf2())
        // If OrZero isn't set, we cannot give back a zero result.
        // Make sure either the LHS or RHS has a bit set.
        if (OrZero || RHSBits.One.getBoolValue() || LHSBits.One.getBoolValue())
          return true;
    }
  }

  // A PHI node is power of two if all incoming values are power of two, or if
  // it is an induction variable where in each step its value is a power of two.
  if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    Query RecQ = Q;

    // Check if it is an induction variable and always power of two.
    if (isPowerOfTwoRecurrence(PN, OrZero, Depth, RecQ))
      return true;

    // Recursively check all incoming values. Limit recursion to 2 levels, so
    // that search complexity is limited to number of operands^2.
    unsigned NewDepth = std::max(Depth, MaxAnalysisRecursionDepth - 1);
    return llvm::all_of(PN->operands(), [&](const Use &U) {
      // Value is power of 2 if it is coming from PHI node itself by induction.
      if (U.get() == PN)
        return true;

      // Change the context instruction to the incoming block where it is
      // evaluated.
      RecQ.CxtI = PN->getIncomingBlock(U)->getTerminator();
      return isKnownToBeAPowerOfTwo(U.get(), OrZero, NewDepth, RecQ);
    });
  }

  // An exact divide or right shift can only shift off zero bits, so the result
  // is a power of two only if the first operand is a power of two and not
  // copying a sign bit (sdiv int_min, 2).
  if (match(V, m_Exact(m_LShr(m_Value(), m_Value()))) ||
      match(V, m_Exact(m_UDiv(m_Value(), m_Value())))) {
    return isKnownToBeAPowerOfTwo(cast<Operator>(V)->getOperand(0), OrZero,
                                  Depth, Q);
  }

  return false;
}

/// Test whether a GEP's result is known to be non-null.
///
/// Uses properties inherent in a GEP to try to determine whether it is known
/// to be non-null.
///
/// Currently this routine does not support vector GEPs.
static bool isGEPKnownNonNull(const GEPOperator *GEP, unsigned Depth,
                              const Query &Q) {
  const Function *F = nullptr;
  if (const Instruction *I = dyn_cast<Instruction>(GEP))
    F = I->getFunction();

  if (!GEP->isInBounds() ||
      NullPointerIsDefined(F, GEP->getPointerAddressSpace()))
    return false;

  // FIXME: Support vector-GEPs.
  assert(GEP->getType()->isPointerTy() && "We only support plain pointer GEP");

  // If the base pointer is non-null, we cannot walk to a null address with an
  // inbounds GEP in address space zero.
  if (isKnownNonZero(GEP->getPointerOperand(), Depth, Q))
    return true;

  // Walk the GEP operands and see if any operand introduces a non-zero offset.
  // If so, then the GEP cannot produce a null pointer, as doing so would
  // inherently violate the inbounds contract within address space zero.
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    // Struct types are easy -- they must always be indexed by a constant.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      ConstantInt *OpC = cast<ConstantInt>(GTI.getOperand());
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = Q.DL.getStructLayout(STy);
      uint64_t ElementOffset = SL->getElementOffset(ElementIdx);
      if (ElementOffset > 0)
        return true;
      continue;
    }

    // If we have a zero-sized type, the index doesn't matter. Keep looping.
    if (Q.DL.getTypeAllocSize(GTI.getIndexedType()).getKnownMinSize() == 0)
      continue;

    // Fast path the constant operand case both for efficiency and so we don't
    // increment Depth when just zipping down an all-constant GEP.
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand())) {
      if (!OpC->isZero())
        return true;
      continue;
    }

    // We post-increment Depth here because while isKnownNonZero increments it
    // as well, when we pop back up that increment won't persist. We don't want
    // to recurse 10k times just because we have 10k GEP operands. We don't
    // bail completely out because we want to handle constant GEPs regardless
    // of depth.
    if (Depth++ >= MaxAnalysisRecursionDepth)
      continue;

    if (isKnownNonZero(GTI.getOperand(), Depth, Q))
      return true;
  }

  return false;
}

static bool isKnownNonNullFromDominatingCondition(const Value *V,
                                                  const Instruction *CtxI,
                                                  const DominatorTree *DT) {
  if (isa<Constant>(V))
    return false;

  if (!CtxI || !DT)
    return false;

  unsigned NumUsesExplored = 0;
  for (auto *U : V->users()) {
    // Avoid massive lists
    if (NumUsesExplored >= DomConditionsMaxUses)
      break;
    NumUsesExplored++;

    // If the value is used as an argument to a call or invoke, then argument
    // attributes may provide an answer about null-ness.
    if (const auto *CB = dyn_cast<CallBase>(U))
      if (auto *CalledFunc = CB->getCalledFunction())
        for (const Argument &Arg : CalledFunc->args())
          if (CB->getArgOperand(Arg.getArgNo()) == V &&
              Arg.hasNonNullAttr(/* AllowUndefOrPoison */ false) &&
              DT->dominates(CB, CtxI))
            return true;

    // If the value is used as a load/store, then the pointer must be non null.
    if (V == getLoadStorePointerOperand(U)) {
      const Instruction *I = cast<Instruction>(U);
      if (!NullPointerIsDefined(I->getFunction(),
                                V->getType()->getPointerAddressSpace()) &&
          DT->dominates(I, CtxI))
        return true;
    }

    // Consider only compare instructions uniquely controlling a branch
    Value *RHS;
    CmpInst::Predicate Pred;
    if (!match(U, m_c_ICmp(Pred, m_Specific(V), m_Value(RHS))))
      continue;

    bool NonNullIfTrue;
    if (cmpExcludesZero(Pred, RHS))
      NonNullIfTrue = true;
    else if (cmpExcludesZero(CmpInst::getInversePredicate(Pred), RHS))
      NonNullIfTrue = false;
    else
      continue;

    SmallVector<const User *, 4> WorkList;
    SmallPtrSet<const User *, 4> Visited;
    for (auto *CmpU : U->users()) {
      assert(WorkList.empty() && "Should be!");
      if (Visited.insert(CmpU).second)
        WorkList.push_back(CmpU);

      while (!WorkList.empty()) {
        auto *Curr = WorkList.pop_back_val();

        // If a user is an AND, add all its users to the work list. We only
        // propagate "pred != null" condition through AND because it is only
        // correct to assume that all conditions of AND are met in true branch.
        // TODO: Support similar logic of OR and EQ predicate?
        if (NonNullIfTrue)
          if (match(Curr, m_LogicalAnd(m_Value(), m_Value()))) {
            for (auto *CurrU : Curr->users())
              if (Visited.insert(CurrU).second)
                WorkList.push_back(CurrU);
            continue;
          }

        if (const BranchInst *BI = dyn_cast<BranchInst>(Curr)) {
          assert(BI->isConditional() && "uses a comparison!");

          BasicBlock *NonNullSuccessor =
              BI->getSuccessor(NonNullIfTrue ? 0 : 1);
          BasicBlockEdge Edge(BI->getParent(), NonNullSuccessor);
          if (Edge.isSingleEdge() && DT->dominates(Edge, CtxI->getParent()))
            return true;
        } else if (NonNullIfTrue && isGuard(Curr) &&
                   DT->dominates(cast<Instruction>(Curr), CtxI)) {
          return true;
        }
      }
    }
  }

  return false;
}

/// Does the 'Range' metadata (which must be a valid MD_range operand list)
/// ensure that the value it's attached to is never Value?  'RangeType' is
/// is the type of the value described by the range.
static bool rangeMetadataExcludesValue(const MDNode* Ranges, const APInt& Value) {
  const unsigned NumRanges = Ranges->getNumOperands() / 2;
  assert(NumRanges >= 1);
  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());
    if (Range.contains(Value))
      return false;
  }
  return true;
}

/// Try to detect a recurrence that monotonically increases/decreases from a
/// non-zero starting value. These are common as induction variables.
static bool isNonZeroRecurrence(const PHINode *PN) {
  BinaryOperator *BO = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  const APInt *StartC, *StepC;
  if (!matchSimpleRecurrence(PN, BO, Start, Step) ||
      !match(Start, m_APInt(StartC)) || StartC->isZero())
    return false;

  switch (BO->getOpcode()) {
  case Instruction::Add:
    // Starting from non-zero and stepping away from zero can never wrap back
    // to zero.
    return BO->hasNoUnsignedWrap() ||
           (BO->hasNoSignedWrap() && match(Step, m_APInt(StepC)) &&
            StartC->isNegative() == StepC->isNegative());
  case Instruction::Mul:
    return (BO->hasNoUnsignedWrap() || BO->hasNoSignedWrap()) &&
           match(Step, m_APInt(StepC)) && !StepC->isZero();
  case Instruction::Shl:
    return BO->hasNoUnsignedWrap() || BO->hasNoSignedWrap();
  case Instruction::AShr:
  case Instruction::LShr:
    return BO->isExact();
  default:
    return false;
  }
}

/// Return true if the given value is known to be non-zero when defined. For
/// vectors, return true if every demanded element is known to be non-zero when
/// defined. For pointers, if the context instruction and dominator tree are
/// specified, perform context-sensitive analysis and return true if the
/// pointer couldn't possibly be null at the specified instruction.
/// Supports values with integer or pointer type and vectors of integers.
bool isKnownNonZero(const Value *V, const APInt &DemandedElts, unsigned Depth,
                    const Query &Q) {
  // FIXME: We currently have no way to represent the DemandedElts of a scalable
  // vector
  if (isa<ScalableVectorType>(V->getType()))
    return false;

  if (auto *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return false;
    if (isa<ConstantInt>(C))
      // Must be non-zero due to null test above.
      return true;

    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
      // See the comment for IntToPtr/PtrToInt instructions below.
      if (CE->getOpcode() == Instruction::IntToPtr ||
          CE->getOpcode() == Instruction::PtrToInt)
        if (Q.DL.getTypeSizeInBits(CE->getOperand(0)->getType())
                .getFixedSize() <=
            Q.DL.getTypeSizeInBits(CE->getType()).getFixedSize())
          return isKnownNonZero(CE->getOperand(0), Depth, Q);
    }

    // For constant vectors, check that all elements are undefined or known
    // non-zero to determine that the whole vector is known non-zero.
    if (auto *VecTy = dyn_cast<FixedVectorType>(C->getType())) {
      for (unsigned i = 0, e = VecTy->getNumElements(); i != e; ++i) {
        if (!DemandedElts[i])
          continue;
        Constant *Elt = C->getAggregateElement(i);
        if (!Elt || Elt->isNullValue())
          return false;
        if (!isa<UndefValue>(Elt) && !isa<ConstantInt>(Elt))
          return false;
      }
      return true;
    }

    // A global variable in address space 0 is non null unless extern weak
    // or an absolute symbol reference. Other address spaces may have null as a
    // valid address for a global, so we can't assume anything.
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      if (!GV->isAbsoluteSymbolRef() && !GV->hasExternalWeakLinkage() &&
          GV->getType()->getAddressSpace() == 0)
        return true;
    } else
      return false;
  }

  if (auto *I = dyn_cast<Instruction>(V)) {
    if (MDNode *Ranges = Q.IIQ.getMetadata(I, LLVMContext::MD_range)) {
      // If the possible ranges don't contain zero, then the value is
      // definitely non-zero.
      if (auto *Ty = dyn_cast<IntegerType>(V->getType())) {
        const APInt ZeroValue(Ty->getBitWidth(), 0);
        if (rangeMetadataExcludesValue(Ranges, ZeroValue))
          return true;
      }
    }
  }

  if (isKnownNonZeroFromAssume(V, Q))
    return true;

  // Some of the tests below are recursive, so bail out if we hit the limit.
  if (Depth++ >= MaxAnalysisRecursionDepth)
    return false;

  // Check for pointer simplifications.

  if (PointerType *PtrTy = dyn_cast<PointerType>(V->getType())) {
    // Alloca never returns null, malloc might.
    if (isa<AllocaInst>(V) && Q.DL.getAllocaAddrSpace() == 0)
      return true;

    // A byval, inalloca may not be null in a non-default addres space. A
    // nonnull argument is assumed never 0.
    if (const Argument *A = dyn_cast<Argument>(V)) {
      if (((A->hasPassPointeeByValueCopyAttr() &&
            !NullPointerIsDefined(A->getParent(), PtrTy->getAddressSpace())) ||
           A->hasNonNullAttr()))
        return true;
    }

    // A Load tagged with nonnull metadata is never null.
    if (const LoadInst *LI = dyn_cast<LoadInst>(V))
      if (Q.IIQ.getMetadata(LI, LLVMContext::MD_nonnull))
        return true;

    if (const auto *Call = dyn_cast<CallBase>(V)) {
      if (Call->isReturnNonNull())
        return true;
      if (const auto *RP = getArgumentAliasingToReturnedPointer(Call, true))
        return isKnownNonZero(RP, Depth, Q);
    }
  }

  if (isKnownNonNullFromDominatingCondition(V, Q.CxtI, Q.DT))
    return true;

  // Check for recursive pointer simplifications.
  if (V->getType()->isPointerTy()) {
    // Look through bitcast operations, GEPs, and int2ptr instructions as they
    // do not alter the value, or at least not the nullness property of the
    // value, e.g., int2ptr is allowed to zero/sign extend the value.
    //
    // Note that we have to take special care to avoid looking through
    // truncating casts, e.g., int2ptr/ptr2int with appropriate sizes, as well
    // as casts that can alter the value, e.g., AddrSpaceCasts.
    if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      return isGEPKnownNonNull(GEP, Depth, Q);

    if (auto *BCO = dyn_cast<BitCastOperator>(V))
      return isKnownNonZero(BCO->getOperand(0), Depth, Q);

    if (auto *I2P = dyn_cast<IntToPtrInst>(V))
      if (Q.DL.getTypeSizeInBits(I2P->getSrcTy()).getFixedSize() <=
          Q.DL.getTypeSizeInBits(I2P->getDestTy()).getFixedSize())
        return isKnownNonZero(I2P->getOperand(0), Depth, Q);
  }

  // Similar to int2ptr above, we can look through ptr2int here if the cast
  // is a no-op or an extend and not a truncate.
  if (auto *P2I = dyn_cast<PtrToIntInst>(V))
    if (Q.DL.getTypeSizeInBits(P2I->getSrcTy()).getFixedSize() <=
        Q.DL.getTypeSizeInBits(P2I->getDestTy()).getFixedSize())
      return isKnownNonZero(P2I->getOperand(0), Depth, Q);

  unsigned BitWidth = getBitWidth(V->getType()->getScalarType(), Q.DL);

  // X | Y != 0 if X != 0 or Y != 0.
  Value *X = nullptr, *Y = nullptr;
  if (match(V, m_Or(m_Value(X), m_Value(Y))))
    return isKnownNonZero(X, DemandedElts, Depth, Q) ||
           isKnownNonZero(Y, DemandedElts, Depth, Q);

  // ext X != 0 if X != 0.
  if (isa<SExtInst>(V) || isa<ZExtInst>(V))
    return isKnownNonZero(cast<Instruction>(V)->getOperand(0), Depth, Q);

  // shl X, Y != 0 if X is odd.  Note that the value of the shift is undefined
  // if the lowest bit is shifted off the end.
  if (match(V, m_Shl(m_Value(X), m_Value(Y)))) {
    // shl nuw can't remove any non-zero bits.
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    if (Q.IIQ.hasNoUnsignedWrap(BO))
      return isKnownNonZero(X, Depth, Q);

    KnownBits Known(BitWidth);
    computeKnownBits(X, DemandedElts, Known, Depth, Q);
    if (Known.One[0])
      return true;
  }
  // shr X, Y != 0 if X is negative.  Note that the value of the shift is not
  // defined if the sign bit is shifted off the end.
  else if (match(V, m_Shr(m_Value(X), m_Value(Y)))) {
    // shr exact can only shift out zero bits.
    const PossiblyExactOperator *BO = cast<PossiblyExactOperator>(V);
    if (BO->isExact())
      return isKnownNonZero(X, Depth, Q);

    KnownBits Known = computeKnownBits(X, DemandedElts, Depth, Q);
    if (Known.isNegative())
      return true;

    // If the shifter operand is a constant, and all of the bits shifted
    // out are known to be zero, and X is known non-zero then at least one
    // non-zero bit must remain.
    if (ConstantInt *Shift = dyn_cast<ConstantInt>(Y)) {
      auto ShiftVal = Shift->getLimitedValue(BitWidth - 1);
      // Is there a known one in the portion not shifted out?
      if (Known.countMaxLeadingZeros() < BitWidth - ShiftVal)
        return true;
      // Are all the bits to be shifted out known zero?
      if (Known.countMinTrailingZeros() >= ShiftVal)
        return isKnownNonZero(X, DemandedElts, Depth, Q);
    }
  }
  // div exact can only produce a zero if the dividend is zero.
  else if (match(V, m_Exact(m_IDiv(m_Value(X), m_Value())))) {
    return isKnownNonZero(X, DemandedElts, Depth, Q);
  }
  // X + Y.
  else if (match(V, m_Add(m_Value(X), m_Value(Y)))) {
    KnownBits XKnown = computeKnownBits(X, DemandedElts, Depth, Q);
    KnownBits YKnown = computeKnownBits(Y, DemandedElts, Depth, Q);

    // If X and Y are both non-negative (as signed values) then their sum is not
    // zero unless both X and Y are zero.
    if (XKnown.isNonNegative() && YKnown.isNonNegative())
      if (isKnownNonZero(X, DemandedElts, Depth, Q) ||
          isKnownNonZero(Y, DemandedElts, Depth, Q))
        return true;

    // If X and Y are both negative (as signed values) then their sum is not
    // zero unless both X and Y equal INT_MIN.
    if (XKnown.isNegative() && YKnown.isNegative()) {
      APInt Mask = APInt::getSignedMaxValue(BitWidth);
      // The sign bit of X is set.  If some other bit is set then X is not equal
      // to INT_MIN.
      if (XKnown.One.intersects(Mask))
        return true;
      // The sign bit of Y is set.  If some other bit is set then Y is not equal
      // to INT_MIN.
      if (YKnown.One.intersects(Mask))
        return true;
    }

    // The sum of a non-negative number and a power of two is not zero.
    if (XKnown.isNonNegative() &&
        isKnownToBeAPowerOfTwo(Y, /*OrZero*/ false, Depth, Q))
      return true;
    if (YKnown.isNonNegative() &&
        isKnownToBeAPowerOfTwo(X, /*OrZero*/ false, Depth, Q))
      return true;
  }
  // X * Y.
  else if (match(V, m_Mul(m_Value(X), m_Value(Y)))) {
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    // If X and Y are non-zero then so is X * Y as long as the multiplication
    // does not overflow.
    if ((Q.IIQ.hasNoSignedWrap(BO) || Q.IIQ.hasNoUnsignedWrap(BO)) &&
        isKnownNonZero(X, DemandedElts, Depth, Q) &&
        isKnownNonZero(Y, DemandedElts, Depth, Q))
      return true;
  }
  // (C ? X : Y) != 0 if X != 0 and Y != 0.
  else if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
    if (isKnownNonZero(SI->getTrueValue(), DemandedElts, Depth, Q) &&
        isKnownNonZero(SI->getFalseValue(), DemandedElts, Depth, Q))
      return true;
  }
  // PHI
  else if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    if (Q.IIQ.UseInstrInfo && isNonZeroRecurrence(PN))
      return true;

    // Check if all incoming values are non-zero using recursion.
    Query RecQ = Q;
    unsigned NewDepth = std::max(Depth, MaxAnalysisRecursionDepth - 1);
    return llvm::all_of(PN->operands(), [&](const Use &U) {
      if (U.get() == PN)
        return true;
      RecQ.CxtI = PN->getIncomingBlock(U)->getTerminator();
      return isKnownNonZero(U.get(), DemandedElts, NewDepth, RecQ);
    });
  }
  // ExtractElement
  else if (const auto *EEI = dyn_cast<ExtractElementInst>(V)) {
    const Value *Vec = EEI->getVectorOperand();
    const Value *Idx = EEI->getIndexOperand();
    auto *CIdx = dyn_cast<ConstantInt>(Idx);
    if (auto *VecTy = dyn_cast<FixedVectorType>(Vec->getType())) {
      unsigned NumElts = VecTy->getNumElements();
      APInt DemandedVecElts = APInt::getAllOnes(NumElts);
      if (CIdx && CIdx->getValue().ult(NumElts))
        DemandedVecElts = APInt::getOneBitSet(NumElts, CIdx->getZExtValue());
      return isKnownNonZero(Vec, DemandedVecElts, Depth, Q);
    }
  }
  // Freeze
  else if (const FreezeInst *FI = dyn_cast<FreezeInst>(V)) {
    auto *Op = FI->getOperand(0);
    if (isKnownNonZero(Op, Depth, Q) &&
        isGuaranteedNotToBePoison(Op, Q.AC, Q.CxtI, Q.DT, Depth))
      return true;
  }

  KnownBits Known(BitWidth);
  computeKnownBits(V, DemandedElts, Known, Depth, Q);
  return Known.One != 0;
}

bool isKnownNonZero(const Value* V, unsigned Depth, const Query& Q) {
  // FIXME: We currently have no way to represent the DemandedElts of a scalable
  // vector
  if (isa<ScalableVectorType>(V->getType()))
    return false;

  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  return isKnownNonZero(V, DemandedElts, Depth, Q);
}

/// If the pair of operators are the same invertible function, return the
/// the operands of the function corresponding to each input. Otherwise,
/// return None.  An invertible function is one that is 1-to-1 and maps
/// every input value to exactly one output value.  This is equivalent to
/// saying that Op1 and Op2 are equal exactly when the specified pair of
/// operands are equal, (except that Op1 and Op2 may be poison more often.)
static Optional<std::pair<Value*, Value*>>
getInvertibleOperands(const Operator *Op1,
                      const Operator *Op2) {
  if (Op1->getOpcode() != Op2->getOpcode())
    return None;

  auto getOperands = [&](unsigned OpNum) -> auto {
    return std::make_pair(Op1->getOperand(OpNum), Op2->getOperand(OpNum));
  };

  switch (Op1->getOpcode()) {
  default:
    break;
  case Instruction::Add:
  case Instruction::Sub:
    if (Op1->getOperand(0) == Op2->getOperand(0))
      return getOperands(1);
    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  case Instruction::Mul: {
    // invertible if A * B == (A * B) mod 2^N where A, and B are integers
    // and N is the bitwdith.  The nsw case is non-obvious, but proven by
    // alive2: https://alive2.llvm.org/ce/z/Z6D5qK
    auto *OBO1 = cast<OverflowingBinaryOperator>(Op1);
    auto *OBO2 = cast<OverflowingBinaryOperator>(Op2);
    if ((!OBO1->hasNoUnsignedWrap() || !OBO2->hasNoUnsignedWrap()) &&
        (!OBO1->hasNoSignedWrap() || !OBO2->hasNoSignedWrap()))
      break;

    // Assume operand order has been canonicalized
    if (Op1->getOperand(1) == Op2->getOperand(1) &&
        isa<ConstantInt>(Op1->getOperand(1)) &&
        !cast<ConstantInt>(Op1->getOperand(1))->isZero())
      return getOperands(0);
    break;
  }
  case Instruction::Shl: {
    // Same as multiplies, with the difference that we don't need to check
    // for a non-zero multiply. Shifts always multiply by non-zero.
    auto *OBO1 = cast<OverflowingBinaryOperator>(Op1);
    auto *OBO2 = cast<OverflowingBinaryOperator>(Op2);
    if ((!OBO1->hasNoUnsignedWrap() || !OBO2->hasNoUnsignedWrap()) &&
        (!OBO1->hasNoSignedWrap() || !OBO2->hasNoSignedWrap()))
      break;

    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  }
  case Instruction::AShr:
  case Instruction::LShr: {
    auto *PEO1 = cast<PossiblyExactOperator>(Op1);
    auto *PEO2 = cast<PossiblyExactOperator>(Op2);
    if (!PEO1->isExact() || !PEO2->isExact())
      break;

    if (Op1->getOperand(1) == Op2->getOperand(1))
      return getOperands(0);
    break;
  }
  case Instruction::SExt:
  case Instruction::ZExt:
    if (Op1->getOperand(0)->getType() == Op2->getOperand(0)->getType())
      return getOperands(0);
    break;
  case Instruction::PHI: {
    const PHINode *PN1 = cast<PHINode>(Op1);
    const PHINode *PN2 = cast<PHINode>(Op2);

    // If PN1 and PN2 are both recurrences, can we prove the entire recurrences
    // are a single invertible function of the start values? Note that repeated
    // application of an invertible function is also invertible
    BinaryOperator *BO1 = nullptr;
    Value *Start1 = nullptr, *Step1 = nullptr;
    BinaryOperator *BO2 = nullptr;
    Value *Start2 = nullptr, *Step2 = nullptr;
    if (PN1->getParent() != PN2->getParent() ||
        !matchSimpleRecurrence(PN1, BO1, Start1, Step1) ||
        !matchSimpleRecurrence(PN2, BO2, Start2, Step2))
      break;

    auto Values = getInvertibleOperands(cast<Operator>(BO1),
                                        cast<Operator>(BO2));
    if (!Values)
       break;

    // We have to be careful of mutually defined recurrences here.  Ex:
    // * X_i = X_(i-1) OP Y_(i-1), and Y_i = X_(i-1) OP V
    // * X_i = Y_i = X_(i-1) OP Y_(i-1)
    // The invertibility of these is complicated, and not worth reasoning
    // about (yet?).
    if (Values->first != PN1 || Values->second != PN2)
      break;

    return std::make_pair(Start1, Start2);
  }
  }
  return None;
}

/// Return true if V2 == V1 + X, where X is known non-zero.
static bool isAddOfNonZero(const Value *V1, const Value *V2, unsigned Depth,
                           const Query &Q) {
  const BinaryOperator *BO = dyn_cast<BinaryOperator>(V1);
  if (!BO || BO->getOpcode() != Instruction::Add)
    return false;
  Value *Op = nullptr;
  if (V2 == BO->getOperand(0))
    Op = BO->getOperand(1);
  else if (V2 == BO->getOperand(1))
    Op = BO->getOperand(0);
  else
    return false;
  return isKnownNonZero(Op, Depth + 1, Q);
}

/// Return true if V2 == V1 * C, where V1 is known non-zero, C is not 0/1 and
/// the multiplication is nuw or nsw.
static bool isNonEqualMul(const Value *V1, const Value *V2, unsigned Depth,
                          const Query &Q) {
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(V2)) {
    const APInt *C;
    return match(OBO, m_Mul(m_Specific(V1), m_APInt(C))) &&
           (OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap()) &&
           !C->isZero() && !C->isOne() && isKnownNonZero(V1, Depth + 1, Q);
  }
  return false;
}

/// Return true if V2 == V1 << C, where V1 is known non-zero, C is not 0 and
/// the shift is nuw or nsw.
static bool isNonEqualShl(const Value *V1, const Value *V2, unsigned Depth,
                          const Query &Q) {
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(V2)) {
    const APInt *C;
    return match(OBO, m_Shl(m_Specific(V1), m_APInt(C))) &&
           (OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap()) &&
           !C->isZero() && isKnownNonZero(V1, Depth + 1, Q);
  }
  return false;
}

static bool isNonEqualPHIs(const PHINode *PN1, const PHINode *PN2,
                           unsigned Depth, const Query &Q) {
  // Check two PHIs are in same block.
  if (PN1->getParent() != PN2->getParent())
    return false;

  SmallPtrSet<const BasicBlock *, 8> VisitedBBs;
  bool UsedFullRecursion = false;
  for (const BasicBlock *IncomBB : PN1->blocks()) {
    if (!VisitedBBs.insert(IncomBB).second)
      continue; // Don't reprocess blocks that we have dealt with already.
    const Value *IV1 = PN1->getIncomingValueForBlock(IncomBB);
    const Value *IV2 = PN2->getIncomingValueForBlock(IncomBB);
    const APInt *C1, *C2;
    if (match(IV1, m_APInt(C1)) && match(IV2, m_APInt(C2)) && *C1 != *C2)
      continue;

    // Only one pair of phi operands is allowed for full recursion.
    if (UsedFullRecursion)
      return false;

    Query RecQ = Q;
    RecQ.CxtI = IncomBB->getTerminator();
    if (!isKnownNonEqual(IV1, IV2, Depth + 1, RecQ))
      return false;
    UsedFullRecursion = true;
  }
  return true;
}

/// Return true if it is known that V1 != V2.
static bool isKnownNonEqual(const Value *V1, const Value *V2, unsigned Depth,
                            const Query &Q) {
  if (V1 == V2)
    return false;
  if (V1->getType() != V2->getType())
    // We can't look through casts yet.
    return false;

  if (Depth >= MaxAnalysisRecursionDepth)
    return false;

  // See if we can recurse through (exactly one of) our operands.  This
  // requires our operation be 1-to-1 and map every input value to exactly
  // one output value.  Such an operation is invertible.
  auto *O1 = dyn_cast<Operator>(V1);
  auto *O2 = dyn_cast<Operator>(V2);
  if (O1 && O2 && O1->getOpcode() == O2->getOpcode()) {
    if (auto Values = getInvertibleOperands(O1, O2))
      return isKnownNonEqual(Values->first, Values->second, Depth + 1, Q);

    if (const PHINode *PN1 = dyn_cast<PHINode>(V1)) {
      const PHINode *PN2 = cast<PHINode>(V2);
      // FIXME: This is missing a generalization to handle the case where one is
      // a PHI and another one isn't.
      if (isNonEqualPHIs(PN1, PN2, Depth, Q))
        return true;
    };
  }

  if (isAddOfNonZero(V1, V2, Depth, Q) || isAddOfNonZero(V2, V1, Depth, Q))
    return true;

  if (isNonEqualMul(V1, V2, Depth, Q) || isNonEqualMul(V2, V1, Depth, Q))
    return true;

  if (isNonEqualShl(V1, V2, Depth, Q) || isNonEqualShl(V2, V1, Depth, Q))
    return true;

  if (V1->getType()->isIntOrIntVectorTy()) {
    // Are any known bits in V1 contradictory to known bits in V2? If V1
    // has a known zero where V2 has a known one, they must not be equal.
    KnownBits Known1 = computeKnownBits(V1, Depth, Q);
    KnownBits Known2 = computeKnownBits(V2, Depth, Q);

    if (Known1.Zero.intersects(Known2.One) ||
        Known2.Zero.intersects(Known1.One))
      return true;
  }
  return false;
}

/// Return true if 'V & Mask' is known to be zero.  We use this predicate to
/// simplify operations downstream. Mask is known to be zero for bits that V
/// cannot have.
///
/// This function is defined on values with integer type, values with pointer
/// type, and vectors of integers.  In the case
/// where V is a vector, the mask, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the elements in the vector.
bool MaskedValueIsZero(const Value *V, const APInt &Mask, unsigned Depth,
                       const Query &Q) {
  KnownBits Known(Mask.getBitWidth());
  computeKnownBits(V, Known, Depth, Q);
  return Mask.isSubsetOf(Known.Zero);
}

// Match a signed min+max clamp pattern like smax(smin(In, CHigh), CLow).
// Returns the input and lower/upper bounds.
static bool isSignedMinMaxClamp(const Value *Select, const Value *&In,
                                const APInt *&CLow, const APInt *&CHigh) {
  assert(isa<Operator>(Select) &&
         cast<Operator>(Select)->getOpcode() == Instruction::Select &&
         "Input should be a Select!");

  const Value *LHS = nullptr, *RHS = nullptr;
  SelectPatternFlavor SPF = matchSelectPattern(Select, LHS, RHS).Flavor;
  if (SPF != SPF_SMAX && SPF != SPF_SMIN)
    return false;

  if (!match(RHS, m_APInt(CLow)))
    return false;

  const Value *LHS2 = nullptr, *RHS2 = nullptr;
  SelectPatternFlavor SPF2 = matchSelectPattern(LHS, LHS2, RHS2).Flavor;
  if (getInverseMinMaxFlavor(SPF) != SPF2)
    return false;

  if (!match(RHS2, m_APInt(CHigh)))
    return false;

  if (SPF == SPF_SMIN)
    std::swap(CLow, CHigh);

  In = LHS2;
  return CLow->sle(*CHigh);
}

static bool isSignedMinMaxIntrinsicClamp(const IntrinsicInst *II,
                                         const APInt *&CLow,
                                         const APInt *&CHigh) {
  assert((II->getIntrinsicID() == Intrinsic::smin ||
          II->getIntrinsicID() == Intrinsic::smax) && "Must be smin/smax");

  Intrinsic::ID InverseID = getInverseMinMaxIntrinsic(II->getIntrinsicID());
  auto *InnerII = dyn_cast<IntrinsicInst>(II->getArgOperand(0));
  if (!InnerII || InnerII->getIntrinsicID() != InverseID ||
      !match(II->getArgOperand(1), m_APInt(CLow)) ||
      !match(InnerII->getArgOperand(1), m_APInt(CHigh)))
    return false;

  if (II->getIntrinsicID() == Intrinsic::smin)
    std::swap(CLow, CHigh);
  return CLow->sle(*CHigh);
}

/// For vector constants, loop over the elements and find the constant with the
/// minimum number of sign bits. Return 0 if the value is not a vector constant
/// or if any element was not analyzed; otherwise, return the count for the
/// element with the minimum number of sign bits.
static unsigned computeNumSignBitsVectorConstant(const Value *V,
                                                 const APInt &DemandedElts,
                                                 unsigned TyBits) {
  const auto *CV = dyn_cast<Constant>(V);
  if (!CV || !isa<FixedVectorType>(CV->getType()))
    return 0;

  unsigned MinSignBits = TyBits;
  unsigned NumElts = cast<FixedVectorType>(CV->getType())->getNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    if (!DemandedElts[i])
      continue;
    // If we find a non-ConstantInt, bail out.
    auto *Elt = dyn_cast_or_null<ConstantInt>(CV->getAggregateElement(i));
    if (!Elt)
      return 0;

    MinSignBits = std::min(MinSignBits, Elt->getValue().getNumSignBits());
  }

  return MinSignBits;
}

static unsigned ComputeNumSignBitsImpl(const Value *V,
                                       const APInt &DemandedElts,
                                       unsigned Depth, const Query &Q);

static unsigned ComputeNumSignBits(const Value *V, const APInt &DemandedElts,
                                   unsigned Depth, const Query &Q) {
  unsigned Result = ComputeNumSignBitsImpl(V, DemandedElts, Depth, Q);
  assert(Result > 0 && "At least one sign bit needs to be present!");
  return Result;
}

/// Return the number of times the sign bit of the register is replicated into
/// the other bits. We know that at least 1 bit is always equal to the sign bit
/// (itself), but other cases can give us information. For example, immediately
/// after an "ashr X, 2", we know that the top 3 bits are all equal to each
/// other, so we return 3. For vectors, return the number of sign bits for the
/// vector element with the minimum number of known sign bits of the demanded
/// elements in the vector specified by DemandedElts.
static unsigned ComputeNumSignBitsImpl(const Value *V,
                                       const APInt &DemandedElts,
                                       unsigned Depth, const Query &Q) {
  Type *Ty = V->getType();

  // FIXME: We currently have no way to represent the DemandedElts of a scalable
  // vector
  if (isa<ScalableVectorType>(Ty))
    return 1;

#ifndef NDEBUG
  assert(Depth <= MaxAnalysisRecursionDepth && "Limit Search Depth");

  if (auto *FVTy = dyn_cast<FixedVectorType>(Ty)) {
    assert(
        FVTy->getNumElements() == DemandedElts.getBitWidth() &&
        "DemandedElt width should equal the fixed vector number of elements");
  } else {
    assert(DemandedElts == APInt(1, 1) &&
           "DemandedElt width should be 1 for scalars");
  }
#endif

  // We return the minimum number of sign bits that are guaranteed to be present
  // in V, so for undef we have to conservatively return 1.  We don't have the
  // same behavior for poison though -- that's a FIXME today.

  Type *ScalarTy = Ty->getScalarType();
  unsigned TyBits = ScalarTy->isPointerTy() ?
    Q.DL.getPointerTypeSizeInBits(ScalarTy) :
    Q.DL.getTypeSizeInBits(ScalarTy);

  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  // Note that ConstantInt is handled by the general computeKnownBits case
  // below.

  if (Depth == MaxAnalysisRecursionDepth)
    return 1;

  if (auto *U = dyn_cast<Operator>(V)) {
    switch (Operator::getOpcode(V)) {
    default: break;
    case Instruction::SExt:
      Tmp = TyBits - U->getOperand(0)->getType()->getScalarSizeInBits();
      return ComputeNumSignBits(U->getOperand(0), Depth + 1, Q) + Tmp;

    case Instruction::SDiv: {
      const APInt *Denominator;
      // sdiv X, C -> adds log(C) sign bits.
      if (match(U->getOperand(1), m_APInt(Denominator))) {

        // Ignore non-positive denominator.
        if (!Denominator->isStrictlyPositive())
          break;

        // Calculate the incoming numerator bits.
        unsigned NumBits = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);

        // Add floor(log(C)) bits to the numerator bits.
        return std::min(TyBits, NumBits + Denominator->logBase2());
      }
      break;
    }

    case Instruction::SRem: {
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);

      const APInt *Denominator;
      // srem X, C -> we know that the result is within [-C+1,C) when C is a
      // positive constant.  This let us put a lower bound on the number of sign
      // bits.
      if (match(U->getOperand(1), m_APInt(Denominator))) {

        // Ignore non-positive denominator.
        if (Denominator->isStrictlyPositive()) {
          // Calculate the leading sign bit constraints by examining the
          // denominator.  Given that the denominator is positive, there are two
          // cases:
          //
          //  1. The numerator is positive. The result range is [0,C) and
          //     [0,C) u< (1 << ceilLogBase2(C)).
          //
          //  2. The numerator is negative. Then the result range is (-C,0] and
          //     integers in (-C,0] are either 0 or >u (-1 << ceilLogBase2(C)).
          //
          // Thus a lower bound on the number of sign bits is `TyBits -
          // ceilLogBase2(C)`.

          unsigned ResBits = TyBits - Denominator->ceilLogBase2();
          Tmp = std::max(Tmp, ResBits);
        }
      }
      return Tmp;
    }

    case Instruction::AShr: {
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
      // ashr X, C   -> adds C sign bits.  Vectors too.
      const APInt *ShAmt;
      if (match(U->getOperand(1), m_APInt(ShAmt))) {
        if (ShAmt->uge(TyBits))
          break; // Bad shift.
        unsigned ShAmtLimited = ShAmt->getZExtValue();
        Tmp += ShAmtLimited;
        if (Tmp > TyBits) Tmp = TyBits;
      }
      return Tmp;
    }
    case Instruction::Shl: {
      const APInt *ShAmt;
      if (match(U->getOperand(1), m_APInt(ShAmt))) {
        // shl destroys sign bits.
        Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
        if (ShAmt->uge(TyBits) ||   // Bad shift.
            ShAmt->uge(Tmp)) break; // Shifted all sign bits out.
        Tmp2 = ShAmt->getZExtValue();
        return Tmp - Tmp2;
      }
      break;
    }
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: // NOT is handled here.
      // Logical binary ops preserve the number of sign bits at the worst.
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
      if (Tmp != 1) {
        Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
        FirstAnswer = std::min(Tmp, Tmp2);
        // We computed what we know about the sign bits as our first
        // answer. Now proceed to the generic code that uses
        // computeKnownBits, and pick whichever answer is better.
      }
      break;

    case Instruction::Select: {
      // If we have a clamp pattern, we know that the number of sign bits will
      // be the minimum of the clamp min/max range.
      const Value *X;
      const APInt *CLow, *CHigh;
      if (isSignedMinMaxClamp(U, X, CLow, CHigh))
        return std::min(CLow->getNumSignBits(), CHigh->getNumSignBits());

      Tmp = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
      if (Tmp == 1) break;
      Tmp2 = ComputeNumSignBits(U->getOperand(2), Depth + 1, Q);
      return std::min(Tmp, Tmp2);
    }

    case Instruction::Add:
      // Add can have at most one carry bit.  Thus we know that the output
      // is, at worst, one more bit than the inputs.
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
      if (Tmp == 1) break;

      // Special case decrementing a value (ADD X, -1):
      if (const auto *CRHS = dyn_cast<Constant>(U->getOperand(1)))
        if (CRHS->isAllOnesValue()) {
          KnownBits Known(TyBits);
          computeKnownBits(U->getOperand(0), Known, Depth + 1, Q);

          // If the input is known to be 0 or 1, the output is 0/-1, which is
          // all sign bits set.
          if ((Known.Zero | 1).isAllOnes())
            return TyBits;

          // If we are subtracting one from a positive number, there is no carry
          // out of the result.
          if (Known.isNonNegative())
            return Tmp;
        }

      Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
      if (Tmp2 == 1) break;
      return std::min(Tmp, Tmp2) - 1;

    case Instruction::Sub:
      Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
      if (Tmp2 == 1) break;

      // Handle NEG.
      if (const auto *CLHS = dyn_cast<Constant>(U->getOperand(0)))
        if (CLHS->isNullValue()) {
          KnownBits Known(TyBits);
          computeKnownBits(U->getOperand(1), Known, Depth + 1, Q);
          // If the input is known to be 0 or 1, the output is 0/-1, which is
          // all sign bits set.
          if ((Known.Zero | 1).isAllOnes())
            return TyBits;

          // If the input is known to be positive (the sign bit is known clear),
          // the output of the NEG has the same number of sign bits as the
          // input.
          if (Known.isNonNegative())
            return Tmp2;

          // Otherwise, we treat this like a SUB.
        }

      // Sub can have at most one carry bit.  Thus we know that the output
      // is, at worst, one more bit than the inputs.
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
      if (Tmp == 1) break;
      return std::min(Tmp, Tmp2) - 1;

    case Instruction::Mul: {
      // The output of the Mul can be at most twice the valid bits in the
      // inputs.
      unsigned SignBitsOp0 = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
      if (SignBitsOp0 == 1) break;
      unsigned SignBitsOp1 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
      if (SignBitsOp1 == 1) break;
      unsigned OutValidBits =
          (TyBits - SignBitsOp0 + 1) + (TyBits - SignBitsOp1 + 1);
      return OutValidBits > TyBits ? 1 : TyBits - OutValidBits + 1;
    }

    case Instruction::PHI: {
      const PHINode *PN = cast<PHINode>(U);
      unsigned NumIncomingValues = PN->getNumIncomingValues();
      // Don't analyze large in-degree PHIs.
      if (NumIncomingValues > 4) break;
      // Unreachable blocks may have zero-operand PHI nodes.
      if (NumIncomingValues == 0) break;

      // Take the minimum of all incoming values.  This can't infinitely loop
      // because of our depth threshold.
      Query RecQ = Q;
      Tmp = TyBits;
      for (unsigned i = 0, e = NumIncomingValues; i != e; ++i) {
        if (Tmp == 1) return Tmp;
        RecQ.CxtI = PN->getIncomingBlock(i)->getTerminator();
        Tmp = std::min(
            Tmp, ComputeNumSignBits(PN->getIncomingValue(i), Depth + 1, RecQ));
      }
      return Tmp;
    }

    case Instruction::Trunc:
      // FIXME: it's tricky to do anything useful for this, but it is an
      // important case for targets like X86.
      break;

    case Instruction::ExtractElement:
      // Look through extract element. At the moment we keep this simple and
      // skip tracking the specific element. But at least we might find
      // information valid for all elements of the vector (for example if vector
      // is sign extended, shifted, etc).
      return ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);

    case Instruction::ShuffleVector: {
      // Collect the minimum number of sign bits that are shared by every vector
      // element referenced by the shuffle.
      auto *Shuf = dyn_cast<ShuffleVectorInst>(U);
      if (!Shuf) {
        // FIXME: Add support for shufflevector constant expressions.
        return 1;
      }
      APInt DemandedLHS, DemandedRHS;
      // For undef elements, we don't know anything about the common state of
      // the shuffle result.
      if (!getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS, DemandedRHS))
        return 1;
      Tmp = std::numeric_limits<unsigned>::max();
      if (!!DemandedLHS) {
        const Value *LHS = Shuf->getOperand(0);
        Tmp = ComputeNumSignBits(LHS, DemandedLHS, Depth + 1, Q);
      }
      // If we don't know anything, early out and try computeKnownBits
      // fall-back.
      if (Tmp == 1)
        break;
      if (!!DemandedRHS) {
        const Value *RHS = Shuf->getOperand(1);
        Tmp2 = ComputeNumSignBits(RHS, DemandedRHS, Depth + 1, Q);
        Tmp = std::min(Tmp, Tmp2);
      }
      // If we don't know anything, early out and try computeKnownBits
      // fall-back.
      if (Tmp == 1)
        break;
      assert(Tmp <= TyBits && "Failed to determine minimum sign bits");
      return Tmp;
    }
    case Instruction::Call: {
      if (const auto *II = dyn_cast<IntrinsicInst>(U)) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::abs:
          Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
          if (Tmp == 1) break;

          // Absolute value reduces number of sign bits by at most 1.
          return Tmp - 1;
        case Intrinsic::smin:
        case Intrinsic::smax: {
          const APInt *CLow, *CHigh;
          if (isSignedMinMaxIntrinsicClamp(II, CLow, CHigh))
            return std::min(CLow->getNumSignBits(), CHigh->getNumSignBits());
        }
        }
      }
    }
    }
  }

  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.

  // If we can examine all elements of a vector constant successfully, we're
  // done (we can't do any better than that). If not, keep trying.
  if (unsigned VecSignBits =
          computeNumSignBitsVectorConstant(V, DemandedElts, TyBits))
    return VecSignBits;

  KnownBits Known(TyBits);
  computeKnownBits(V, DemandedElts, Known, Depth, Q);

  // If we know that the sign bit is either zero or one, determine the number of
  // identical bits in the top of the input value.
  return std::max(FirstAnswer, Known.countMinSignBits());
}

Intrinsic::ID llvm::getIntrinsicForCallSite(const CallBase &CB,
                                            const TargetLibraryInfo *TLI) {
  const Function *F = CB.getCalledFunction();
  if (!F)
    return Intrinsic::not_intrinsic;

  if (F->isIntrinsic())
    return F->getIntrinsicID();

  // We are going to infer semantics of a library function based on mapping it
  // to an LLVM intrinsic. Check that the library function is available from
  // this callbase and in this environment.
  LibFunc Func;
  if (F->hasLocalLinkage() || !TLI || !TLI->getLibFunc(CB, Func) ||
      !CB.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  switch (Func) {
  default:
    break;
  case LibFunc_sin:
  case LibFunc_sinf:
  case LibFunc_sinl:
    return Intrinsic::sin;
  case LibFunc_cos:
  case LibFunc_cosf:
  case LibFunc_cosl:
    return Intrinsic::cos;
  case LibFunc_exp:
  case LibFunc_expf:
  case LibFunc_expl:
    return Intrinsic::exp;
  case LibFunc_exp2:
  case LibFunc_exp2f:
  case LibFunc_exp2l:
    return Intrinsic::exp2;
  case LibFunc_log:
  case LibFunc_logf:
  case LibFunc_logl:
    return Intrinsic::log;
  case LibFunc_log10:
  case LibFunc_log10f:
  case LibFunc_log10l:
    return Intrinsic::log10;
  case LibFunc_log2:
  case LibFunc_log2f:
  case LibFunc_log2l:
    return Intrinsic::log2;
  case LibFunc_fabs:
  case LibFunc_fabsf:
  case LibFunc_fabsl:
    return Intrinsic::fabs;
  case LibFunc_fmin:
  case LibFunc_fminf:
  case LibFunc_fminl:
    return Intrinsic::minnum;
  case LibFunc_fmax:
  case LibFunc_fmaxf:
  case LibFunc_fmaxl:
    return Intrinsic::maxnum;
  case LibFunc_copysign:
  case LibFunc_copysignf:
  case LibFunc_copysignl:
    return Intrinsic::copysign;
  case LibFunc_floor:
  case LibFunc_floorf:
  case LibFunc_floorl:
    return Intrinsic::floor;
  case LibFunc_ceil:
  case LibFunc_ceilf:
  case LibFunc_ceill:
    return Intrinsic::ceil;
  case LibFunc_trunc:
  case LibFunc_truncf:
  case LibFunc_truncl:
    return Intrinsic::trunc;
  case LibFunc_rint:
  case LibFunc_rintf:
  case LibFunc_rintl:
    return Intrinsic::rint;
  case LibFunc_nearbyint:
  case LibFunc_nearbyintf:
  case LibFunc_nearbyintl:
    return Intrinsic::nearbyint;
  case LibFunc_round:
  case LibFunc_roundf:
  case LibFunc_roundl:
    return Intrinsic::round;
  case LibFunc_roundeven:
  case LibFunc_roundevenf:
  case LibFunc_roundevenl:
    return Intrinsic::roundeven;
  case LibFunc_pow:
  case LibFunc_powf:
  case LibFunc_powl:
    return Intrinsic::pow;
  case LibFunc_sqrt:
  case LibFunc_sqrtf:
  case LibFunc_sqrtl:
    return Intrinsic::sqrt;
  }

  return Intrinsic::not_intrinsic;
}

/// Return true if we can prove that the specified FP value is never equal to
/// -0.0.
/// NOTE: Do not check 'nsz' here because that fast-math-flag does not guarantee
///       that a value is not -0.0. It only guarantees that -0.0 may be treated
///       the same as +0.0 in floating-point ops.
bool llvm::CannotBeNegativeZero(const Value *V, const TargetLibraryInfo *TLI,
                                unsigned Depth) {
  if (auto *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->getValueAPF().isNegZero();

  if (Depth == MaxAnalysisRecursionDepth)
    return false;

  auto *Op = dyn_cast<Operator>(V);
  if (!Op)
    return false;

  // (fadd x, 0.0) is guaranteed to return +0.0, not -0.0.
  if (match(Op, m_FAdd(m_Value(), m_PosZeroFP())))
    return true;

  // sitofp and uitofp turn into +0.0 for zero.
  if (isa<SIToFPInst>(Op) || isa<UIToFPInst>(Op))
    return true;

  if (auto *Call = dyn_cast<CallInst>(Op)) {
    Intrinsic::ID IID = getIntrinsicForCallSite(*Call, TLI);
    switch (IID) {
    default:
      break;
    // sqrt(-0.0) = -0.0, no other negative results are possible.
    case Intrinsic::sqrt:
    case Intrinsic::canonicalize:
      return CannotBeNegativeZero(Call->getArgOperand(0), TLI, Depth + 1);
    case Intrinsic::experimental_constrained_sqrt: {
      // NOTE: This rounding mode restriction may be too strict.
      const auto *CI = cast<ConstrainedFPIntrinsic>(Call);
      if (CI->getRoundingMode() == RoundingMode::NearestTiesToEven)
        return CannotBeNegativeZero(Call->getArgOperand(0), TLI, Depth + 1);
      else
        return false;
    }
    // fabs(x) != -0.0
    case Intrinsic::fabs:
      return true;
    // sitofp and uitofp turn into +0.0 for zero.
    case Intrinsic::experimental_constrained_sitofp:
    case Intrinsic::experimental_constrained_uitofp:
      return true;
    }
  }

  return false;
}

/// If \p SignBitOnly is true, test for a known 0 sign bit rather than a
/// standard ordered compare. e.g. make -0.0 olt 0.0 be true because of the sign
/// bit despite comparing equal.
static bool cannotBeOrderedLessThanZeroImpl(const Value *V,
                                            const TargetLibraryInfo *TLI,
                                            bool SignBitOnly,
                                            unsigned Depth) {
  // TODO: This function does not do the right thing when SignBitOnly is true
  // and we're lowering to a hypothetical IEEE 754-compliant-but-evil platform
  // which flips the sign bits of NaNs.  See
  // https://llvm.org/bugs/show_bug.cgi?id=31702.

  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    return !CFP->getValueAPF().isNegative() ||
           (!SignBitOnly && CFP->getValueAPF().isZero());
  }

  // Handle vector of constants.
  if (auto *CV = dyn_cast<Constant>(V)) {
    if (auto *CVFVTy = dyn_cast<FixedVectorType>(CV->getType())) {
      unsigned NumElts = CVFVTy->getNumElements();
      for (unsigned i = 0; i != NumElts; ++i) {
        auto *CFP = dyn_cast_or_null<ConstantFP>(CV->getAggregateElement(i));
        if (!CFP)
          return false;
        if (CFP->getValueAPF().isNegative() &&
            (SignBitOnly || !CFP->getValueAPF().isZero()))
          return false;
      }

      // All non-negative ConstantFPs.
      return true;
    }
  }

  if (Depth == MaxAnalysisRecursionDepth)
    return false;

  const Operator *I = dyn_cast<Operator>(V);
  if (!I)
    return false;

  switch (I->getOpcode()) {
  default:
    break;
  // Unsigned integers are always nonnegative.
  case Instruction::UIToFP:
    return true;
  case Instruction::FMul:
  case Instruction::FDiv:
    // X * X is always non-negative or a NaN.
    // X / X is always exactly 1.0 or a NaN.
    if (I->getOperand(0) == I->getOperand(1) &&
        (!SignBitOnly || cast<FPMathOperator>(I)->hasNoNaNs()))
      return true;

    LLVM_FALLTHROUGH;
  case Instruction::FAdd:
  case Instruction::FRem:
    return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                           Depth + 1) &&
           cannotBeOrderedLessThanZeroImpl(I->getOperand(1), TLI, SignBitOnly,
                                           Depth + 1);
  case Instruction::Select:
    return cannotBeOrderedLessThanZeroImpl(I->getOperand(1), TLI, SignBitOnly,
                                           Depth + 1) &&
           cannotBeOrderedLessThanZeroImpl(I->getOperand(2), TLI, SignBitOnly,
                                           Depth + 1);
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    // Widening/narrowing never change sign.
    return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                           Depth + 1);
  case Instruction::ExtractElement:
    // Look through extract element. At the moment we keep this simple and skip
    // tracking the specific element. But at least we might find information
    // valid for all elements of the vector.
    return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                           Depth + 1);
  case Instruction::Call:
    const auto *CI = cast<CallInst>(I);
    Intrinsic::ID IID = getIntrinsicForCallSite(*CI, TLI);
    switch (IID) {
    default:
      break;
    case Intrinsic::maxnum: {
      Value *V0 = I->getOperand(0), *V1 = I->getOperand(1);
      auto isPositiveNum = [&](Value *V) {
        if (SignBitOnly) {
          // With SignBitOnly, this is tricky because the result of
          // maxnum(+0.0, -0.0) is unspecified. Just check if the operand is
          // a constant strictly greater than 0.0.
          const APFloat *C;
          return match(V, m_APFloat(C)) &&
                 *C > APFloat::getZero(C->getSemantics());
        }

        // -0.0 compares equal to 0.0, so if this operand is at least -0.0,
        // maxnum can't be ordered-less-than-zero.
        return isKnownNeverNaN(V, TLI) &&
               cannotBeOrderedLessThanZeroImpl(V, TLI, false, Depth + 1);
      };

      // TODO: This could be improved. We could also check that neither operand
      //       has its sign bit set (and at least 1 is not-NAN?).
      return isPositiveNum(V0) || isPositiveNum(V1);
    }

    case Intrinsic::maximum:
      return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                             Depth + 1) ||
             cannotBeOrderedLessThanZeroImpl(I->getOperand(1), TLI, SignBitOnly,
                                             Depth + 1);
    case Intrinsic::minnum:
    case Intrinsic::minimum:
      return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                             Depth + 1) &&
             cannotBeOrderedLessThanZeroImpl(I->getOperand(1), TLI, SignBitOnly,
                                             Depth + 1);
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::fabs:
      return true;

    case Intrinsic::sqrt:
      // sqrt(x) is always >= -0 or NaN.  Moreover, sqrt(x) == -0 iff x == -0.
      if (!SignBitOnly)
        return true;
      return CI->hasNoNaNs() && (CI->hasNoSignedZeros() ||
                                 CannotBeNegativeZero(CI->getOperand(0), TLI));

    case Intrinsic::powi:
      if (ConstantInt *Exponent = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // powi(x,n) is non-negative if n is even.
        if (Exponent->getBitWidth() <= 64 && Exponent->getSExtValue() % 2u == 0)
          return true;
      }
      // TODO: This is not correct.  Given that exp is an integer, here are the
      // ways that pow can return a negative value:
      //
      //   pow(x, exp)    --> negative if exp is odd and x is negative.
      //   pow(-0, exp)   --> -inf if exp is negative odd.
      //   pow(-0, exp)   --> -0 if exp is positive odd.
      //   pow(-inf, exp) --> -0 if exp is negative odd.
      //   pow(-inf, exp) --> -inf if exp is positive odd.
      //
      // Therefore, if !SignBitOnly, we can return true if x >= +0 or x is NaN,
      // but we must return false if x == -0.  Unfortunately we do not currently
      // have a way of expressing this constraint.  See details in
      // https://llvm.org/bugs/show_bug.cgi?id=31702.
      return cannotBeOrderedLessThanZeroImpl(I->getOperand(0), TLI, SignBitOnly,
                                             Depth + 1);

    case Intrinsic::fma:
    case Intrinsic::fmuladd:
      // x*x+y is non-negative if y is non-negative.
      return I->getOperand(0) == I->getOperand(1) &&
             (!SignBitOnly || cast<FPMathOperator>(I)->hasNoNaNs()) &&
             cannotBeOrderedLessThanZeroImpl(I->getOperand(2), TLI, SignBitOnly,
                                             Depth + 1);
    }
    break;
  }
  return false;
}

bool llvm::CannotBeOrderedLessThanZero(const Value *V,
                                       const TargetLibraryInfo *TLI) {
  return cannotBeOrderedLessThanZeroImpl(V, TLI, false, 0);
}

bool llvm::SignBitMustBeZero(const Value *V, const TargetLibraryInfo *TLI) {
  return cannotBeOrderedLessThanZeroImpl(V, TLI, true, 0);
}

bool llvm::isKnownNeverInfinity(const Value *V, const TargetLibraryInfo *TLI,
                                unsigned Depth) {
  assert(V->getType()->isFPOrFPVectorTy() && "Querying for Inf on non-FP type");

  // If we're told that infinities won't happen, assume they won't.
  if (auto *FPMathOp = dyn_cast<FPMathOperator>(V))
    if (FPMathOp->hasNoInfs())
      return true;

  // Handle scalar constants.
  if (auto *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->isInfinity();

  if (Depth == MaxAnalysisRecursionDepth)
    return false;

  if (auto *Inst = dyn_cast<Instruction>(V)) {
    switch (Inst->getOpcode()) {
    case Instruction::Select: {
      return isKnownNeverInfinity(Inst->getOperand(1), TLI, Depth + 1) &&
             isKnownNeverInfinity(Inst->getOperand(2), TLI, Depth + 1);
    }
    case Instruction::SIToFP:
    case Instruction::UIToFP: {
      // Get width of largest magnitude integer (remove a bit if signed).
      // This still works for a signed minimum value because the largest FP
      // value is scaled by some fraction close to 2.0 (1.0 + 0.xxxx).
      int IntSize = Inst->getOperand(0)->getType()->getScalarSizeInBits();
      if (Inst->getOpcode() == Instruction::SIToFP)
        --IntSize;

      // If the exponent of the largest finite FP value can hold the largest
      // integer, the result of the cast must be finite.
      Type *FPTy = Inst->getType()->getScalarType();
      return ilogb(APFloat::getLargest(FPTy->getFltSemantics())) >= IntSize;
    }
    default:
      break;
    }
  }

  // try to handle fixed width vector constants
  auto *VFVTy = dyn_cast<FixedVectorType>(V->getType());
  if (VFVTy && isa<Constant>(V)) {
    // For vectors, verify that each element is not infinity.
    unsigned NumElts = VFVTy->getNumElements();
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *Elt = cast<Constant>(V)->getAggregateElement(i);
      if (!Elt)
        return false;
      if (isa<UndefValue>(Elt))
        continue;
      auto *CElt = dyn_cast<ConstantFP>(Elt);
      if (!CElt || CElt->isInfinity())
        return false;
    }
    // All elements were confirmed non-infinity or undefined.
    return true;
  }

  // was not able to prove that V never contains infinity
  return false;
}

bool llvm::isKnownNeverNaN(const Value *V, const TargetLibraryInfo *TLI,
                           unsigned Depth) {
  assert(V->getType()->isFPOrFPVectorTy() && "Querying for NaN on non-FP type");

  // If we're told that NaNs won't happen, assume they won't.
  if (auto *FPMathOp = dyn_cast<FPMathOperator>(V))
    if (FPMathOp->hasNoNaNs())
      return true;

  // Handle scalar constants.
  if (auto *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->isNaN();

  if (Depth == MaxAnalysisRecursionDepth)
    return false;

  if (auto *Inst = dyn_cast<Instruction>(V)) {
    switch (Inst->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
      // Adding positive and negative infinity produces NaN.
      return isKnownNeverNaN(Inst->getOperand(0), TLI, Depth + 1) &&
             isKnownNeverNaN(Inst->getOperand(1), TLI, Depth + 1) &&
             (isKnownNeverInfinity(Inst->getOperand(0), TLI, Depth + 1) ||
              isKnownNeverInfinity(Inst->getOperand(1), TLI, Depth + 1));

    case Instruction::FMul:
      // Zero multiplied with infinity produces NaN.
      // FIXME: If neither side can be zero fmul never produces NaN.
      return isKnownNeverNaN(Inst->getOperand(0), TLI, Depth + 1) &&
             isKnownNeverInfinity(Inst->getOperand(0), TLI, Depth + 1) &&
             isKnownNeverNaN(Inst->getOperand(1), TLI, Depth + 1) &&
             isKnownNeverInfinity(Inst->getOperand(1), TLI, Depth + 1);

    case Instruction::FDiv:
    case Instruction::FRem:
      // FIXME: Only 0/0, Inf/Inf, Inf REM x and x REM 0 produce NaN.
      return false;

    case Instruction::Select: {
      return isKnownNeverNaN(Inst->getOperand(1), TLI, Depth + 1) &&
             isKnownNeverNaN(Inst->getOperand(2), TLI, Depth + 1);
    }
    case Instruction::SIToFP:
    case Instruction::UIToFP:
      return true;
    case Instruction::FPTrunc:
    case Instruction::FPExt:
      return isKnownNeverNaN(Inst->getOperand(0), TLI, Depth + 1);
    default:
      break;
    }
  }

  if (const auto *II = dyn_cast<IntrinsicInst>(V)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::canonicalize:
    case Intrinsic::fabs:
    case Intrinsic::copysign:
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::trunc:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::roundeven:
      return isKnownNeverNaN(II->getArgOperand(0), TLI, Depth + 1);
    case Intrinsic::sqrt:
      return isKnownNeverNaN(II->getArgOperand(0), TLI, Depth + 1) &&
             CannotBeOrderedLessThanZero(II->getArgOperand(0), TLI);
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
      // If either operand is not NaN, the result is not NaN.
      return isKnownNeverNaN(II->getArgOperand(0), TLI, Depth + 1) ||
             isKnownNeverNaN(II->getArgOperand(1), TLI, Depth + 1);
    default:
      return false;
    }
  }

  // Try to handle fixed width vector constants
  auto *VFVTy = dyn_cast<FixedVectorType>(V->getType());
  if (VFVTy && isa<Constant>(V)) {
    // For vectors, verify that each element is not NaN.
    unsigned NumElts = VFVTy->getNumElements();
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *Elt = cast<Constant>(V)->getAggregateElement(i);
      if (!Elt)
        return false;
      if (isa<UndefValue>(Elt))
        continue;
      auto *CElt = dyn_cast<ConstantFP>(Elt);
      if (!CElt || CElt->isNaN())
        return false;
    }
    // All elements were confirmed not-NaN or undefined.
    return true;
  }

  // Was not able to prove that V never contains NaN
  return false;
}

Value *llvm::isBytewiseValue(Value *V, const DataLayout &DL) {

  // All byte-wide stores are splatable, even of arbitrary variables.
  if (V->getType()->isIntegerTy(8))
    return V;

  LLVMContext &Ctx = V->getContext();

  // Undef don't care.
  auto *UndefInt8 = UndefValue::get(Type::getInt8Ty(Ctx));
  if (isa<UndefValue>(V))
    return UndefInt8;

  // Return Undef for zero-sized type.
  if (!DL.getTypeStoreSize(V->getType()).isNonZero())
    return UndefInt8;

  Constant *C = dyn_cast<Constant>(V);
  if (!C) {
    // Conceptually, we could handle things like:
    //   %a = zext i8 %X to i16
    //   %b = shl i16 %a, 8
    //   %c = or i16 %a, %b
    // but until there is an example that actually needs this, it doesn't seem
    // worth worrying about.
    return nullptr;
  }

  // Handle 'null' ConstantArrayZero etc.
  if (C->isNullValue())
    return Constant::getNullValue(Type::getInt8Ty(Ctx));

  // Constant floating-point values can be handled as integer values if the
  // corresponding integer value is "byteable".  An important case is 0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    Type *Ty = nullptr;
    if (CFP->getType()->isHalfTy())
      Ty = Type::getInt16Ty(Ctx);
    else if (CFP->getType()->isFloatTy())
      Ty = Type::getInt32Ty(Ctx);
    else if (CFP->getType()->isDoubleTy())
      Ty = Type::getInt64Ty(Ctx);
    // Don't handle long double formats, which have strange constraints.
    return Ty ? isBytewiseValue(ConstantExpr::getBitCast(CFP, Ty), DL)
              : nullptr;
  }

  // We can handle constant integers that are multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
    if (CI->getBitWidth() % 8 == 0) {
      assert(CI->getBitWidth() > 8 && "8 bits should be handled above!");
      if (!CI->getValue().isSplat(8))
        return nullptr;
      return ConstantInt::get(Ctx, CI->getValue().trunc(8));
    }
  }

  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::IntToPtr) {
      if (auto *PtrTy = dyn_cast<PointerType>(CE->getType())) {
        unsigned BitWidth = DL.getPointerSizeInBits(PtrTy->getAddressSpace());
        return isBytewiseValue(
            ConstantExpr::getIntegerCast(CE->getOperand(0),
                                         Type::getIntNTy(Ctx, BitWidth), false),
            DL);
      }
    }
  }

  auto Merge = [&](Value *LHS, Value *RHS) -> Value * {
    if (LHS == RHS)
      return LHS;
    if (!LHS || !RHS)
      return nullptr;
    if (LHS == UndefInt8)
      return RHS;
    if (RHS == UndefInt8)
      return LHS;
    return nullptr;
  };

  if (ConstantDataSequential *CA = dyn_cast<ConstantDataSequential>(C)) {
    Value *Val = UndefInt8;
    for (unsigned I = 0, E = CA->getNumElements(); I != E; ++I)
      if (!(Val = Merge(Val, isBytewiseValue(CA->getElementAsConstant(I), DL))))
        return nullptr;
    return Val;
  }

  if (isa<ConstantAggregate>(C)) {
    Value *Val = UndefInt8;
    for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I)
      if (!(Val = Merge(Val, isBytewiseValue(C->getOperand(I), DL))))
        return nullptr;
    return Val;
  }

  // Don't try to handle the handful of other constants.
  return nullptr;
}

// This is the recursive version of BuildSubAggregate. It takes a few different
// arguments. Idxs is the index within the nested struct From that we are
// looking at now (which is of type IndexedType). IdxSkip is the number of
// indices from Idxs that should be left out when inserting into the resulting
// struct. To is the result struct built so far, new insertvalue instructions
// build on that.
static Value *BuildSubAggregate(Value *From, Value* To, Type *IndexedType,
                                SmallVectorImpl<unsigned> &Idxs,
                                unsigned IdxSkip,
                                Instruction *InsertBefore) {
  StructType *STy = dyn_cast<StructType>(IndexedType);
  if (STy) {
    // Save the original To argument so we can modify it
    Value *OrigTo = To;
    // General case, the type indexed by Idxs is a struct
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      // Process each struct element recursively
      Idxs.push_back(i);
      Value *PrevTo = To;
      To = BuildSubAggregate(From, To, STy->getElementType(i), Idxs, IdxSkip,
                             InsertBefore);
      Idxs.pop_back();
      if (!To) {
        // Couldn't find any inserted value for this index? Cleanup
        while (PrevTo != OrigTo) {
          InsertValueInst* Del = cast<InsertValueInst>(PrevTo);
          PrevTo = Del->getAggregateOperand();
          Del->eraseFromParent();
        }
        // Stop processing elements
        break;
      }
    }
    // If we successfully found a value for each of our subaggregates
    if (To)
      return To;
  }
  // Base case, the type indexed by SourceIdxs is not a struct, or not all of
  // the struct's elements had a value that was inserted directly. In the latter
  // case, perhaps we can't determine each of the subelements individually, but
  // we might be able to find the complete struct somewhere.

  // Find the value that is at that particular spot
  Value *V = FindInsertedValue(From, Idxs);

  if (!V)
    return nullptr;

  // Insert the value in the new (sub) aggregate
  return InsertValueInst::Create(To, V, makeArrayRef(Idxs).slice(IdxSkip),
                                 "tmp", InsertBefore);
}

// This helper takes a nested struct and extracts a part of it (which is again a
// struct) into a new value. For example, given the struct:
// { a, { b, { c, d }, e } }
// and the indices "1, 1" this returns
// { c, d }.
//
// It does this by inserting an insertvalue for each element in the resulting
// struct, as opposed to just inserting a single struct. This will only work if
// each of the elements of the substruct are known (ie, inserted into From by an
// insertvalue instruction somewhere).
//
// All inserted insertvalue instructions are inserted before InsertBefore
static Value *BuildSubAggregate(Value *From, ArrayRef<unsigned> idx_range,
                                Instruction *InsertBefore) {
  assert(InsertBefore && "Must have someplace to insert!");
  Type *IndexedType = ExtractValueInst::getIndexedType(From->getType(),
                                                             idx_range);
  Value *To = UndefValue::get(IndexedType);
  SmallVector<unsigned, 10> Idxs(idx_range.begin(), idx_range.end());
  unsigned IdxSkip = Idxs.size();

  return BuildSubAggregate(From, To, IndexedType, Idxs, IdxSkip, InsertBefore);
}

/// Given an aggregate and a sequence of indices, see if the scalar value
/// indexed is already around as a register, for example if it was inserted
/// directly into the aggregate.
///
/// If InsertBefore is not null, this function will duplicate (modified)
/// insertvalues when a part of a nested struct is extracted.
Value *llvm::FindInsertedValue(Value *V, ArrayRef<unsigned> idx_range,
                               Instruction *InsertBefore) {
  // Nothing to index? Just return V then (this is useful at the end of our
  // recursion).
  if (idx_range.empty())
    return V;
  // We have indices, so V should have an indexable type.
  assert((V->getType()->isStructTy() || V->getType()->isArrayTy()) &&
         "Not looking at a struct or array?");
  assert(ExtractValueInst::getIndexedType(V->getType(), idx_range) &&
         "Invalid indices for type?");

  if (Constant *C = dyn_cast<Constant>(V)) {
    C = C->getAggregateElement(idx_range[0]);
    if (!C) return nullptr;
    return FindInsertedValue(C, idx_range.slice(1), InsertBefore);
  }

  if (InsertValueInst *I = dyn_cast<InsertValueInst>(V)) {
    // Loop the indices for the insertvalue instruction in parallel with the
    // requested indices
    const unsigned *req_idx = idx_range.begin();
    for (const unsigned *i = I->idx_begin(), *e = I->idx_end();
         i != e; ++i, ++req_idx) {
      if (req_idx == idx_range.end()) {
        // We can't handle this without inserting insertvalues
        if (!InsertBefore)
          return nullptr;

        // The requested index identifies a part of a nested aggregate. Handle
        // this specially. For example,
        // %A = insertvalue { i32, {i32, i32 } } undef, i32 10, 1, 0
        // %B = insertvalue { i32, {i32, i32 } } %A, i32 11, 1, 1
        // %C = extractvalue {i32, { i32, i32 } } %B, 1
        // This can be changed into
        // %A = insertvalue {i32, i32 } undef, i32 10, 0
        // %C = insertvalue {i32, i32 } %A, i32 11, 1
        // which allows the unused 0,0 element from the nested struct to be
        // removed.
        return BuildSubAggregate(V, makeArrayRef(idx_range.begin(), req_idx),
                                 InsertBefore);
      }

      // This insert value inserts something else than what we are looking for.
      // See if the (aggregate) value inserted into has the value we are
      // looking for, then.
      if (*req_idx != *i)
        return FindInsertedValue(I->getAggregateOperand(), idx_range,
                                 InsertBefore);
    }
    // If we end up here, the indices of the insertvalue match with those
    // requested (though possibly only partially). Now we recursively look at
    // the inserted value, passing any remaining indices.
    return FindInsertedValue(I->getInsertedValueOperand(),
                             makeArrayRef(req_idx, idx_range.end()),
                             InsertBefore);
  }

  if (ExtractValueInst *I = dyn_cast<ExtractValueInst>(V)) {
    // If we're extracting a value from an aggregate that was extracted from
    // something else, we can extract from that something else directly instead.
    // However, we will need to chain I's indices with the requested indices.

    // Calculate the number of indices required
    unsigned size = I->getNumIndices() + idx_range.size();
    // Allocate some space to put the new indices in
    SmallVector<unsigned, 5> Idxs;
    Idxs.reserve(size);
    // Add indices from the extract value instruction
    Idxs.append(I->idx_begin(), I->idx_end());

    // Add requested indices
    Idxs.append(idx_range.begin(), idx_range.end());

    assert(Idxs.size() == size
           && "Number of indices added not correct?");

    return FindInsertedValue(I->getAggregateOperand(), Idxs, InsertBefore);
  }
  // Otherwise, we don't know (such as, extracting from a function return value
  // or load instruction)
  return nullptr;
}

bool llvm::isGEPBasedOnPointerToString(const GEPOperator *GEP,
                                       unsigned CharSize) {
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return false;

  // Make sure the index-ee is a pointer to array of \p CharSize integers.
  // CharSize.
  ArrayType *AT = dyn_cast<ArrayType>(GEP->getSourceElementType());
  if (!AT || !AT->getElementType()->isIntegerTy(CharSize))
    return false;

  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  const ConstantInt *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!FirstIdx || !FirstIdx->isZero())
    return false;

  return true;
}

// If V refers to an initialized global constant, set Slice either to
// its initializer if the size of its elements equals ElementSize, or,
// for ElementSize == 8, to its representation as an array of unsiged
// char. Return true on success.
bool llvm::getConstantDataArrayInfo(const Value *V,
                                    ConstantDataArraySlice &Slice,
                                    unsigned ElementSize, uint64_t Offset) {
  assert(V);

  // Look through bitcast instructions and geps.
  V = V->stripPointerCasts();

  // If the value is a GEP instruction or constant expression, treat it as an
  // offset.
  if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    // Fail if the first GEP operand is not a constant zero and we're
    // not indexing into the initializer.
    const ConstantInt *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
    if (!FirstIdx || !FirstIdx->isZero())
      return false;

    Value *Op0 = GEP->getOperand(0);
    const GlobalVariable *GV = dyn_cast<GlobalVariable>(Op0);
    if (!GV)
      return false;

    // Fail if the offset into the initializer is not constant.
    const DataLayout &DL = GV->getParent()->getDataLayout();
    APInt Off(DL.getIndexSizeInBits(GEP->getPointerAddressSpace()), 0);
    if (!GEP->accumulateConstantOffset(DL, Off))
      return false;

    // Fail if the constant offset is excessive.
    uint64_t StartIdx = Off.getLimitedValue();
    if (StartIdx == UINT64_MAX)
      return false;

    return getConstantDataArrayInfo(Op0, Slice, ElementSize, StartIdx + Offset);
  }

  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return false;

  const DataLayout &DL = GV->getParent()->getDataLayout();
  ConstantDataArray *Array = nullptr;
  ArrayType *ArrayTy = nullptr;

  if (GV->getInitializer()->isNullValue()) {
    Type *GVTy = GV->getValueType();
    uint64_t SizeInBytes = DL.getTypeStoreSize(GVTy).getFixedSize();
    uint64_t Length = SizeInBytes / (ElementSize / 8);
    if (Length <= Offset)
      // Bail on undersized constants to let sanitizers detect library
      // calls with them as arguments.
      return false;

    Slice.Array = nullptr;
    Slice.Offset = 0;
    Slice.Length = Length - Offset;
    return true;
  }

  auto *Init = const_cast<Constant *>(GV->getInitializer());
  if (auto *ArrayInit = dyn_cast<ConstantDataArray>(Init)) {
    Type *InitElTy = ArrayInit->getElementType();
    if (InitElTy->isIntegerTy(ElementSize)) {
      // If Init is an initializer for an array of the expected type
      // and size, use it as is.
      Array = ArrayInit;
      ArrayTy = ArrayInit->getType();
    }
  }

  if (!Array) {
    if (ElementSize != 8)
      // TODO: Handle conversions to larger integral types.
      return false;

    // Otherwise extract the portion of the initializer starting
    // at Offset as an array of bytes, and reset Offset.
    Init = ReadByteArrayFromGlobal(GV, Offset);
    if (!Init)
      return false;

    Offset = 0;
    Array = dyn_cast<ConstantDataArray>(Init);
    ArrayTy = dyn_cast<ArrayType>(Init->getType());
  }

  uint64_t NumElts = ArrayTy->getArrayNumElements();
  if (Offset > NumElts)
    return false;

  Slice.Array = Array;
  Slice.Offset = Offset;
  Slice.Length = NumElts - Offset;
  return true;
}

/// This function computes the length of a null-terminated C string pointed to
/// by V. If successful, it returns true and returns the string in Str.
/// If unsuccessful, it returns false.
bool llvm::getConstantStringInfo(const Value *V, StringRef &Str,
                                 uint64_t Offset, bool TrimAtNul) {
  ConstantDataArraySlice Slice;
  if (!getConstantDataArrayInfo(V, Slice, 8, Offset))
    return false;

  if (Slice.Array == nullptr) {
    if (TrimAtNul) {
      Str = StringRef();
      return true;
    }
    if (Slice.Length == 1) {
      Str = StringRef("", 1);
      return true;
    }
    // We cannot instantiate a StringRef as we do not have an appropriate string
    // of 0s at hand.
    return false;
  }

  // Start out with the entire array in the StringRef.
  Str = Slice.Array->getAsString();
  // Skip over 'offset' bytes.
  Str = Str.substr(Slice.Offset);

  if (TrimAtNul) {
    // Trim off the \0 and anything after it.  If the array is not nul
    // terminated, we just return the whole end of string.  The client may know
    // some other way that the string is length-bound.
    Str = Str.substr(0, Str.find('\0'));
  }
  return true;
}

// These next two are very similar to the above, but also look through PHI
// nodes.
// TODO: See if we can integrate these two together.

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(const Value *V,
                                 SmallPtrSetImpl<const PHINode*> &PHIs,
                                 unsigned CharSize) {
  // Look through noop bitcast instructions.
  V = V->stripPointerCasts();

  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN).second)
      return ~0ULL;  // already in the set.

    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (Value *IncValue : PN->incoming_values()) {
      uint64_t Len = GetStringLengthH(IncValue, PHIs, CharSize);
      if (Len == 0) return 0; // Unknown length -> unknown.

      if (Len == ~0ULL) continue;

      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0;    // Disagree -> unknown.
      LenSoFar = Len;
    }

    // Success, all agree.
    return LenSoFar;
  }

  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs, CharSize);
    if (Len1 == 0) return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs, CharSize);
    if (Len2 == 0) return 0;
    if (Len1 == ~0ULL) return Len2;
    if (Len2 == ~0ULL) return Len1;
    if (Len1 != Len2) return 0;
    return Len1;
  }

  // Otherwise, see if we can read the string.
  ConstantDataArraySlice Slice;
  if (!getConstantDataArrayInfo(V, Slice, CharSize))
    return 0;

  if (Slice.Array == nullptr)
    return 1;

  // Search for nul characters
  unsigned NullIndex = 0;
  for (unsigned E = Slice.Length; NullIndex < E; ++NullIndex) {
    if (Slice.Array->getElementAsInteger(Slice.Offset + NullIndex) == 0)
      break;
  }

  return NullIndex + 1;
}

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
uint64_t llvm::GetStringLength(const Value *V, unsigned CharSize) {
  if (!V->getType()->isPointerTy())
    return 0;

  SmallPtrSet<const PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs, CharSize);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

const Value *
llvm::getArgumentAliasingToReturnedPointer(const CallBase *Call,
                                           bool MustPreserveNullness) {
  assert(Call &&
         "getArgumentAliasingToReturnedPointer only works on nonnull calls");
  if (const Value *RV = Call->getReturnedArgOperand())
    return RV;
  // This can be used only as a aliasing property.
  if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(
          Call, MustPreserveNullness))
    return Call->getArgOperand(0);
  return nullptr;
}

bool llvm::isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(
    const CallBase *Call, bool MustPreserveNullness) {
  switch (Call->getIntrinsicID()) {
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group:
  case Intrinsic::aarch64_irg:
  case Intrinsic::aarch64_tagp:
    return true;
  case Intrinsic::ptrmask:
    return !MustPreserveNullness;
  default:
    return false;
  }
}

/// \p PN defines a loop-variant pointer to an object.  Check if the
/// previous iteration of the loop was referring to the same object as \p PN.
static bool isSameUnderlyingObjectInLoop(const PHINode *PN,
                                         const LoopInfo *LI) {
  // Find the loop-defined value.
  Loop *L = LI->getLoopFor(PN->getParent());
  if (PN->getNumIncomingValues() != 2)
    return true;

  // Find the value from previous iteration.
  auto *PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(0));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(1));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    return true;

  // If a new pointer is loaded in the loop, the pointer references a different
  // object in every iteration.  E.g.:
  //    for (i)
  //       int *p = a[i];
  //       ...
  if (auto *Load = dyn_cast<LoadInst>(PrevValue))
    if (!L->isLoopInvariant(Load->getPointerOperand()))
      return false;
  return true;
}

const Value *llvm::getUnderlyingObject(const Value *V, unsigned MaxLookup) {
  if (!V->getType()->isPointerTy())
    return V;
  for (unsigned Count = 0; MaxLookup == 0 || Count < MaxLookup; ++Count) {
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast ||
               Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
      V = cast<Operator>(V)->getOperand(0);
      if (!V->getType()->isPointerTy())
        return V;
    } else if (auto *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->isInterposable())
        return V;
      V = GA->getAliasee();
    } else {
      if (auto *PHI = dyn_cast<PHINode>(V)) {
        // Look through single-arg phi nodes created by LCSSA.
        if (PHI->getNumIncomingValues() == 1) {
          V = PHI->getIncomingValue(0);
          continue;
        }
      } else if (auto *Call = dyn_cast<CallBase>(V)) {
        // CaptureTracking can know about special capturing properties of some
        // intrinsics like launder.invariant.group, that can't be expressed with
        // the attributes, but have properties like returning aliasing pointer.
        // Because some analysis may assume that nocaptured pointer is not
        // returned from some special intrinsic (because function would have to
        // be marked with returns attribute), it is crucial to use this function
        // because it should be in sync with CaptureTracking. Not using it may
        // cause weird miscompilations where 2 aliasing pointers are assumed to
        // noalias.
        if (auto *RP = getArgumentAliasingToReturnedPointer(Call, false)) {
          V = RP;
          continue;
        }
      }

      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  }
  return V;
}

void llvm::getUnderlyingObjects(const Value *V,
                                SmallVectorImpl<const Value *> &Objects,
                                LoopInfo *LI, unsigned MaxLookup) {
  SmallPtrSet<const Value *, 4> Visited;
  SmallVector<const Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    const Value *P = Worklist.pop_back_val();
    P = getUnderlyingObject(P, MaxLookup);

    if (!Visited.insert(P).second)
      continue;

    if (auto *SI = dyn_cast<SelectInst>(P)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(P)) {
      // If this PHI changes the underlying object in every iteration of the
      // loop, don't look through it.  Consider:
      //   int **A;
      //   for (i) {
      //     Prev = Curr;     // Prev = PHI (Prev_0, Curr)
      //     Curr = A[i];
      //     *Prev, *Curr;
      //
      // Prev is tracking Curr one iteration behind so they refer to different
      // underlying objects.
      if (!LI || !LI->isLoopHeader(PN->getParent()) ||
          isSameUnderlyingObjectInLoop(PN, LI))
        append_range(Worklist, PN->incoming_values());
      continue;
    }

    Objects.push_back(P);
  } while (!Worklist.empty());
}

/// This is the function that does the work of looking through basic
/// ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant, a multiplied value, or a phi, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul &&
           !isa<PHINode>(U->getOperand(1))))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (true);
}

/// This is a wrapper around getUnderlyingObjects and adds support for basic
/// ptrtoint+arithmetic+inttoptr sequences.
/// It returns false if unidentified object is found in getUnderlyingObjects.
bool llvm::getUnderlyingObjectsForCodeGen(const Value *V,
                                          SmallVectorImpl<Value *> &Objects) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 4> Working(1, V);
  do {
    V = Working.pop_back_val();

    SmallVector<const Value *, 4> Objs;
    getUnderlyingObjects(V, Objs);

    for (const Value *V : Objs) {
      if (!Visited.insert(V).second)
        continue;
      if (Operator::getOpcode(V) == Instruction::IntToPtr) {
        const Value *O =
          getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
        if (O->getType()->isPointerTy()) {
          Working.push_back(O);
          continue;
        }
      }
      // If getUnderlyingObjects fails to find an identifiable object,
      // getUnderlyingObjectsForCodeGen also fails for safety.
      if (!isIdentifiedObject(V)) {
        Objects.clear();
        return false;
      }
      Objects.push_back(const_cast<Value *>(V));
    }
  } while (!Working.empty());
  return true;
}

AllocaInst *llvm::findAllocaForValue(Value *V, bool OffsetZero) {
  AllocaInst *Result = nullptr;
  SmallPtrSet<Value *, 4> Visited;
  SmallVector<Value *, 4> Worklist;

  auto AddWork = [&](Value *V) {
    if (Visited.insert(V).second)
      Worklist.push_back(V);
  };

  AddWork(V);
  do {
    V = Worklist.pop_back_val();
    assert(Visited.count(V));

    if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      if (Result && Result != AI)
        return nullptr;
      Result = AI;
    } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
      AddWork(CI->getOperand(0));
    } else if (PHINode *PN = dyn_cast<PHINode>(V)) {
      for (Value *IncValue : PN->incoming_values())
        AddWork(IncValue);
    } else if (auto *SI = dyn_cast<SelectInst>(V)) {
      AddWork(SI->getTrueValue());
      AddWork(SI->getFalseValue());
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if (OffsetZero && !GEP->hasAllZeroIndices())
        return nullptr;
      AddWork(GEP->getPointerOperand());
    } else if (CallBase *CB = dyn_cast<CallBase>(V)) {
      Value *Returned = CB->getReturnedArgOperand();
      if (Returned)
        AddWork(Returned);
      else
        return nullptr;
    } else {
      return nullptr;
    }
  } while (!Worklist.empty());

  return Result;
}

static bool onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
    const Value *V, bool AllowLifetime, bool AllowDroppable) {
  for (const User *U : V->users()) {
    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    if (!II)
      return false;

    if (AllowLifetime && II->isLifetimeStartOrEnd())
      continue;

    if (AllowDroppable && II->isDroppable())
      continue;

    return false;
  }
  return true;
}

bool llvm::onlyUsedByLifetimeMarkers(const Value *V) {
  return onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
      V, /* AllowLifetime */ true, /* AllowDroppable */ false);
}
bool llvm::onlyUsedByLifetimeMarkersOrDroppableInsts(const Value *V) {
  return onlyUsedByLifetimeMarkersOrDroppableInstsHelper(
      V, /* AllowLifetime */ true, /* AllowDroppable */ true);
}

bool llvm::mustSuppressSpeculation(const LoadInst &LI) {
  if (!LI.isUnordered())
    return true;
  const Function &F = *LI.getFunction();
  // Speculative load may create a race that did not exist in the source.
  return F.hasFnAttribute(Attribute::SanitizeThread) ||
    // Speculative load may load data from dirty regions.
    F.hasFnAttribute(Attribute::SanitizeAddress) ||
    F.hasFnAttribute(Attribute::SanitizeHWAddress);
}


bool llvm::isSafeToSpeculativelyExecute(const Value *V,
                                        const Instruction *CtxI,
                                        const DominatorTree *DT,
                                        const TargetLibraryInfo *TLI) {
  const Operator *Inst = dyn_cast<Operator>(V);
  if (!Inst)
    return false;
  return isSafeToSpeculativelyExecuteWithOpcode(Inst->getOpcode(), Inst, CtxI, DT, TLI);
}

bool llvm::isSafeToSpeculativelyExecuteWithOpcode(unsigned Opcode,
                                        const Operator *Inst,
                                        const Instruction *CtxI,
                                        const DominatorTree *DT,
                                        const TargetLibraryInfo *TLI) {
#ifndef NDEBUG
  if (Inst->getOpcode() != Opcode) {
    // Check that the operands are actually compatible with the Opcode override.
    auto hasEqualReturnAndLeadingOperandTypes =
        [](const Operator *Inst, unsigned NumLeadingOperands) {
          if (Inst->getNumOperands() < NumLeadingOperands)
            return false;
          const Type *ExpectedType = Inst->getType();
          for (unsigned ItOp = 0; ItOp < NumLeadingOperands; ++ItOp)
            if (Inst->getOperand(ItOp)->getType() != ExpectedType)
              return false;
          return true;
        };
    assert(!Instruction::isBinaryOp(Opcode) ||
           hasEqualReturnAndLeadingOperandTypes(Inst, 2));
    assert(!Instruction::isUnaryOp(Opcode) ||
           hasEqualReturnAndLeadingOperandTypes(Inst, 1));
  }
#endif

  for (unsigned i = 0, e = Inst->getNumOperands(); i != e; ++i)
    if (Constant *C = dyn_cast<Constant>(Inst->getOperand(i)))
      if (C->canTrap())
        return false;

  switch (Opcode) {
  default:
    return true;
  case Instruction::UDiv:
  case Instruction::URem: {
    // x / y is undefined if y == 0.
    const APInt *V;
    if (match(Inst->getOperand(1), m_APInt(V)))
      return *V != 0;
    return false;
  }
  case Instruction::SDiv:
  case Instruction::SRem: {
    // x / y is undefined if y == 0 or x == INT_MIN and y == -1
    const APInt *Numerator, *Denominator;
    if (!match(Inst->getOperand(1), m_APInt(Denominator)))
      return false;
    // We cannot hoist this division if the denominator is 0.
    if (*Denominator == 0)
      return false;
    // It's safe to hoist if the denominator is not 0 or -1.
    if (!Denominator->isAllOnes())
      return true;
    // At this point we know that the denominator is -1.  It is safe to hoist as
    // long we know that the numerator is not INT_MIN.
    if (match(Inst->getOperand(0), m_APInt(Numerator)))
      return !Numerator->isMinSignedValue();
    // The numerator *might* be MinSignedValue.
    return false;
  }
  case Instruction::Load: {
    const LoadInst *LI = dyn_cast<LoadInst>(Inst);
    if (!LI)
      return false;
    if (mustSuppressSpeculation(*LI))
      return false;
    const DataLayout &DL = LI->getModule()->getDataLayout();
    return isDereferenceableAndAlignedPointer(
        LI->getPointerOperand(), LI->getType(), LI->getAlign(), DL, CtxI, DT,
        TLI);
  }
  case Instruction::Call: {
    auto *CI = dyn_cast<const CallInst>(Inst);
    if (!CI)
      return false;
    const Function *Callee = CI->getCalledFunction();

    // The called function could have undefined behavior or side-effects, even
    // if marked readnone nounwind.
    return Callee && Callee->isSpeculatable();
  }
  case Instruction::VAArg:
  case Instruction::Alloca:
  case Instruction::Invoke:
  case Instruction::CallBr:
  case Instruction::PHI:
  case Instruction::Store:
  case Instruction::Ret:
  case Instruction::Br:
  case Instruction::IndirectBr:
  case Instruction::Switch:
  case Instruction::Unreachable:
  case Instruction::Fence:
  case Instruction::AtomicRMW:
  case Instruction::AtomicCmpXchg:
  case Instruction::LandingPad:
  case Instruction::Resume:
  case Instruction::CatchSwitch:
  case Instruction::CatchPad:
  case Instruction::CatchRet:
  case Instruction::CleanupPad:
  case Instruction::CleanupRet:
    return false; // Misc instructions which have effects
  }
}

bool llvm::mayHaveNonDefUseDependency(const Instruction &I) {
  if (I.mayReadOrWriteMemory())
    // Memory dependency possible
    return true;
  if (!isSafeToSpeculativelyExecute(&I))
    // Can't move above a maythrow call or infinite loop.  Or if an
    // inalloca alloca, above a stacksave call.
    return true;
  if (!isGuaranteedToTransferExecutionToSuccessor(&I))
    // 1) Can't reorder two inf-loop calls, even if readonly
    // 2) Also can't reorder an inf-loop call below a instruction which isn't
    //    safe to speculative execute.  (Inverse of above)
    return true;
  return false;
}

/// Convert ConstantRange OverflowResult into ValueTracking OverflowResult.
static OverflowResult mapOverflowResult(ConstantRange::OverflowResult OR) {
  switch (OR) {
    case ConstantRange::OverflowResult::MayOverflow:
      return OverflowResult::MayOverflow;
    case ConstantRange::OverflowResult::AlwaysOverflowsLow:
      return OverflowResult::AlwaysOverflowsLow;
    case ConstantRange::OverflowResult::AlwaysOverflowsHigh:
      return OverflowResult::AlwaysOverflowsHigh;
    case ConstantRange::OverflowResult::NeverOverflows:
      return OverflowResult::NeverOverflows;
  }
  llvm_unreachable("Unknown OverflowResult");
}

/// Combine constant ranges from computeConstantRange() and computeKnownBits().
static ConstantRange computeConstantRangeIncludingKnownBits(
    const Value *V, bool ForSigned, const DataLayout &DL, unsigned Depth,
    AssumptionCache *AC, const Instruction *CxtI, const DominatorTree *DT,
    OptimizationRemarkEmitter *ORE = nullptr, bool UseInstrInfo = true) {
  KnownBits Known = computeKnownBits(
      V, DL, Depth, AC, CxtI, DT, ORE, UseInstrInfo);
  ConstantRange CR1 = ConstantRange::fromKnownBits(Known, ForSigned);
  ConstantRange CR2 = computeConstantRange(V, UseInstrInfo);
  ConstantRange::PreferredRangeType RangeType =
      ForSigned ? ConstantRange::Signed : ConstantRange::Unsigned;
  return CR1.intersectWith(CR2, RangeType);
}

OverflowResult llvm::computeOverflowForUnsignedMul(
    const Value *LHS, const Value *RHS, const DataLayout &DL,
    AssumptionCache *AC, const Instruction *CxtI, const DominatorTree *DT,
    bool UseInstrInfo) {
  KnownBits LHSKnown = computeKnownBits(LHS, DL, /*Depth=*/0, AC, CxtI, DT,
                                        nullptr, UseInstrInfo);
  KnownBits RHSKnown = computeKnownBits(RHS, DL, /*Depth=*/0, AC, CxtI, DT,
                                        nullptr, UseInstrInfo);
  ConstantRange LHSRange = ConstantRange::fromKnownBits(LHSKnown, false);
  ConstantRange RHSRange = ConstantRange::fromKnownBits(RHSKnown, false);
  return mapOverflowResult(LHSRange.unsignedMulMayOverflow(RHSRange));
}

OverflowResult
llvm::computeOverflowForSignedMul(const Value *LHS, const Value *RHS,
                                  const DataLayout &DL, AssumptionCache *AC,
                                  const Instruction *CxtI,
                                  const DominatorTree *DT, bool UseInstrInfo) {
  // Multiplying n * m significant bits yields a result of n + m significant
  // bits. If the total number of significant bits does not exceed the
  // result bit width (minus 1), there is no overflow.
  // This means if we have enough leading sign bits in the operands
  // we can guarantee that the result does not overflow.
  // Ref: "Hacker's Delight" by Henry Warren
  unsigned BitWidth = LHS->getType()->getScalarSizeInBits();

  // Note that underestimating the number of sign bits gives a more
  // conservative answer.
  unsigned SignBits = ComputeNumSignBits(LHS, DL, 0, AC, CxtI, DT) +
                      ComputeNumSignBits(RHS, DL, 0, AC, CxtI, DT);

  // First handle the easy case: if we have enough sign bits there's
  // definitely no overflow.
  if (SignBits > BitWidth + 1)
    return OverflowResult::NeverOverflows;

  // There are two ambiguous cases where there can be no overflow:
  //   SignBits == BitWidth + 1    and
  //   SignBits == BitWidth
  // The second case is difficult to check, therefore we only handle the
  // first case.
  if (SignBits == BitWidth + 1) {
    // It overflows only when both arguments are negative and the true
    // product is exactly the minimum negative number.
    // E.g. mul i16 with 17 sign bits: 0xff00 * 0xff80 = 0x8000
    // For simplicity we just check if at least one side is not negative.
    KnownBits LHSKnown = computeKnownBits(LHS, DL, /*Depth=*/0, AC, CxtI, DT,
                                          nullptr, UseInstrInfo);
    KnownBits RHSKnown = computeKnownBits(RHS, DL, /*Depth=*/0, AC, CxtI, DT,
                                          nullptr, UseInstrInfo);
    if (LHSKnown.isNonNegative() || RHSKnown.isNonNegative())
      return OverflowResult::NeverOverflows;
  }
  return OverflowResult::MayOverflow;
}

OverflowResult llvm::computeOverflowForUnsignedAdd(
    const Value *LHS, const Value *RHS, const DataLayout &DL,
    AssumptionCache *AC, const Instruction *CxtI, const DominatorTree *DT,
    bool UseInstrInfo) {
  ConstantRange LHSRange = computeConstantRangeIncludingKnownBits(
      LHS, /*ForSigned=*/false, DL, /*Depth=*/0, AC, CxtI, DT,
      nullptr, UseInstrInfo);
  ConstantRange RHSRange = computeConstantRangeIncludingKnownBits(
      RHS, /*ForSigned=*/false, DL, /*Depth=*/0, AC, CxtI, DT,
      nullptr, UseInstrInfo);
  return mapOverflowResult(LHSRange.unsignedAddMayOverflow(RHSRange));
}

static OverflowResult computeOverflowForSignedAdd(const Value *LHS,
                                                  const Value *RHS,
                                                  const AddOperator *Add,
                                                  const DataLayout &DL,
                                                  AssumptionCache *AC,
                                                  const Instruction *CxtI,
                                                  const DominatorTree *DT) {
  if (Add && Add->hasNoSignedWrap()) {
    return OverflowResult::NeverOverflows;
  }

  // If LHS and RHS each have at least two sign bits, the addition will look
  // like
  //
  // XX..... +
  // YY.....
  //
  // If the carry into the most significant position is 0, X and Y can't both
  // be 1 and therefore the carry out of the addition is also 0.
  //
  // If the carry into the most significant position is 1, X and Y can't both
  // be 0 and therefore the carry out of the addition is also 1.
  //
  // Since the carry into the most significant position is always equal to
  // the carry out of the addition, there is no signed overflow.
  if (ComputeNumSignBits(LHS, DL, 0, AC, CxtI, DT) > 1 &&
      ComputeNumSignBits(RHS, DL, 0, AC, CxtI, DT) > 1)
    return OverflowResult::NeverOverflows;

  ConstantRange LHSRange = computeConstantRangeIncludingKnownBits(
      LHS, /*ForSigned=*/true, DL, /*Depth=*/0, AC, CxtI, DT);
  ConstantRange RHSRange = computeConstantRangeIncludingKnownBits(
      RHS, /*ForSigned=*/true, DL, /*Depth=*/0, AC, CxtI, DT);
  OverflowResult OR =
      mapOverflowResult(LHSRange.signedAddMayOverflow(RHSRange));
  if (OR != OverflowResult::MayOverflow)
    return OR;

  // The remaining code needs Add to be available. Early returns if not so.
  if (!Add)
    return OverflowResult::MayOverflow;

  // If the sign of Add is the same as at least one of the operands, this add
  // CANNOT overflow. If this can be determined from the known bits of the
  // operands the above signedAddMayOverflow() check will have already done so.
  // The only other way to improve on the known bits is from an assumption, so
  // call computeKnownBitsFromAssume() directly.
  bool LHSOrRHSKnownNonNegative =
      (LHSRange.isAllNonNegative() || RHSRange.isAllNonNegative());
  bool LHSOrRHSKnownNegative =
      (LHSRange.isAllNegative() || RHSRange.isAllNegative());
  if (LHSOrRHSKnownNonNegative || LHSOrRHSKnownNegative) {
    KnownBits AddKnown(LHSRange.getBitWidth());
    computeKnownBitsFromAssume(
        Add, AddKnown, /*Depth=*/0, Query(DL, AC, CxtI, DT, true));
    if ((AddKnown.isNonNegative() && LHSOrRHSKnownNonNegative) ||
        (AddKnown.isNegative() && LHSOrRHSKnownNegative))
      return OverflowResult::NeverOverflows;
  }

  return OverflowResult::MayOverflow;
}

OverflowResult llvm::computeOverflowForUnsignedSub(const Value *LHS,
                                                   const Value *RHS,
                                                   const DataLayout &DL,
                                                   AssumptionCache *AC,
                                                   const Instruction *CxtI,
                                                   const DominatorTree *DT) {
  // X - (X % ?)
  // The remainder of a value can't have greater magnitude than itself,
  // so the subtraction can't overflow.

  // X - (X -nuw ?)
  // In the minimal case, this would simplify to "?", so there's no subtract
  // at all. But if this analysis is used to peek through casts, for example,
  // then determining no-overflow may allow other transforms.

  // TODO: There are other patterns like this.
  //       See simplifyICmpWithBinOpOnLHS() for candidates.
  if (match(RHS, m_URem(m_Specific(LHS), m_Value())) ||
      match(RHS, m_NUWSub(m_Specific(LHS), m_Value())))
    if (isGuaranteedNotToBeUndefOrPoison(LHS, AC, CxtI, DT))
      return OverflowResult::NeverOverflows;

  // Checking for conditions implied by dominating conditions may be expensive.
  // Limit it to usub_with_overflow calls for now.
  if (match(CxtI,
            m_Intrinsic<Intrinsic::usub_with_overflow>(m_Value(), m_Value())))
    if (auto C =
            isImpliedByDomCondition(CmpInst::ICMP_UGE, LHS, RHS, CxtI, DL)) {
      if (*C)
        return OverflowResult::NeverOverflows;
      return OverflowResult::AlwaysOverflowsLow;
    }
  ConstantRange LHSRange = computeConstantRangeIncludingKnownBits(
      LHS, /*ForSigned=*/false, DL, /*Depth=*/0, AC, CxtI, DT);
  ConstantRange RHSRange = computeConstantRangeIncludingKnownBits(
      RHS, /*ForSigned=*/false, DL, /*Depth=*/0, AC, CxtI, DT);
  return mapOverflowResult(LHSRange.unsignedSubMayOverflow(RHSRange));
}

OverflowResult llvm::computeOverflowForSignedSub(const Value *LHS,
                                                 const Value *RHS,
                                                 const DataLayout &DL,
                                                 AssumptionCache *AC,
                                                 const Instruction *CxtI,
                                                 const DominatorTree *DT) {
  // X - (X % ?)
  // The remainder of a value can't have greater magnitude than itself,
  // so the subtraction can't overflow.

  // X - (X -nsw ?)
  // In the minimal case, this would simplify to "?", so there's no subtract
  // at all. But if this analysis is used to peek through casts, for example,
  // then determining no-overflow may allow other transforms.
  if (match(RHS, m_SRem(m_Specific(LHS), m_Value())) ||
      match(RHS, m_NSWSub(m_Specific(LHS), m_Value())))
    if (isGuaranteedNotToBeUndefOrPoison(LHS, AC, CxtI, DT))
      return OverflowResult::NeverOverflows;

  // If LHS and RHS each have at least two sign bits, the subtraction
  // cannot overflow.
  if (ComputeNumSignBits(LHS, DL, 0, AC, CxtI, DT) > 1 &&
      ComputeNumSignBits(RHS, DL, 0, AC, CxtI, DT) > 1)
    return OverflowResult::NeverOverflows;

  ConstantRange LHSRange = computeConstantRangeIncludingKnownBits(
      LHS, /*ForSigned=*/true, DL, /*Depth=*/0, AC, CxtI, DT);
  ConstantRange RHSRange = computeConstantRangeIncludingKnownBits(
      RHS, /*ForSigned=*/true, DL, /*Depth=*/0, AC, CxtI, DT);
  return mapOverflowResult(LHSRange.signedSubMayOverflow(RHSRange));
}

bool llvm::isOverflowIntrinsicNoWrap(const WithOverflowInst *WO,
                                     const DominatorTree &DT) {
  SmallVector<const BranchInst *, 2> GuardingBranches;
  SmallVector<const ExtractValueInst *, 2> Results;

  for (const User *U : WO->users()) {
    if (const auto *EVI = dyn_cast<ExtractValueInst>(U)) {
      assert(EVI->getNumIndices() == 1 && "Obvious from CI's type");

      if (EVI->getIndices()[0] == 0)
        Results.push_back(EVI);
      else {
        assert(EVI->getIndices()[0] == 1 && "Obvious from CI's type");

        for (const auto *U : EVI->users())
          if (const auto *B = dyn_cast<BranchInst>(U)) {
            assert(B->isConditional() && "How else is it using an i1?");
            GuardingBranches.push_back(B);
          }
      }
    } else {
      // We are using the aggregate directly in a way we don't want to analyze
      // here (storing it to a global, say).
      return false;
    }
  }

  auto AllUsesGuardedByBranch = [&](const BranchInst *BI) {
    BasicBlockEdge NoWrapEdge(BI->getParent(), BI->getSuccessor(1));
    if (!NoWrapEdge.isSingleEdge())
      return false;

    // Check if all users of the add are provably no-wrap.
    for (const auto *Result : Results) {
      // If the extractvalue itself is not executed on overflow, the we don't
      // need to check each use separately, since domination is transitive.
      if (DT.dominates(NoWrapEdge, Result->getParent()))
        continue;

      for (auto &RU : Result->uses())
        if (!DT.dominates(NoWrapEdge, RU))
          return false;
    }

    return true;
  };

  return llvm::any_of(GuardingBranches, AllUsesGuardedByBranch);
}

static bool canCreateUndefOrPoison(const Operator *Op, bool PoisonOnly,
                                   bool ConsiderFlags) {

  if (ConsiderFlags && Op->hasPoisonGeneratingFlags())
    return true;

  unsigned Opcode = Op->getOpcode();

  // Check whether opcode is a poison/undef-generating operation
  switch (Opcode) {
  case Instruction::Shl:
  case Instruction::AShr:
  case Instruction::LShr: {
    // Shifts return poison if shiftwidth is larger than the bitwidth.
    if (auto *C = dyn_cast<Constant>(Op->getOperand(1))) {
      SmallVector<Constant *, 4> ShiftAmounts;
      if (auto *FVTy = dyn_cast<FixedVectorType>(C->getType())) {
        unsigned NumElts = FVTy->getNumElements();
        for (unsigned i = 0; i < NumElts; ++i)
          ShiftAmounts.push_back(C->getAggregateElement(i));
      } else if (isa<ScalableVectorType>(C->getType()))
        return true; // Can't tell, just return true to be safe
      else
        ShiftAmounts.push_back(C);

      bool Safe = llvm::all_of(ShiftAmounts, [](Constant *C) {
        auto *CI = dyn_cast_or_null<ConstantInt>(C);
        return CI && CI->getValue().ult(C->getType()->getIntegerBitWidth());
      });
      return !Safe;
    }
    return true;
  }
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    // fptosi/ui yields poison if the resulting value does not fit in the
    // destination type.
    return true;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(Op)) {
      switch (II->getIntrinsicID()) {
      // TODO: Add more intrinsics.
      case Intrinsic::ctpop:
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::smul_with_overflow:
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::usub_with_overflow:
      case Intrinsic::umul_with_overflow:
        return false;
      }
    }
    LLVM_FALLTHROUGH;
  case Instruction::CallBr:
  case Instruction::Invoke: {
    const auto *CB = cast<CallBase>(Op);
    return !CB->hasRetAttr(Attribute::NoUndef);
  }
  case Instruction::InsertElement:
  case Instruction::ExtractElement: {
    // If index exceeds the length of the vector, it returns poison
    auto *VTy = cast<VectorType>(Op->getOperand(0)->getType());
    unsigned IdxOp = Op->getOpcode() == Instruction::InsertElement ? 2 : 1;
    auto *Idx = dyn_cast<ConstantInt>(Op->getOperand(IdxOp));
    if (!Idx || Idx->getValue().uge(VTy->getElementCount().getKnownMinValue()))
      return true;
    return false;
  }
  case Instruction::ShuffleVector: {
    // shufflevector may return undef.
    if (PoisonOnly)
      return false;
    ArrayRef<int> Mask = isa<ConstantExpr>(Op)
                             ? cast<ConstantExpr>(Op)->getShuffleMask()
                             : cast<ShuffleVectorInst>(Op)->getShuffleMask();
    return is_contained(Mask, UndefMaskElem);
  }
  case Instruction::FNeg:
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
  case Instruction::Freeze:
  case Instruction::ICmp:
  case Instruction::FCmp:
    return false;
  case Instruction::GetElementPtr:
    // inbounds is handled above
    // TODO: what about inrange on constexpr?
    return false;
  default: {
    const auto *CE = dyn_cast<ConstantExpr>(Op);
    if (isa<CastInst>(Op) || (CE && CE->isCast()))
      return false;
    else if (Instruction::isBinaryOp(Opcode))
      return false;
    // Be conservative and return true.
    return true;
  }
  }
}

bool llvm::canCreateUndefOrPoison(const Operator *Op, bool ConsiderFlags) {
  return ::canCreateUndefOrPoison(Op, /*PoisonOnly=*/false, ConsiderFlags);
}

bool llvm::canCreatePoison(const Operator *Op, bool ConsiderFlags) {
  return ::canCreateUndefOrPoison(Op, /*PoisonOnly=*/true, ConsiderFlags);
}

static bool directlyImpliesPoison(const Value *ValAssumedPoison,
                                  const Value *V, unsigned Depth) {
  if (ValAssumedPoison == V)
    return true;

  const unsigned MaxDepth = 2;
  if (Depth >= MaxDepth)
    return false;

  if (const auto *I = dyn_cast<Instruction>(V)) {
    if (propagatesPoison(cast<Operator>(I)))
      return any_of(I->operands(), [=](const Value *Op) {
        return directlyImpliesPoison(ValAssumedPoison, Op, Depth + 1);
      });

    // 'select ValAssumedPoison, _, _' is poison.
    if (const auto *SI = dyn_cast<SelectInst>(I))
      return directlyImpliesPoison(ValAssumedPoison, SI->getCondition(),
                                   Depth + 1);
    // V  = extractvalue V0, idx
    // V2 = extractvalue V0, idx2
    // V0's elements are all poison or not. (e.g., add_with_overflow)
    const WithOverflowInst *II;
    if (match(I, m_ExtractValue(m_WithOverflowInst(II))) &&
        (match(ValAssumedPoison, m_ExtractValue(m_Specific(II))) ||
         llvm::is_contained(II->args(), ValAssumedPoison)))
      return true;
  }
  return false;
}

static bool impliesPoison(const Value *ValAssumedPoison, const Value *V,
                          unsigned Depth) {
  if (isGuaranteedNotToBeUndefOrPoison(ValAssumedPoison))
    return true;

  if (directlyImpliesPoison(ValAssumedPoison, V, /* Depth */ 0))
    return true;

  const unsigned MaxDepth = 2;
  if (Depth >= MaxDepth)
    return false;

  const auto *I = dyn_cast<Instruction>(ValAssumedPoison);
  if (I && !canCreatePoison(cast<Operator>(I))) {
    return all_of(I->operands(), [=](const Value *Op) {
      return impliesPoison(Op, V, Depth + 1);
    });
  }
  return false;
}

bool llvm::impliesPoison(const Value *ValAssumedPoison, const Value *V) {
  return ::impliesPoison(ValAssumedPoison, V, /* Depth */ 0);
}

static bool programUndefinedIfUndefOrPoison(const Value *V,
                                            bool PoisonOnly);

static bool isGuaranteedNotToBeUndefOrPoison(const Value *V,
                                             AssumptionCache *AC,
                                             const Instruction *CtxI,
                                             const DominatorTree *DT,
                                             unsigned Depth, bool PoisonOnly) {
  if (Depth >= MaxAnalysisRecursionDepth)
    return false;

  if (isa<MetadataAsValue>(V))
    return false;

  if (const auto *A = dyn_cast<Argument>(V)) {
    if (A->hasAttribute(Attribute::NoUndef))
      return true;
  }

  if (auto *C = dyn_cast<Constant>(V)) {
    if (isa<UndefValue>(C))
      return PoisonOnly && !isa<PoisonValue>(C);

    if (isa<ConstantInt>(C) || isa<GlobalVariable>(C) || isa<ConstantFP>(V) ||
        isa<ConstantPointerNull>(C) || isa<Function>(C))
      return true;

    if (C->getType()->isVectorTy() && !isa<ConstantExpr>(C))
      return (PoisonOnly ? !C->containsPoisonElement()
                         : !C->containsUndefOrPoisonElement()) &&
             !C->containsConstantExpression();
  }

  // Strip cast operations from a pointer value.
  // Note that stripPointerCastsSameRepresentation can strip off getelementptr
  // inbounds with zero offset. To guarantee that the result isn't poison, the
  // stripped pointer is checked as it has to be pointing into an allocated
  // object or be null `null` to ensure `inbounds` getelement pointers with a
  // zero offset could not produce poison.
  // It can strip off addrspacecast that do not change bit representation as
  // well. We believe that such addrspacecast is equivalent to no-op.
  auto *StrippedV = V->stripPointerCastsSameRepresentation();
  if (isa<AllocaInst>(StrippedV) || isa<GlobalVariable>(StrippedV) ||
      isa<Function>(StrippedV) || isa<ConstantPointerNull>(StrippedV))
    return true;

  auto OpCheck = [&](const Value *V) {
    return isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth + 1,
                                            PoisonOnly);
  };

  if (auto *Opr = dyn_cast<Operator>(V)) {
    // If the value is a freeze instruction, then it can never
    // be undef or poison.
    if (isa<FreezeInst>(V))
      return true;

    if (const auto *CB = dyn_cast<CallBase>(V)) {
      if (CB->hasRetAttr(Attribute::NoUndef))
        return true;
    }

    if (const auto *PN = dyn_cast<PHINode>(V)) {
      unsigned Num = PN->getNumIncomingValues();
      bool IsWellDefined = true;
      for (unsigned i = 0; i < Num; ++i) {
        auto *TI = PN->getIncomingBlock(i)->getTerminator();
        if (!isGuaranteedNotToBeUndefOrPoison(PN->getIncomingValue(i), AC, TI,
                                              DT, Depth + 1, PoisonOnly)) {
          IsWellDefined = false;
          break;
        }
      }
      if (IsWellDefined)
        return true;
    } else if (!canCreateUndefOrPoison(Opr) && all_of(Opr->operands(), OpCheck))
      return true;
  }

  if (auto *I = dyn_cast<LoadInst>(V))
    if (I->hasMetadata(LLVMContext::MD_noundef) ||
        I->hasMetadata(LLVMContext::MD_dereferenceable) ||
        I->hasMetadata(LLVMContext::MD_dereferenceable_or_null))
      return true;

  if (programUndefinedIfUndefOrPoison(V, PoisonOnly))
    return true;

  // CxtI may be null or a cloned instruction.
  if (!CtxI || !CtxI->getParent() || !DT)
    return false;

  auto *DNode = DT->getNode(CtxI->getParent());
  if (!DNode)
    // Unreachable block
    return false;

  // If V is used as a branch condition before reaching CtxI, V cannot be
  // undef or poison.
  //   br V, BB1, BB2
  // BB1:
  //   CtxI ; V cannot be undef or poison here
  auto *Dominator = DNode->getIDom();
  while (Dominator) {
    auto *TI = Dominator->getBlock()->getTerminator();

    Value *Cond = nullptr;
    if (auto BI = dyn_cast_or_null<BranchInst>(TI)) {
      if (BI->isConditional())
        Cond = BI->getCondition();
    } else if (auto SI = dyn_cast_or_null<SwitchInst>(TI)) {
      Cond = SI->getCondition();
    }

    if (Cond) {
      if (Cond == V)
        return true;
      else if (PoisonOnly && isa<Operator>(Cond)) {
        // For poison, we can analyze further
        auto *Opr = cast<Operator>(Cond);
        if (propagatesPoison(Opr) && is_contained(Opr->operand_values(), V))
          return true;
      }
    }

    Dominator = Dominator->getIDom();
  }

  if (getKnowledgeValidInContext(V, {Attribute::NoUndef}, CtxI, DT, AC))
    return true;

  return false;
}

bool llvm::isGuaranteedNotToBeUndefOrPoison(const Value *V, AssumptionCache *AC,
                                            const Instruction *CtxI,
                                            const DominatorTree *DT,
                                            unsigned Depth) {
  return ::isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth, false);
}

bool llvm::isGuaranteedNotToBePoison(const Value *V, AssumptionCache *AC,
                                     const Instruction *CtxI,
                                     const DominatorTree *DT, unsigned Depth) {
  return ::isGuaranteedNotToBeUndefOrPoison(V, AC, CtxI, DT, Depth, true);
}

OverflowResult llvm::computeOverflowForSignedAdd(const AddOperator *Add,
                                                 const DataLayout &DL,
                                                 AssumptionCache *AC,
                                                 const Instruction *CxtI,
                                                 const DominatorTree *DT) {
  return ::computeOverflowForSignedAdd(Add->getOperand(0), Add->getOperand(1),
                                       Add, DL, AC, CxtI, DT);
}

OverflowResult llvm::computeOverflowForSignedAdd(const Value *LHS,
                                                 const Value *RHS,
                                                 const DataLayout &DL,
                                                 AssumptionCache *AC,
                                                 const Instruction *CxtI,
                                                 const DominatorTree *DT) {
  return ::computeOverflowForSignedAdd(LHS, RHS, nullptr, DL, AC, CxtI, DT);
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(const Instruction *I) {
  // Note: An atomic operation isn't guaranteed to return in a reasonable amount
  // of time because it's possible for another thread to interfere with it for an
  // arbitrary length of time, but programs aren't allowed to rely on that.

  // If there is no successor, then execution can't transfer to it.
  if (isa<ReturnInst>(I))
    return false;
  if (isa<UnreachableInst>(I))
    return false;

  // Note: Do not add new checks here; instead, change Instruction::mayThrow or
  // Instruction::willReturn.
  //
  // FIXME: Move this check into Instruction::willReturn.
  if (isa<CatchPadInst>(I)) {
    switch (classifyEHPersonality(I->getFunction()->getPersonalityFn())) {
    default:
      // A catchpad may invoke exception object constructors and such, which
      // in some languages can be arbitrary code, so be conservative by default.
      return false;
    case EHPersonality::CoreCLR:
      // For CoreCLR, it just involves a type test.
      return true;
    }
  }

  // An instruction that returns without throwing must transfer control flow
  // to a successor.
  return !I->mayThrow() && I->willReturn();
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(const BasicBlock *BB) {
  // TODO: This is slightly conservative for invoke instruction since exiting
  // via an exception *is* normal control for them.
  for (const Instruction &I : *BB)
    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      return false;
  return true;
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(
   BasicBlock::const_iterator Begin, BasicBlock::const_iterator End,
   unsigned ScanLimit) {
  return isGuaranteedToTransferExecutionToSuccessor(make_range(Begin, End),
                                                    ScanLimit);
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(
   iterator_range<BasicBlock::const_iterator> Range, unsigned ScanLimit) {
  assert(ScanLimit && "scan limit must be non-zero");
  for (const Instruction &I : Range) {
    if (isa<DbgInfoIntrinsic>(I))
        continue;
    if (--ScanLimit == 0)
      return false;
    if (!isGuaranteedToTransferExecutionToSuccessor(&I))
      return false;
  }
  return true;
}

bool llvm::isGuaranteedToExecuteForEveryIteration(const Instruction *I,
                                                  const Loop *L) {
  // The loop header is guaranteed to be executed for every iteration.
  //
  // FIXME: Relax this constraint to cover all basic blocks that are
  // guaranteed to be executed at every iteration.
  if (I->getParent() != L->getHeader()) return false;

  for (const Instruction &LI : *L->getHeader()) {
    if (&LI == I) return true;
    if (!isGuaranteedToTransferExecutionToSuccessor(&LI)) return false;
  }
  llvm_unreachable("Instruction not contained in its own parent basic block.");
}

bool llvm::propagatesPoison(const Operator *I) {
  switch (I->getOpcode()) {
  case Instruction::Freeze:
  case Instruction::Select:
  case Instruction::PHI:
  case Instruction::Invoke:
    return false;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      // TODO: Add more intrinsics.
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::smul_with_overflow:
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::usub_with_overflow:
      case Intrinsic::umul_with_overflow:
        // If an input is a vector containing a poison element, the
        // two output vectors (calculated results, overflow bits)'
        // corresponding lanes are poison.
        return true;
      case Intrinsic::ctpop:
        return true;
      }
    }
    return false;
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::GetElementPtr:
    return true;
  default:
    if (isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<CastInst>(I))
      return true;

    // Be conservative and return false.
    return false;
  }
}

void llvm::getGuaranteedWellDefinedOps(
    const Instruction *I, SmallPtrSetImpl<const Value *> &Operands) {
  switch (I->getOpcode()) {
    case Instruction::Store:
      Operands.insert(cast<StoreInst>(I)->getPointerOperand());
      break;

    case Instruction::Load:
      Operands.insert(cast<LoadInst>(I)->getPointerOperand());
      break;

    // Since dereferenceable attribute imply noundef, atomic operations
    // also implicitly have noundef pointers too
    case Instruction::AtomicCmpXchg:
      Operands.insert(cast<AtomicCmpXchgInst>(I)->getPointerOperand());
      break;

    case Instruction::AtomicRMW:
      Operands.insert(cast<AtomicRMWInst>(I)->getPointerOperand());
      break;

    case Instruction::Call:
    case Instruction::Invoke: {
      const CallBase *CB = cast<CallBase>(I);
      if (CB->isIndirectCall())
        Operands.insert(CB->getCalledOperand());
      for (unsigned i = 0; i < CB->arg_size(); ++i) {
        if (CB->paramHasAttr(i, Attribute::NoUndef) ||
            CB->paramHasAttr(i, Attribute::Dereferenceable))
          Operands.insert(CB->getArgOperand(i));
      }
      break;
    }
    case Instruction::Ret:
      if (I->getFunction()->hasRetAttribute(Attribute::NoUndef))
        Operands.insert(I->getOperand(0));
      break;
    default:
      break;
  }
}

void llvm::getGuaranteedNonPoisonOps(const Instruction *I,
                                     SmallPtrSetImpl<const Value *> &Operands) {
  getGuaranteedWellDefinedOps(I, Operands);
  switch (I->getOpcode()) {
  // Divisors of these operations are allowed to be partially undef.
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    Operands.insert(I->getOperand(1));
    break;
  case Instruction::Switch:
    if (BranchOnPoisonAsUB)
      Operands.insert(cast<SwitchInst>(I)->getCondition());
    break;
  case Instruction::Br: {
    auto *BR = cast<BranchInst>(I);
    if (BranchOnPoisonAsUB && BR->isConditional())
      Operands.insert(BR->getCondition());
    break;
  }
  default:
    break;
  }
}

bool llvm::mustTriggerUB(const Instruction *I,
                         const SmallSet<const Value *, 16>& KnownPoison) {
  SmallPtrSet<const Value *, 4> NonPoisonOps;
  getGuaranteedNonPoisonOps(I, NonPoisonOps);

  for (const auto *V : NonPoisonOps)
    if (KnownPoison.count(V))
      return true;

  return false;
}

static bool programUndefinedIfUndefOrPoison(const Value *V,
                                            bool PoisonOnly) {
  // We currently only look for uses of values within the same basic
  // block, as that makes it easier to guarantee that the uses will be
  // executed given that Inst is executed.
  //
  // FIXME: Expand this to consider uses beyond the same basic block. To do
  // this, look out for the distinction between post-dominance and strong
  // post-dominance.
  const BasicBlock *BB = nullptr;
  BasicBlock::const_iterator Begin;
  if (const auto *Inst = dyn_cast<Instruction>(V)) {
    BB = Inst->getParent();
    Begin = Inst->getIterator();
    Begin++;
  } else if (const auto *Arg = dyn_cast<Argument>(V)) {
    BB = &Arg->getParent()->getEntryBlock();
    Begin = BB->begin();
  } else {
    return false;
  }

  // Limit number of instructions we look at, to avoid scanning through large
  // blocks. The current limit is chosen arbitrarily.
  unsigned ScanLimit = 32;
  BasicBlock::const_iterator End = BB->end();

  if (!PoisonOnly) {
    // Since undef does not propagate eagerly, be conservative & just check
    // whether a value is directly passed to an instruction that must take
    // well-defined operands.

    for (auto &I : make_range(Begin, End)) {
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      if (--ScanLimit == 0)
        break;

      SmallPtrSet<const Value *, 4> WellDefinedOps;
      getGuaranteedWellDefinedOps(&I, WellDefinedOps);
      if (WellDefinedOps.contains(V))
        return true;

      if (!isGuaranteedToTransferExecutionToSuccessor(&I))
        break;
    }
    return false;
  }

  // Set of instructions that we have proved will yield poison if Inst
  // does.
  SmallSet<const Value *, 16> YieldsPoison;
  SmallSet<const BasicBlock *, 4> Visited;

  YieldsPoison.insert(V);
  auto Propagate = [&](const User *User) {
    if (propagatesPoison(cast<Operator>(User)))
      YieldsPoison.insert(User);
  };
  for_each(V->users(), Propagate);
  Visited.insert(BB);

  while (true) {
    for (auto &I : make_range(Begin, End)) {
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      if (--ScanLimit == 0)
        return false;
      if (mustTriggerUB(&I, YieldsPoison))
        return true;
      if (!isGuaranteedToTransferExecutionToSuccessor(&I))
        return false;

      // Mark poison that propagates from I through uses of I.
      if (YieldsPoison.count(&I))
        for_each(I.users(), Propagate);
    }

    BB = BB->getSingleSuccessor();
    if (!BB || !Visited.insert(BB).second)
      break;

    Begin = BB->getFirstNonPHI()->getIterator();
    End = BB->end();
  }
  return false;
}

bool llvm::programUndefinedIfUndefOrPoison(const Instruction *Inst) {
  return ::programUndefinedIfUndefOrPoison(Inst, false);
}

bool llvm::programUndefinedIfPoison(const Instruction *Inst) {
  return ::programUndefinedIfUndefOrPoison(Inst, true);
}

static bool isKnownNonNaN(const Value *V, FastMathFlags FMF) {
  if (FMF.noNaNs())
    return true;

  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isNaN();

  if (auto *C = dyn_cast<ConstantDataVector>(V)) {
    if (!C->getElementType()->isFloatingPointTy())
      return false;
    for (unsigned I = 0, E = C->getNumElements(); I < E; ++I) {
      if (C->getElementAsAPFloat(I).isNaN())
        return false;
    }
    return true;
  }

  if (isa<ConstantAggregateZero>(V))
    return true;

  return false;
}

static bool isKnownNonZero(const Value *V) {
  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isZero();

  if (auto *C = dyn_cast<ConstantDataVector>(V)) {
    if (!C->getElementType()->isFloatingPointTy())
      return false;
    for (unsigned I = 0, E = C->getNumElements(); I < E; ++I) {
      if (C->getElementAsAPFloat(I).isZero())
        return false;
    }
    return true;
  }

  return false;
}

/// Match clamp pattern for float types without care about NaNs or signed zeros.
/// Given non-min/max outer cmp/select from the clamp pattern this
/// function recognizes if it can be substitued by a "canonical" min/max
/// pattern.
static SelectPatternResult matchFastFloatClamp(CmpInst::Predicate Pred,
                                               Value *CmpLHS, Value *CmpRHS,
                                               Value *TrueVal, Value *FalseVal,
                                               Value *&LHS, Value *&RHS) {
  // Try to match
  //   X < C1 ? C1 : Min(X, C2) --> Max(C1, Min(X, C2))
  //   X > C1 ? C1 : Max(X, C2) --> Min(C1, Max(X, C2))
  // and return description of the outer Max/Min.

  // First, check if select has inverse order:
  if (CmpRHS == FalseVal) {
    std::swap(TrueVal, FalseVal);
    Pred = CmpInst::getInversePredicate(Pred);
  }

  // Assume success now. If there's no match, callers should not use these anyway.
  LHS = TrueVal;
  RHS = FalseVal;

  const APFloat *FC1;
  if (CmpRHS != TrueVal || !match(CmpRHS, m_APFloat(FC1)) || !FC1->isFinite())
    return {SPF_UNKNOWN, SPNB_NA, false};

  const APFloat *FC2;
  switch (Pred) {
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULT:
  case CmpInst::FCMP_ULE:
    if (match(FalseVal,
              m_CombineOr(m_OrdFMin(m_Specific(CmpLHS), m_APFloat(FC2)),
                          m_UnordFMin(m_Specific(CmpLHS), m_APFloat(FC2)))) &&
        *FC1 < *FC2)
      return {SPF_FMAXNUM, SPNB_RETURNS_ANY, false};
    break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGT:
  case CmpInst::FCMP_UGE:
    if (match(FalseVal,
              m_CombineOr(m_OrdFMax(m_Specific(CmpLHS), m_APFloat(FC2)),
                          m_UnordFMax(m_Specific(CmpLHS), m_APFloat(FC2)))) &&
        *FC1 > *FC2)
      return {SPF_FMINNUM, SPNB_RETURNS_ANY, false};
    break;
  default:
    break;
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// Recognize variations of:
///   CLAMP(v,l,h) ==> ((v) < (l) ? (l) : ((v) > (h) ? (h) : (v)))
static SelectPatternResult matchClamp(CmpInst::Predicate Pred,
                                      Value *CmpLHS, Value *CmpRHS,
                                      Value *TrueVal, Value *FalseVal) {
  // Swap the select operands and predicate to match the patterns below.
  if (CmpRHS != TrueVal) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(TrueVal, FalseVal);
  }
  const APInt *C1;
  if (CmpRHS == TrueVal && match(CmpRHS, m_APInt(C1))) {
    const APInt *C2;
    // (X <s C1) ? C1 : SMIN(X, C2) ==> SMAX(SMIN(X, C2), C1)
    if (match(FalseVal, m_SMin(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->slt(*C2) && Pred == CmpInst::ICMP_SLT)
      return {SPF_SMAX, SPNB_NA, false};

    // (X >s C1) ? C1 : SMAX(X, C2) ==> SMIN(SMAX(X, C2), C1)
    if (match(FalseVal, m_SMax(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->sgt(*C2) && Pred == CmpInst::ICMP_SGT)
      return {SPF_SMIN, SPNB_NA, false};

    // (X <u C1) ? C1 : UMIN(X, C2) ==> UMAX(UMIN(X, C2), C1)
    if (match(FalseVal, m_UMin(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->ult(*C2) && Pred == CmpInst::ICMP_ULT)
      return {SPF_UMAX, SPNB_NA, false};

    // (X >u C1) ? C1 : UMAX(X, C2) ==> UMIN(UMAX(X, C2), C1)
    if (match(FalseVal, m_UMax(m_Specific(CmpLHS), m_APInt(C2))) &&
        C1->ugt(*C2) && Pred == CmpInst::ICMP_UGT)
      return {SPF_UMIN, SPNB_NA, false};
  }
  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// Recognize variations of:
///   a < c ? min(a,b) : min(b,c) ==> min(min(a,b),min(b,c))
static SelectPatternResult matchMinMaxOfMinMax(CmpInst::Predicate Pred,
                                               Value *CmpLHS, Value *CmpRHS,
                                               Value *TVal, Value *FVal,
                                               unsigned Depth) {
  // TODO: Allow FP min/max with nnan/nsz.
  assert(CmpInst::isIntPredicate(Pred) && "Expected integer comparison");

  Value *A = nullptr, *B = nullptr;
  SelectPatternResult L = matchSelectPattern(TVal, A, B, nullptr, Depth + 1);
  if (!SelectPatternResult::isMinOrMax(L.Flavor))
    return {SPF_UNKNOWN, SPNB_NA, false};

  Value *C = nullptr, *D = nullptr;
  SelectPatternResult R = matchSelectPattern(FVal, C, D, nullptr, Depth + 1);
  if (L.Flavor != R.Flavor)
    return {SPF_UNKNOWN, SPNB_NA, false};

  // We have something like: x Pred y ? min(a, b) : min(c, d).
  // Try to match the compare to the min/max operations of the select operands.
  // First, make sure we have the right compare predicate.
  switch (L.Flavor) {
  case SPF_SMIN:
    if (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SGE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_SMAX:
    if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SLE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_SGE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_UMIN:
    if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  case SPF_UMAX:
    if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_ULE) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(CmpLHS, CmpRHS);
    }
    if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE)
      break;
    return {SPF_UNKNOWN, SPNB_NA, false};
  default:
    return {SPF_UNKNOWN, SPNB_NA, false};
  }

  // If there is a common operand in the already matched min/max and the other
  // min/max operands match the compare operands (either directly or inverted),
  // then this is min/max of the same flavor.

  // a pred c ? m(a, b) : m(c, b) --> m(m(a, b), m(c, b))
  // ~c pred ~a ? m(a, b) : m(c, b) --> m(m(a, b), m(c, b))
  if (D == B) {
    if ((CmpLHS == A && CmpRHS == C) || (match(C, m_Not(m_Specific(CmpLHS))) &&
                                         match(A, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // a pred d ? m(a, b) : m(b, d) --> m(m(a, b), m(b, d))
  // ~d pred ~a ? m(a, b) : m(b, d) --> m(m(a, b), m(b, d))
  if (C == B) {
    if ((CmpLHS == A && CmpRHS == D) || (match(D, m_Not(m_Specific(CmpLHS))) &&
                                         match(A, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // b pred c ? m(a, b) : m(c, a) --> m(m(a, b), m(c, a))
  // ~c pred ~b ? m(a, b) : m(c, a) --> m(m(a, b), m(c, a))
  if (D == A) {
    if ((CmpLHS == B && CmpRHS == C) || (match(C, m_Not(m_Specific(CmpLHS))) &&
                                         match(B, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }
  // b pred d ? m(a, b) : m(a, d) --> m(m(a, b), m(a, d))
  // ~d pred ~b ? m(a, b) : m(a, d) --> m(m(a, b), m(a, d))
  if (C == A) {
    if ((CmpLHS == B && CmpRHS == D) || (match(D, m_Not(m_Specific(CmpLHS))) &&
                                         match(B, m_Not(m_Specific(CmpRHS)))))
      return {L.Flavor, SPNB_NA, false};
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

/// If the input value is the result of a 'not' op, constant integer, or vector
/// splat of a constant integer, return the bitwise-not source value.
/// TODO: This could be extended to handle non-splat vector integer constants.
static Value *getNotValue(Value *V) {
  Value *NotV;
  if (match(V, m_Not(m_Value(NotV))))
    return NotV;

  const APInt *C;
  if (match(V, m_APInt(C)))
    return ConstantInt::get(V->getType(), ~(*C));

  return nullptr;
}

/// Match non-obvious integer minimum and maximum sequences.
static SelectPatternResult matchMinMax(CmpInst::Predicate Pred,
                                       Value *CmpLHS, Value *CmpRHS,
                                       Value *TrueVal, Value *FalseVal,
                                       Value *&LHS, Value *&RHS,
                                       unsigned Depth) {
  // Assume success. If there's no match, callers should not use these anyway.
  LHS = TrueVal;
  RHS = FalseVal;

  SelectPatternResult SPR = matchClamp(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal);
  if (SPR.Flavor != SelectPatternFlavor::SPF_UNKNOWN)
    return SPR;

  SPR = matchMinMaxOfMinMax(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, Depth);
  if (SPR.Flavor != SelectPatternFlavor::SPF_UNKNOWN)
    return SPR;

  // Look through 'not' ops to find disguised min/max.
  // (X > Y) ? ~X : ~Y ==> (~X < ~Y) ? ~X : ~Y ==> MIN(~X, ~Y)
  // (X < Y) ? ~X : ~Y ==> (~X > ~Y) ? ~X : ~Y ==> MAX(~X, ~Y)
  if (CmpLHS == getNotValue(TrueVal) && CmpRHS == getNotValue(FalseVal)) {
    switch (Pred) {
    case CmpInst::ICMP_SGT: return {SPF_SMIN, SPNB_NA, false};
    case CmpInst::ICMP_SLT: return {SPF_SMAX, SPNB_NA, false};
    case CmpInst::ICMP_UGT: return {SPF_UMIN, SPNB_NA, false};
    case CmpInst::ICMP_ULT: return {SPF_UMAX, SPNB_NA, false};
    default: break;
    }
  }

  // (X > Y) ? ~Y : ~X ==> (~X < ~Y) ? ~Y : ~X ==> MAX(~Y, ~X)
  // (X < Y) ? ~Y : ~X ==> (~X > ~Y) ? ~Y : ~X ==> MIN(~Y, ~X)
  if (CmpLHS == getNotValue(FalseVal) && CmpRHS == getNotValue(TrueVal)) {
    switch (Pred) {
    case CmpInst::ICMP_SGT: return {SPF_SMAX, SPNB_NA, false};
    case CmpInst::ICMP_SLT: return {SPF_SMIN, SPNB_NA, false};
    case CmpInst::ICMP_UGT: return {SPF_UMAX, SPNB_NA, false};
    case CmpInst::ICMP_ULT: return {SPF_UMIN, SPNB_NA, false};
    default: break;
    }
  }

  if (Pred != CmpInst::ICMP_SGT && Pred != CmpInst::ICMP_SLT)
    return {SPF_UNKNOWN, SPNB_NA, false};

  const APInt *C1;
  if (!match(CmpRHS, m_APInt(C1)))
    return {SPF_UNKNOWN, SPNB_NA, false};

  // An unsigned min/max can be written with a signed compare.
  const APInt *C2;
  if ((CmpLHS == TrueVal && match(FalseVal, m_APInt(C2))) ||
      (CmpLHS == FalseVal && match(TrueVal, m_APInt(C2)))) {
    // Is the sign bit set?
    // (X <s 0) ? X : MAXVAL ==> (X >u MAXVAL) ? X : MAXVAL ==> UMAX
    // (X <s 0) ? MAXVAL : X ==> (X >u MAXVAL) ? MAXVAL : X ==> UMIN
    if (Pred == CmpInst::ICMP_SLT && C1->isZero() && C2->isMaxSignedValue())
      return {CmpLHS == TrueVal ? SPF_UMAX : SPF_UMIN, SPNB_NA, false};

    // Is the sign bit clear?
    // (X >s -1) ? MINVAL : X ==> (X <u MINVAL) ? MINVAL : X ==> UMAX
    // (X >s -1) ? X : MINVAL ==> (X <u MINVAL) ? X : MINVAL ==> UMIN
    if (Pred == CmpInst::ICMP_SGT && C1->isAllOnes() && C2->isMinSignedValue())
      return {CmpLHS == FalseVal ? SPF_UMAX : SPF_UMIN, SPNB_NA, false};
  }

  return {SPF_UNKNOWN, SPNB_NA, false};
}

bool llvm::isKnownNegation(const Value *X, const Value *Y, bool NeedNSW) {
  assert(X && Y && "Invalid operand");

  // X = sub (0, Y) || X = sub nsw (0, Y)
  if ((!NeedNSW && match(X, m_Sub(m_ZeroInt(), m_Specific(Y)))) ||
      (NeedNSW && match(X, m_NSWSub(m_ZeroInt(), m_Specific(Y)))))
    return true;

  // Y = sub (0, X) || Y = sub nsw (0, X)
  if ((!NeedNSW && match(Y, m_Sub(m_ZeroInt(), m_Specific(X)))) ||
      (NeedNSW && match(Y, m_NSWSub(m_ZeroInt(), m_Specific(X)))))
    return true;

  // X = sub (A, B), Y = sub (B, A) || X = sub nsw (A, B), Y = sub nsw (B, A)
  Value *A, *B;
  return (!NeedNSW && (match(X, m_Sub(m_Value(A), m_Value(B))) &&
                        match(Y, m_Sub(m_Specific(B), m_Specific(A))))) ||
         (NeedNSW && (match(X, m_NSWSub(m_Value(A), m_Value(B))) &&
                       match(Y, m_NSWSub(m_Specific(B), m_Specific(A)))));
}

static SelectPatternResult matchSelectPattern(CmpInst::Predicate Pred,
                                              FastMathFlags FMF,
                                              Value *CmpLHS, Value *CmpRHS,
                                              Value *TrueVal, Value *FalseVal,
                                              Value *&LHS, Value *&RHS,
                                              unsigned Depth) {
  if (CmpInst::isFPPredicate(Pred)) {
    // IEEE-754 ignores the sign of 0.0 in comparisons. So if the select has one
    // 0.0 operand, set the compare's 0.0 operands to that same value for the
    // purpose of identifying min/max. Disregard vector constants with undefined
    // elements because those can not be back-propagated for analysis.
    Value *OutputZeroVal = nullptr;
    if (match(TrueVal, m_AnyZeroFP()) && !match(FalseVal, m_AnyZeroFP()) &&
        !cast<Constant>(TrueVal)->containsUndefOrPoisonElement())
      OutputZeroVal = TrueVal;
    else if (match(FalseVal, m_AnyZeroFP()) && !match(TrueVal, m_AnyZeroFP()) &&
             !cast<Constant>(FalseVal)->containsUndefOrPoisonElement())
      OutputZeroVal = FalseVal;

    if (OutputZeroVal) {
      if (match(CmpLHS, m_AnyZeroFP()))
        CmpLHS = OutputZeroVal;
      if (match(CmpRHS, m_AnyZeroFP()))
        CmpRHS = OutputZeroVal;
    }
  }

  LHS = CmpLHS;
  RHS = CmpRHS;

  // Signed zero may return inconsistent results between implementations.
  //  (0.0 <= -0.0) ? 0.0 : -0.0 // Returns 0.0
  //  minNum(0.0, -0.0)          // May return -0.0 or 0.0 (IEEE 754-2008 5.3.1)
  // Therefore, we behave conservatively and only proceed if at least one of the
  // operands is known to not be zero or if we don't care about signed zero.
  switch (Pred) {
  default: break;
  // FIXME: Include OGT/OLT/UGT/ULT.
  case CmpInst::FCMP_OGE: case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_UGE: case CmpInst::FCMP_ULE:
    if (!FMF.noSignedZeros() && !isKnownNonZero(CmpLHS) &&
        !isKnownNonZero(CmpRHS))
      return {SPF_UNKNOWN, SPNB_NA, false};
  }

  SelectPatternNaNBehavior NaNBehavior = SPNB_NA;
  bool Ordered = false;

  // When given one NaN and one non-NaN input:
  //   - maxnum/minnum (C99 fmaxf()/fminf()) return the non-NaN input.
  //   - A simple C99 (a < b ? a : b) construction will return 'b' (as the
  //     ordered comparison fails), which could be NaN or non-NaN.
  // so here we discover exactly what NaN behavior is required/accepted.
  if (CmpInst::isFPPredicate(Pred)) {
    bool LHSSafe = isKnownNonNaN(CmpLHS, FMF);
    bool RHSSafe = isKnownNonNaN(CmpRHS, FMF);

    if (LHSSafe && RHSSafe) {
      // Both operands are known non-NaN.
      NaNBehavior = SPNB_RETURNS_ANY;
    } else if (CmpInst::isOrdered(Pred)) {
      // An ordered comparison will return false when given a NaN, so it
      // returns the RHS.
      Ordered = true;
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then NaN will be returned.
        NaNBehavior = SPNB_RETURNS_NAN;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_OTHER;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    } else {
      Ordered = false;
      // An unordered comparison will return true when given a NaN, so it
      // returns the LHS.
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then non-NaN will be returned.
        NaNBehavior = SPNB_RETURNS_OTHER;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_NAN;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    }
  }

  if (TrueVal == CmpRHS && FalseVal == CmpLHS) {
    std::swap(CmpLHS, CmpRHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
    if (NaNBehavior == SPNB_RETURNS_NAN)
      NaNBehavior = SPNB_RETURNS_OTHER;
    else if (NaNBehavior == SPNB_RETURNS_OTHER)
      NaNBehavior = SPNB_RETURNS_NAN;
    Ordered = !Ordered;
  }

  // ([if]cmp X, Y) ? X : Y
  if (TrueVal == CmpLHS && FalseVal == CmpRHS) {
    switch (Pred) {
    default: return {SPF_UNKNOWN, SPNB_NA, false}; // Equality.
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return {SPF_UMAX, SPNB_NA, false};
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return {SPF_SMAX, SPNB_NA, false};
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE: return {SPF_UMIN, SPNB_NA, false};
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: return {SPF_SMIN, SPNB_NA, false};
    case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_OGT:
    case FCmpInst::FCMP_OGE: return {SPF_FMAXNUM, NaNBehavior, Ordered};
    case FCmpInst::FCMP_ULT:
    case FCmpInst::FCMP_ULE:
    case FCmpInst::FCMP_OLT:
    case FCmpInst::FCMP_OLE: return {SPF_FMINNUM, NaNBehavior, Ordered};
    }
  }

  if (isKnownNegation(TrueVal, FalseVal)) {
    // Sign-extending LHS does not change its sign, so TrueVal/FalseVal can
    // match against either LHS or sext(LHS).
    auto MaybeSExtCmpLHS =
        m_CombineOr(m_Specific(CmpLHS), m_SExt(m_Specific(CmpLHS)));
    auto ZeroOrAllOnes = m_CombineOr(m_ZeroInt(), m_AllOnes());
    auto ZeroOrOne = m_CombineOr(m_ZeroInt(), m_One());
    if (match(TrueVal, MaybeSExtCmpLHS)) {
      // Set the return values. If the compare uses the negated value (-X >s 0),
      // swap the return values because the negated value is always 'RHS'.
      LHS = TrueVal;
      RHS = FalseVal;
      if (match(CmpLHS, m_Neg(m_Specific(FalseVal))))
        std::swap(LHS, RHS);

      // (X >s 0) ? X : -X or (X >s -1) ? X : -X --> ABS(X)
      // (-X >s 0) ? -X : X or (-X >s -1) ? -X : X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, ZeroOrAllOnes))
        return {SPF_ABS, SPNB_NA, false};

      // (X >=s 0) ? X : -X or (X >=s 1) ? X : -X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SGE && match(CmpRHS, ZeroOrOne))
        return {SPF_ABS, SPNB_NA, false};

      // (X <s 0) ? X : -X or (X <s 1) ? X : -X --> NABS(X)
      // (-X <s 0) ? -X : X or (-X <s 1) ? -X : X --> NABS(X)
      if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, ZeroOrOne))
        return {SPF_NABS, SPNB_NA, false};
    }
    else if (match(FalseVal, MaybeSExtCmpLHS)) {
      // Set the return values. If the compare uses the negated value (-X >s 0),
      // swap the return values because the negated value is always 'RHS'.
      LHS = FalseVal;
      RHS = TrueVal;
      if (match(CmpLHS, m_Neg(m_Specific(TrueVal))))
        std::swap(LHS, RHS);

      // (X >s 0) ? -X : X or (X >s -1) ? -X : X --> NABS(X)
      // (-X >s 0) ? X : -X or (-X >s -1) ? X : -X --> NABS(X)
      if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, ZeroOrAllOnes))
        return {SPF_NABS, SPNB_NA, false};

      // (X <s 0) ? -X : X or (X <s 1) ? -X : X --> ABS(X)
      // (-X <s 0) ? X : -X or (-X <s 1) ? X : -X --> ABS(X)
      if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, ZeroOrOne))
        return {SPF_ABS, SPNB_NA, false};
    }
  }

  if (CmpInst::isIntPredicate(Pred))
    return matchMinMax(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, LHS, RHS, Depth);

  // According to (IEEE 754-2008 5.3.1), minNum(0.0, -0.0) and similar
  // may return either -0.0 or 0.0, so fcmp/select pair has stricter
  // semantics than minNum. Be conservative in such case.
  if (NaNBehavior != SPNB_RETURNS_ANY ||
      (!FMF.noSignedZeros() && !isKnownNonZero(CmpLHS) &&
       !isKnownNonZero(CmpRHS)))
    return {SPF_UNKNOWN, SPNB_NA, false};

  return matchFastFloatClamp(Pred, CmpLHS, CmpRHS, TrueVal, FalseVal, LHS, RHS);
}

/// Helps to match a select pattern in case of a type mismatch.
///
/// The function processes the case when type of true and false values of a
/// select instruction differs from type of the cmp instruction operands because
/// of a cast instruction. The function checks if it is legal to move the cast
/// operation after "select". If yes, it returns the new second value of
/// "select" (with the assumption that cast is moved):
/// 1. As operand of cast instruction when both values of "select" are same cast
/// instructions.
/// 2. As restored constant (by applying reverse cast operation) when the first
/// value of the "select" is a cast operation and the second value is a
/// constant.
/// NOTE: We return only the new second value because the first value could be
/// accessed as operand of cast instruction.
static Value *lookThroughCast(CmpInst *CmpI, Value *V1, Value *V2,
                              Instruction::CastOps *CastOp) {
  auto *Cast1 = dyn_cast<CastInst>(V1);
  if (!Cast1)
    return nullptr;

  *CastOp = Cast1->getOpcode();
  Type *SrcTy = Cast1->getSrcTy();
  if (auto *Cast2 = dyn_cast<CastInst>(V2)) {
    // If V1 and V2 are both the same cast from the same type, look through V1.
    if (*CastOp == Cast2->getOpcode() && SrcTy == Cast2->getSrcTy())
      return Cast2->getOperand(0);
    return nullptr;
  }

  auto *C = dyn_cast<Constant>(V2);
  if (!C)
    return nullptr;

  Constant *CastedTo = nullptr;
  switch (*CastOp) {
  case Instruction::ZExt:
    if (CmpI->isUnsigned())
      CastedTo = ConstantExpr::getTrunc(C, SrcTy);
    break;
  case Instruction::SExt:
    if (CmpI->isSigned())
      CastedTo = ConstantExpr::getTrunc(C, SrcTy, true);
    break;
  case Instruction::Trunc:
    Constant *CmpConst;
    if (match(CmpI->getOperand(1), m_Constant(CmpConst)) &&
        CmpConst->getType() == SrcTy) {
      // Here we have the following case:
      //
      //   %cond = cmp iN %x, CmpConst
      //   %tr = trunc iN %x to iK
      //   %narrowsel = select i1 %cond, iK %t, iK C
      //
      // We can always move trunc after select operation:
      //
      //   %cond = cmp iN %x, CmpConst
      //   %widesel = select i1 %cond, iN %x, iN CmpConst
      //   %tr = trunc iN %widesel to iK
      //
      // Note that C could be extended in any way because we don't care about
      // upper bits after truncation. It can't be abs pattern, because it would
      // look like:
      //
      //   select i1 %cond, x, -x.
      //
      // So only min/max pattern could be matched. Such match requires widened C
      // == CmpConst. That is why set widened C = CmpConst, condition trunc
      // CmpConst == C is checked below.
      CastedTo = CmpConst;
    } else {
      CastedTo = ConstantExpr::getIntegerCast(C, SrcTy, CmpI->isSigned());
    }
    break;
  case Instruction::FPTrunc:
    CastedTo = ConstantExpr::getFPExtend(C, SrcTy, true);
    break;
  case Instruction::FPExt:
    CastedTo = ConstantExpr::getFPTrunc(C, SrcTy, true);
    break;
  case Instruction::FPToUI:
    CastedTo = ConstantExpr::getUIToFP(C, SrcTy, true);
    break;
  case Instruction::FPToSI:
    CastedTo = ConstantExpr::getSIToFP(C, SrcTy, true);
    break;
  case Instruction::UIToFP:
    CastedTo = ConstantExpr::getFPToUI(C, SrcTy, true);
    break;
  case Instruction::SIToFP:
    CastedTo = ConstantExpr::getFPToSI(C, SrcTy, true);
    break;
  default:
    break;
  }

  if (!CastedTo)
    return nullptr;

  // Make sure the cast doesn't lose any information.
  Constant *CastedBack =
      ConstantExpr::getCast(*CastOp, CastedTo, C->getType(), true);
  if (CastedBack != C)
    return nullptr;

  return CastedTo;
}

SelectPatternResult llvm::matchSelectPattern(Value *V, Value *&LHS, Value *&RHS,
                                             Instruction::CastOps *CastOp,
                                             unsigned Depth) {
  if (Depth >= MaxAnalysisRecursionDepth)
    return {SPF_UNKNOWN, SPNB_NA, false};

  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI) return {SPF_UNKNOWN, SPNB_NA, false};

  CmpInst *CmpI = dyn_cast<CmpInst>(SI->getCondition());
  if (!CmpI) return {SPF_UNKNOWN, SPNB_NA, false};

  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();

  return llvm::matchDecomposedSelectPattern(CmpI, TrueVal, FalseVal, LHS, RHS,
                                            CastOp, Depth);
}

SelectPatternResult llvm::matchDecomposedSelectPattern(
    CmpInst *CmpI, Value *TrueVal, Value *FalseVal, Value *&LHS, Value *&RHS,
    Instruction::CastOps *CastOp, unsigned Depth) {
  CmpInst::Predicate Pred = CmpI->getPredicate();
  Value *CmpLHS = CmpI->getOperand(0);
  Value *CmpRHS = CmpI->getOperand(1);
  FastMathFlags FMF;
  if (isa<FPMathOperator>(CmpI))
    FMF = CmpI->getFastMathFlags();

  // Bail out early.
  if (CmpI->isEquality())
    return {SPF_UNKNOWN, SPNB_NA, false};

  // Deal with type mismatches.
  if (CastOp && CmpLHS->getType() != TrueVal->getType()) {
    if (Value *C = lookThroughCast(CmpI, TrueVal, FalseVal, CastOp)) {
      // If this is a potential fmin/fmax with a cast to integer, then ignore
      // -0.0 because there is no corresponding integer value.
      if (*CastOp == Instruction::FPToSI || *CastOp == Instruction::FPToUI)
        FMF.setNoSignedZeros();
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS,
                                  cast<CastInst>(TrueVal)->getOperand(0), C,
                                  LHS, RHS, Depth);
    }
    if (Value *C = lookThroughCast(CmpI, FalseVal, TrueVal, CastOp)) {
      // If this is a potential fmin/fmax with a cast to integer, then ignore
      // -0.0 because there is no corresponding integer value.
      if (*CastOp == Instruction::FPToSI || *CastOp == Instruction::FPToUI)
        FMF.setNoSignedZeros();
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS,
                                  C, cast<CastInst>(FalseVal)->getOperand(0),
                                  LHS, RHS, Depth);
    }
  }
  return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS, TrueVal, FalseVal,
                              LHS, RHS, Depth);
}

CmpInst::Predicate llvm::getMinMaxPred(SelectPatternFlavor SPF, bool Ordered) {
  if (SPF == SPF_SMIN) return ICmpInst::ICMP_SLT;
  if (SPF == SPF_UMIN) return ICmpInst::ICMP_ULT;
  if (SPF == SPF_SMAX) return ICmpInst::ICMP_SGT;
  if (SPF == SPF_UMAX) return ICmpInst::ICMP_UGT;
  if (SPF == SPF_FMINNUM)
    return Ordered ? FCmpInst::FCMP_OLT : FCmpInst::FCMP_ULT;
  if (SPF == SPF_FMAXNUM)
    return Ordered ? FCmpInst::FCMP_OGT : FCmpInst::FCMP_UGT;
  llvm_unreachable("unhandled!");
}

SelectPatternFlavor llvm::getInverseMinMaxFlavor(SelectPatternFlavor SPF) {
  if (SPF == SPF_SMIN) return SPF_SMAX;
  if (SPF == SPF_UMIN) return SPF_UMAX;
  if (SPF == SPF_SMAX) return SPF_SMIN;
  if (SPF == SPF_UMAX) return SPF_UMIN;
  llvm_unreachable("unhandled!");
}

Intrinsic::ID llvm::getInverseMinMaxIntrinsic(Intrinsic::ID MinMaxID) {
  switch (MinMaxID) {
  case Intrinsic::smax: return Intrinsic::smin;
  case Intrinsic::smin: return Intrinsic::smax;
  case Intrinsic::umax: return Intrinsic::umin;
  case Intrinsic::umin: return Intrinsic::umax;
  default: llvm_unreachable("Unexpected intrinsic");
  }
}

CmpInst::Predicate llvm::getInverseMinMaxPred(SelectPatternFlavor SPF) {
  return getMinMaxPred(getInverseMinMaxFlavor(SPF));
}

APInt llvm::getMinMaxLimit(SelectPatternFlavor SPF, unsigned BitWidth) {
  switch (SPF) {
  case SPF_SMAX: return APInt::getSignedMaxValue(BitWidth);
  case SPF_SMIN: return APInt::getSignedMinValue(BitWidth);
  case SPF_UMAX: return APInt::getMaxValue(BitWidth);
  case SPF_UMIN: return APInt::getMinValue(BitWidth);
  default: llvm_unreachable("Unexpected flavor");
  }
}

std::pair<Intrinsic::ID, bool>
llvm::canConvertToMinOrMaxIntrinsic(ArrayRef<Value *> VL) {
  // Check if VL contains select instructions that can be folded into a min/max
  // vector intrinsic and return the intrinsic if it is possible.
  // TODO: Support floating point min/max.
  bool AllCmpSingleUse = true;
  SelectPatternResult SelectPattern;
  SelectPattern.Flavor = SPF_UNKNOWN;
  if (all_of(VL, [&SelectPattern, &AllCmpSingleUse](Value *I) {
        Value *LHS, *RHS;
        auto CurrentPattern = matchSelectPattern(I, LHS, RHS);
        if (!SelectPatternResult::isMinOrMax(CurrentPattern.Flavor) ||
            CurrentPattern.Flavor == SPF_FMINNUM ||
            CurrentPattern.Flavor == SPF_FMAXNUM ||
            !I->getType()->isIntOrIntVectorTy())
          return false;
        if (SelectPattern.Flavor != SPF_UNKNOWN &&
            SelectPattern.Flavor != CurrentPattern.Flavor)
          return false;
        SelectPattern = CurrentPattern;
        AllCmpSingleUse &=
            match(I, m_Select(m_OneUse(m_Value()), m_Value(), m_Value()));
        return true;
      })) {
    switch (SelectPattern.Flavor) {
    case SPF_SMIN:
      return {Intrinsic::smin, AllCmpSingleUse};
    case SPF_UMIN:
      return {Intrinsic::umin, AllCmpSingleUse};
    case SPF_SMAX:
      return {Intrinsic::smax, AllCmpSingleUse};
    case SPF_UMAX:
      return {Intrinsic::umax, AllCmpSingleUse};
    default:
      llvm_unreachable("unexpected select pattern flavor");
    }
  }
  return {Intrinsic::not_intrinsic, false};
}

bool llvm::matchSimpleRecurrence(const PHINode *P, BinaryOperator *&BO,
                                 Value *&Start, Value *&Step) {
  // Handle the case of a simple two-predecessor recurrence PHI.
  // There's a lot more that could theoretically be done here, but
  // this is sufficient to catch some interesting cases.
  if (P->getNumIncomingValues() != 2)
    return false;

  for (unsigned i = 0; i != 2; ++i) {
    Value *L = P->getIncomingValue(i);
    Value *R = P->getIncomingValue(!i);
    Operator *LU = dyn_cast<Operator>(L);
    if (!LU)
      continue;
    unsigned Opcode = LU->getOpcode();

    switch (Opcode) {
    default:
      continue;
    // TODO: Expand list -- xor, div, gep, uaddo, etc..
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Shl:
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Mul: {
      Value *LL = LU->getOperand(0);
      Value *LR = LU->getOperand(1);
      // Find a recurrence.
      if (LL == P)
        L = LR;
      else if (LR == P)
        L = LL;
      else
        continue; // Check for recurrence with L and R flipped.

      break; // Match!
    }
    };

    // We have matched a recurrence of the form:
    //   %iv = [R, %entry], [%iv.next, %backedge]
    //   %iv.next = binop %iv, L
    // OR
    //   %iv = [R, %entry], [%iv.next, %backedge]
    //   %iv.next = binop L, %iv
    BO = cast<BinaryOperator>(LU);
    Start = R;
    Step = L;
    return true;
  }
  return false;
}

bool llvm::matchSimpleRecurrence(const BinaryOperator *I, PHINode *&P,
                                 Value *&Start, Value *&Step) {
  BinaryOperator *BO = nullptr;
  P = dyn_cast<PHINode>(I->getOperand(0));
  if (!P)
    P = dyn_cast<PHINode>(I->getOperand(1));
  return P && matchSimpleRecurrence(P, BO, Start, Step) && BO == I;
}

/// Return true if "icmp Pred LHS RHS" is always true.
static bool isTruePredicate(CmpInst::Predicate Pred, const Value *LHS,
                            const Value *RHS, const DataLayout &DL,
                            unsigned Depth) {
  assert(!LHS->getType()->isVectorTy() && "TODO: extend to handle vectors!");
  if (ICmpInst::isTrueWhenEqual(Pred) && LHS == RHS)
    return true;

  switch (Pred) {
  default:
    return false;

  case CmpInst::ICMP_SLE: {
    const APInt *C;

    // LHS s<= LHS +_{nsw} C   if C >= 0
    if (match(RHS, m_NSWAdd(m_Specific(LHS), m_APInt(C))))
      return !C->isNegative();
    return false;
  }

  case CmpInst::ICMP_ULE: {
    const APInt *C;

    // LHS u<= LHS +_{nuw} C   for any C
    if (match(RHS, m_NUWAdd(m_Specific(LHS), m_APInt(C))))
      return true;

    // Match A to (X +_{nuw} CA) and B to (X +_{nuw} CB)
    auto MatchNUWAddsToSameValue = [&](const Value *A, const Value *B,
                                       const Value *&X,
                                       const APInt *&CA, const APInt *&CB) {
      if (match(A, m_NUWAdd(m_Value(X), m_APInt(CA))) &&
          match(B, m_NUWAdd(m_Specific(X), m_APInt(CB))))
        return true;

      // If X & C == 0 then (X | C) == X +_{nuw} C
      if (match(A, m_Or(m_Value(X), m_APInt(CA))) &&
          match(B, m_Or(m_Specific(X), m_APInt(CB)))) {
        KnownBits Known(CA->getBitWidth());
        computeKnownBits(X, Known, DL, Depth + 1, /*AC*/ nullptr,
                         /*CxtI*/ nullptr, /*DT*/ nullptr);
        if (CA->isSubsetOf(Known.Zero) && CB->isSubsetOf(Known.Zero))
          return true;
      }

      return false;
    };

    const Value *X;
    const APInt *CLHS, *CRHS;
    if (MatchNUWAddsToSameValue(LHS, RHS, X, CLHS, CRHS))
      return CLHS->ule(*CRHS);

    return false;
  }
  }
}

/// Return true if "icmp Pred BLHS BRHS" is true whenever "icmp Pred
/// ALHS ARHS" is true.  Otherwise, return None.
static Optional<bool>
isImpliedCondOperands(CmpInst::Predicate Pred, const Value *ALHS,
                      const Value *ARHS, const Value *BLHS, const Value *BRHS,
                      const DataLayout &DL, unsigned Depth) {
  switch (Pred) {
  default:
    return None;

  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    if (isTruePredicate(CmpInst::ICMP_SLE, BLHS, ALHS, DL, Depth) &&
        isTruePredicate(CmpInst::ICMP_SLE, ARHS, BRHS, DL, Depth))
      return true;
    return None;

  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
    if (isTruePredicate(CmpInst::ICMP_ULE, BLHS, ALHS, DL, Depth) &&
        isTruePredicate(CmpInst::ICMP_ULE, ARHS, BRHS, DL, Depth))
      return true;
    return None;
  }
}

/// Return true if the operands of the two compares match.  IsSwappedOps is true
/// when the operands match, but are swapped.
static bool isMatchingOps(const Value *ALHS, const Value *ARHS,
                          const Value *BLHS, const Value *BRHS,
                          bool &IsSwappedOps) {

  bool IsMatchingOps = (ALHS == BLHS && ARHS == BRHS);
  IsSwappedOps = (ALHS == BRHS && ARHS == BLHS);
  return IsMatchingOps || IsSwappedOps;
}

/// Return true if "icmp1 APred X, Y" implies "icmp2 BPred X, Y" is true.
/// Return false if "icmp1 APred X, Y" implies "icmp2 BPred X, Y" is false.
/// Otherwise, return None if we can't infer anything.
static Optional<bool> isImpliedCondMatchingOperands(CmpInst::Predicate APred,
                                                    CmpInst::Predicate BPred,
                                                    bool AreSwappedOps) {
  // Canonicalize the predicate as if the operands were not commuted.
  if (AreSwappedOps)
    BPred = ICmpInst::getSwappedPredicate(BPred);

  if (CmpInst::isImpliedTrueByMatchingCmp(APred, BPred))
    return true;
  if (CmpInst::isImpliedFalseByMatchingCmp(APred, BPred))
    return false;

  return None;
}

/// Return true if "icmp APred X, C1" implies "icmp BPred X, C2" is true.
/// Return false if "icmp APred X, C1" implies "icmp BPred X, C2" is false.
/// Otherwise, return None if we can't infer anything.
static Optional<bool>
isImpliedCondMatchingImmOperands(CmpInst::Predicate APred,
                                 const ConstantInt *C1,
                                 CmpInst::Predicate BPred,
                                 const ConstantInt *C2) {
  ConstantRange DomCR =
      ConstantRange::makeExactICmpRegion(APred, C1->getValue());
  ConstantRange CR = ConstantRange::makeExactICmpRegion(BPred, C2->getValue());
  ConstantRange Intersection = DomCR.intersectWith(CR);
  ConstantRange Difference = DomCR.difference(CR);
  if (Intersection.isEmptySet())
    return false;
  if (Difference.isEmptySet())
    return true;
  return None;
}

/// Return true if LHS implies RHS is true.  Return false if LHS implies RHS is
/// false.  Otherwise, return None if we can't infer anything.
static Optional<bool> isImpliedCondICmps(const ICmpInst *LHS,
                                         CmpInst::Predicate BPred,
                                         const Value *BLHS, const Value *BRHS,
                                         const DataLayout &DL, bool LHSIsTrue,
                                         unsigned Depth) {
  Value *ALHS = LHS->getOperand(0);
  Value *ARHS = LHS->getOperand(1);

  // The rest of the logic assumes the LHS condition is true.  If that's not the
  // case, invert the predicate to make it so.
  CmpInst::Predicate APred =
      LHSIsTrue ? LHS->getPredicate() : LHS->getInversePredicate();

  // Can we infer anything when the two compares have matching operands?
  bool AreSwappedOps;
  if (isMatchingOps(ALHS, ARHS, BLHS, BRHS, AreSwappedOps)) {
    if (Optional<bool> Implication = isImpliedCondMatchingOperands(
            APred, BPred, AreSwappedOps))
      return Implication;
    // No amount of additional analysis will infer the second condition, so
    // early exit.
    return None;
  }

  // Can we infer anything when the LHS operands match and the RHS operands are
  // constants (not necessarily matching)?
  if (ALHS == BLHS && isa<ConstantInt>(ARHS) && isa<ConstantInt>(BRHS)) {
    if (Optional<bool> Implication = isImpliedCondMatchingImmOperands(
            APred, cast<ConstantInt>(ARHS), BPred, cast<ConstantInt>(BRHS)))
      return Implication;
    // No amount of additional analysis will infer the second condition, so
    // early exit.
    return None;
  }

  if (APred == BPred)
    return isImpliedCondOperands(APred, ALHS, ARHS, BLHS, BRHS, DL, Depth);
  return None;
}

/// Return true if LHS implies RHS is true.  Return false if LHS implies RHS is
/// false.  Otherwise, return None if we can't infer anything.  We expect the
/// RHS to be an icmp and the LHS to be an 'and', 'or', or a 'select' instruction.
static Optional<bool>
isImpliedCondAndOr(const Instruction *LHS, CmpInst::Predicate RHSPred,
                   const Value *RHSOp0, const Value *RHSOp1,
                   const DataLayout &DL, bool LHSIsTrue, unsigned Depth) {
  // The LHS must be an 'or', 'and', or a 'select' instruction.
  assert((LHS->getOpcode() == Instruction::And ||
          LHS->getOpcode() == Instruction::Or ||
          LHS->getOpcode() == Instruction::Select) &&
         "Expected LHS to be 'and', 'or', or 'select'.");

  assert(Depth <= MaxAnalysisRecursionDepth && "Hit recursion limit");

  // If the result of an 'or' is false, then we know both legs of the 'or' are
  // false.  Similarly, if the result of an 'and' is true, then we know both
  // legs of the 'and' are true.
  const Value *ALHS, *ARHS;
  if ((!LHSIsTrue && match(LHS, m_LogicalOr(m_Value(ALHS), m_Value(ARHS)))) ||
      (LHSIsTrue && match(LHS, m_LogicalAnd(m_Value(ALHS), m_Value(ARHS))))) {
    // FIXME: Make this non-recursion.
    if (Optional<bool> Implication = isImpliedCondition(
            ALHS, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue, Depth + 1))
      return Implication;
    if (Optional<bool> Implication = isImpliedCondition(
            ARHS, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue, Depth + 1))
      return Implication;
    return None;
  }
  return None;
}

Optional<bool>
llvm::isImpliedCondition(const Value *LHS, CmpInst::Predicate RHSPred,
                         const Value *RHSOp0, const Value *RHSOp1,
                         const DataLayout &DL, bool LHSIsTrue, unsigned Depth) {
  // Bail out when we hit the limit.
  if (Depth == MaxAnalysisRecursionDepth)
    return None;

  // A mismatch occurs when we compare a scalar cmp to a vector cmp, for
  // example.
  if (RHSOp0->getType()->isVectorTy() != LHS->getType()->isVectorTy())
    return None;

  Type *OpTy = LHS->getType();
  assert(OpTy->isIntOrIntVectorTy(1) && "Expected integer type only!");

  // FIXME: Extending the code below to handle vectors.
  if (OpTy->isVectorTy())
    return None;

  assert(OpTy->isIntegerTy(1) && "implied by above");

  // Both LHS and RHS are icmps.
  const ICmpInst *LHSCmp = dyn_cast<ICmpInst>(LHS);
  if (LHSCmp)
    return isImpliedCondICmps(LHSCmp, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue,
                              Depth);

  /// The LHS should be an 'or', 'and', or a 'select' instruction.  We expect
  /// the RHS to be an icmp.
  /// FIXME: Add support for and/or/select on the RHS.
  if (const Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
    if ((LHSI->getOpcode() == Instruction::And ||
         LHSI->getOpcode() == Instruction::Or ||
         LHSI->getOpcode() == Instruction::Select))
      return isImpliedCondAndOr(LHSI, RHSPred, RHSOp0, RHSOp1, DL, LHSIsTrue,
                                Depth);
  }
  return None;
}

Optional<bool> llvm::isImpliedCondition(const Value *LHS, const Value *RHS,
                                        const DataLayout &DL, bool LHSIsTrue,
                                        unsigned Depth) {
  // LHS ==> RHS by definition
  if (LHS == RHS)
    return LHSIsTrue;

  if (const ICmpInst *RHSCmp = dyn_cast<ICmpInst>(RHS))
    return isImpliedCondition(LHS, RHSCmp->getPredicate(),
                              RHSCmp->getOperand(0), RHSCmp->getOperand(1), DL,
                              LHSIsTrue, Depth);

  if (Depth == MaxAnalysisRecursionDepth)
    return None;

  // LHS ==> (RHS1 || RHS2) if LHS ==> RHS1 or LHS ==> RHS2
  // LHS ==> !(RHS1 && RHS2) if LHS ==> !RHS1 or LHS ==> !RHS2
  const Value *RHS1, *RHS2;
  if (match(RHS, m_LogicalOr(m_Value(RHS1), m_Value(RHS2)))) {
    if (Optional<bool> Imp =
            isImpliedCondition(LHS, RHS1, DL, LHSIsTrue, Depth + 1))
      if (*Imp == true)
        return true;
    if (Optional<bool> Imp =
            isImpliedCondition(LHS, RHS2, DL, LHSIsTrue, Depth + 1))
      if (*Imp == true)
        return true;
  }
  if (match(RHS, m_LogicalAnd(m_Value(RHS1), m_Value(RHS2)))) {
    if (Optional<bool> Imp =
            isImpliedCondition(LHS, RHS1, DL, LHSIsTrue, Depth + 1))
      if (*Imp == false)
        return false;
    if (Optional<bool> Imp =
            isImpliedCondition(LHS, RHS2, DL, LHSIsTrue, Depth + 1))
      if (*Imp == false)
        return false;
  }

  return None;
}

// Returns a pair (Condition, ConditionIsTrue), where Condition is a branch
// condition dominating ContextI or nullptr, if no condition is found.
static std::pair<Value *, bool>
getDomPredecessorCondition(const Instruction *ContextI) {
  if (!ContextI || !ContextI->getParent())
    return {nullptr, false};

  // TODO: This is a poor/cheap way to determine dominance. Should we use a
  // dominator tree (eg, from a SimplifyQuery) instead?
  const BasicBlock *ContextBB = ContextI->getParent();
  const BasicBlock *PredBB = ContextBB->getSinglePredecessor();
  if (!PredBB)
    return {nullptr, false};

  // We need a conditional branch in the predecessor.
  Value *PredCond;
  BasicBlock *TrueBB, *FalseBB;
  if (!match(PredBB->getTerminator(), m_Br(m_Value(PredCond), TrueBB, FalseBB)))
    return {nullptr, false};

  // The branch should get simplified. Don't bother simplifying this condition.
  if (TrueBB == FalseBB)
    return {nullptr, false};

  assert((TrueBB == ContextBB || FalseBB == ContextBB) &&
         "Predecessor block does not point to successor?");

  // Is this condition implied by the predecessor condition?
  return {PredCond, TrueBB == ContextBB};
}

Optional<bool> llvm::isImpliedByDomCondition(const Value *Cond,
                                             const Instruction *ContextI,
                                             const DataLayout &DL) {
  assert(Cond->getType()->isIntOrIntVectorTy(1) && "Condition must be bool");
  auto PredCond = getDomPredecessorCondition(ContextI);
  if (PredCond.first)
    return isImpliedCondition(PredCond.first, Cond, DL, PredCond.second);
  return None;
}

Optional<bool> llvm::isImpliedByDomCondition(CmpInst::Predicate Pred,
                                             const Value *LHS, const Value *RHS,
                                             const Instruction *ContextI,
                                             const DataLayout &DL) {
  auto PredCond = getDomPredecessorCondition(ContextI);
  if (PredCond.first)
    return isImpliedCondition(PredCond.first, Pred, LHS, RHS, DL,
                              PredCond.second);
  return None;
}

static void setLimitsForBinOp(const BinaryOperator &BO, APInt &Lower,
                              APInt &Upper, const InstrInfoQuery &IIQ,
                              bool PreferSignedRange) {
  unsigned Width = Lower.getBitWidth();
  const APInt *C;
  switch (BO.getOpcode()) {
  case Instruction::Add:
    if (match(BO.getOperand(1), m_APInt(C)) && !C->isZero()) {
      bool HasNSW = IIQ.hasNoSignedWrap(&BO);
      bool HasNUW = IIQ.hasNoUnsignedWrap(&BO);

      // If the caller expects a signed compare, then try to use a signed range.
      // Otherwise if both no-wraps are set, use the unsigned range because it
      // is never larger than the signed range. Example:
      // "add nuw nsw i8 X, -2" is unsigned [254,255] vs. signed [-128, 125].
      if (PreferSignedRange && HasNSW && HasNUW)
        HasNUW = false;

      if (HasNUW) {
        // 'add nuw x, C' produces [C, UINT_MAX].
        Lower = *C;
      } else if (HasNSW) {
        if (C->isNegative()) {
          // 'add nsw x, -C' produces [SINT_MIN, SINT_MAX - C].
          Lower = APInt::getSignedMinValue(Width);
          Upper = APInt::getSignedMaxValue(Width) + *C + 1;
        } else {
          // 'add nsw x, +C' produces [SINT_MIN + C, SINT_MAX].
          Lower = APInt::getSignedMinValue(Width) + *C;
          Upper = APInt::getSignedMaxValue(Width) + 1;
        }
      }
    }
    break;

  case Instruction::And:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'and x, C' produces [0, C].
      Upper = *C + 1;
    break;

  case Instruction::Or:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'or x, C' produces [C, UINT_MAX].
      Lower = *C;
    break;

  case Instruction::AShr:
    if (match(BO.getOperand(1), m_APInt(C)) && C->ult(Width)) {
      // 'ashr x, C' produces [INT_MIN >> C, INT_MAX >> C].
      Lower = APInt::getSignedMinValue(Width).ashr(*C);
      Upper = APInt::getSignedMaxValue(Width).ashr(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      unsigned ShiftAmount = Width - 1;
      if (!C->isZero() && IIQ.isExact(&BO))
        ShiftAmount = C->countTrailingZeros();
      if (C->isNegative()) {
        // 'ashr C, x' produces [C, C >> (Width-1)]
        Lower = *C;
        Upper = C->ashr(ShiftAmount) + 1;
      } else {
        // 'ashr C, x' produces [C >> (Width-1), C]
        Lower = C->ashr(ShiftAmount);
        Upper = *C + 1;
      }
    }
    break;

  case Instruction::LShr:
    if (match(BO.getOperand(1), m_APInt(C)) && C->ult(Width)) {
      // 'lshr x, C' produces [0, UINT_MAX >> C].
      Upper = APInt::getAllOnes(Width).lshr(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      // 'lshr C, x' produces [C >> (Width-1), C].
      unsigned ShiftAmount = Width - 1;
      if (!C->isZero() && IIQ.isExact(&BO))
        ShiftAmount = C->countTrailingZeros();
      Lower = C->lshr(ShiftAmount);
      Upper = *C + 1;
    }
    break;

  case Instruction::Shl:
    if (match(BO.getOperand(0), m_APInt(C))) {
      if (IIQ.hasNoUnsignedWrap(&BO)) {
        // 'shl nuw C, x' produces [C, C << CLZ(C)]
        Lower = *C;
        Upper = Lower.shl(Lower.countLeadingZeros()) + 1;
      } else if (BO.hasNoSignedWrap()) { // TODO: What if both nuw+nsw?
        if (C->isNegative()) {
          // 'shl nsw C, x' produces [C << CLO(C)-1, C]
          unsigned ShiftAmount = C->countLeadingOnes() - 1;
          Lower = C->shl(ShiftAmount);
          Upper = *C + 1;
        } else {
          // 'shl nsw C, x' produces [C, C << CLZ(C)-1]
          unsigned ShiftAmount = C->countLeadingZeros() - 1;
          Lower = *C;
          Upper = C->shl(ShiftAmount) + 1;
        }
      }
    }
    break;

  case Instruction::SDiv:
    if (match(BO.getOperand(1), m_APInt(C))) {
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      if (C->isAllOnes()) {
        // 'sdiv x, -1' produces [INT_MIN + 1, INT_MAX]
        //    where C != -1 and C != 0 and C != 1
        Lower = IntMin + 1;
        Upper = IntMax + 1;
      } else if (C->countLeadingZeros() < Width - 1) {
        // 'sdiv x, C' produces [INT_MIN / C, INT_MAX / C]
        //    where C != -1 and C != 0 and C != 1
        Lower = IntMin.sdiv(*C);
        Upper = IntMax.sdiv(*C);
        if (Lower.sgt(Upper))
          std::swap(Lower, Upper);
        Upper = Upper + 1;
        assert(Upper != Lower && "Upper part of range has wrapped!");
      }
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      if (C->isMinSignedValue()) {
        // 'sdiv INT_MIN, x' produces [INT_MIN, INT_MIN / -2].
        Lower = *C;
        Upper = Lower.lshr(1) + 1;
      } else {
        // 'sdiv C, x' produces [-|C|, |C|].
        Upper = C->abs() + 1;
        Lower = (-Upper) + 1;
      }
    }
    break;

  case Instruction::UDiv:
    if (match(BO.getOperand(1), m_APInt(C)) && !C->isZero()) {
      // 'udiv x, C' produces [0, UINT_MAX / C].
      Upper = APInt::getMaxValue(Width).udiv(*C) + 1;
    } else if (match(BO.getOperand(0), m_APInt(C))) {
      // 'udiv C, x' produces [0, C].
      Upper = *C + 1;
    }
    break;

  case Instruction::SRem:
    if (match(BO.getOperand(1), m_APInt(C))) {
      // 'srem x, C' produces (-|C|, |C|).
      Upper = C->abs();
      Lower = (-Upper) + 1;
    }
    break;

  case Instruction::URem:
    if (match(BO.getOperand(1), m_APInt(C)))
      // 'urem x, C' produces [0, C).
      Upper = *C;
    break;

  default:
    break;
  }
}

static void setLimitsForIntrinsic(const IntrinsicInst &II, APInt &Lower,
                                  APInt &Upper) {
  unsigned Width = Lower.getBitWidth();
  const APInt *C;
  switch (II.getIntrinsicID()) {
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    // Maximum of set/clear bits is the bit width.
    assert(Lower == 0 && "Expected lower bound to be zero");
    Upper = Width + 1;
    break;
  case Intrinsic::uadd_sat:
    // uadd.sat(x, C) produces [C, UINT_MAX].
    if (match(II.getOperand(0), m_APInt(C)) ||
        match(II.getOperand(1), m_APInt(C)))
      Lower = *C;
    break;
  case Intrinsic::sadd_sat:
    if (match(II.getOperand(0), m_APInt(C)) ||
        match(II.getOperand(1), m_APInt(C))) {
      if (C->isNegative()) {
        // sadd.sat(x, -C) produces [SINT_MIN, SINT_MAX + (-C)].
        Lower = APInt::getSignedMinValue(Width);
        Upper = APInt::getSignedMaxValue(Width) + *C + 1;
      } else {
        // sadd.sat(x, +C) produces [SINT_MIN + C, SINT_MAX].
        Lower = APInt::getSignedMinValue(Width) + *C;
        Upper = APInt::getSignedMaxValue(Width) + 1;
      }
    }
    break;
  case Intrinsic::usub_sat:
    // usub.sat(C, x) produces [0, C].
    if (match(II.getOperand(0), m_APInt(C)))
      Upper = *C + 1;
    // usub.sat(x, C) produces [0, UINT_MAX - C].
    else if (match(II.getOperand(1), m_APInt(C)))
      Upper = APInt::getMaxValue(Width) - *C + 1;
    break;
  case Intrinsic::ssub_sat:
    if (match(II.getOperand(0), m_APInt(C))) {
      if (C->isNegative()) {
        // ssub.sat(-C, x) produces [SINT_MIN, -SINT_MIN + (-C)].
        Lower = APInt::getSignedMinValue(Width);
        Upper = *C - APInt::getSignedMinValue(Width) + 1;
      } else {
        // ssub.sat(+C, x) produces [-SINT_MAX + C, SINT_MAX].
        Lower = *C - APInt::getSignedMaxValue(Width);
        Upper = APInt::getSignedMaxValue(Width) + 1;
      }
    } else if (match(II.getOperand(1), m_APInt(C))) {
      if (C->isNegative()) {
        // ssub.sat(x, -C) produces [SINT_MIN - (-C), SINT_MAX]:
        Lower = APInt::getSignedMinValue(Width) - *C;
        Upper = APInt::getSignedMaxValue(Width) + 1;
      } else {
        // ssub.sat(x, +C) produces [SINT_MIN, SINT_MAX - C].
        Lower = APInt::getSignedMinValue(Width);
        Upper = APInt::getSignedMaxValue(Width) - *C + 1;
      }
    }
    break;
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax:
    if (!match(II.getOperand(0), m_APInt(C)) &&
        !match(II.getOperand(1), m_APInt(C)))
      break;

    switch (II.getIntrinsicID()) {
    case Intrinsic::umin:
      Upper = *C + 1;
      break;
    case Intrinsic::umax:
      Lower = *C;
      break;
    case Intrinsic::smin:
      Lower = APInt::getSignedMinValue(Width);
      Upper = *C + 1;
      break;
    case Intrinsic::smax:
      Lower = *C;
      Upper = APInt::getSignedMaxValue(Width) + 1;
      break;
    default:
      llvm_unreachable("Must be min/max intrinsic");
    }
    break;
  case Intrinsic::abs:
    // If abs of SIGNED_MIN is poison, then the result is [0..SIGNED_MAX],
    // otherwise it is [0..SIGNED_MIN], as -SIGNED_MIN == SIGNED_MIN.
    if (match(II.getOperand(1), m_One()))
      Upper = APInt::getSignedMaxValue(Width) + 1;
    else
      Upper = APInt::getSignedMinValue(Width) + 1;
    break;
  default:
    break;
  }
}

static void setLimitsForSelectPattern(const SelectInst &SI, APInt &Lower,
                                      APInt &Upper, const InstrInfoQuery &IIQ) {
  const Value *LHS = nullptr, *RHS = nullptr;
  SelectPatternResult R = matchSelectPattern(&SI, LHS, RHS);
  if (R.Flavor == SPF_UNKNOWN)
    return;

  unsigned BitWidth = SI.getType()->getScalarSizeInBits();

  if (R.Flavor == SelectPatternFlavor::SPF_ABS) {
    // If the negation part of the abs (in RHS) has the NSW flag,
    // then the result of abs(X) is [0..SIGNED_MAX],
    // otherwise it is [0..SIGNED_MIN], as -SIGNED_MIN == SIGNED_MIN.
    Lower = APInt::getZero(BitWidth);
    if (match(RHS, m_Neg(m_Specific(LHS))) &&
        IIQ.hasNoSignedWrap(cast<Instruction>(RHS)))
      Upper = APInt::getSignedMaxValue(BitWidth) + 1;
    else
      Upper = APInt::getSignedMinValue(BitWidth) + 1;
    return;
  }

  if (R.Flavor == SelectPatternFlavor::SPF_NABS) {
    // The result of -abs(X) is <= 0.
    Lower = APInt::getSignedMinValue(BitWidth);
    Upper = APInt(BitWidth, 1);
    return;
  }

  const APInt *C;
  if (!match(LHS, m_APInt(C)) && !match(RHS, m_APInt(C)))
    return;

  switch (R.Flavor) {
    case SPF_UMIN:
      Upper = *C + 1;
      break;
    case SPF_UMAX:
      Lower = *C;
      break;
    case SPF_SMIN:
      Lower = APInt::getSignedMinValue(BitWidth);
      Upper = *C + 1;
      break;
    case SPF_SMAX:
      Lower = *C;
      Upper = APInt::getSignedMaxValue(BitWidth) + 1;
      break;
    default:
      break;
  }
}

static void setLimitForFPToI(const Instruction *I, APInt &Lower, APInt &Upper) {
  // The maximum representable value of a half is 65504. For floats the maximum
  // value is 3.4e38 which requires roughly 129 bits.
  unsigned BitWidth = I->getType()->getScalarSizeInBits();
  if (!I->getOperand(0)->getType()->getScalarType()->isHalfTy())
    return;
  if (isa<FPToSIInst>(I) && BitWidth >= 17) {
    Lower = APInt(BitWidth, -65504);
    Upper = APInt(BitWidth, 65505);
  }

  if (isa<FPToUIInst>(I) && BitWidth >= 16) {
    // For a fptoui the lower limit is left as 0.
    Upper = APInt(BitWidth, 65505);
  }
}

ConstantRange llvm::computeConstantRange(const Value *V, bool ForSigned,
                                         bool UseInstrInfo, AssumptionCache *AC,
                                         const Instruction *CtxI,
                                         const DominatorTree *DT,
                                         unsigned Depth) {
  assert(V->getType()->isIntOrIntVectorTy() && "Expected integer instruction");

  if (Depth == MaxAnalysisRecursionDepth)
    return ConstantRange::getFull(V->getType()->getScalarSizeInBits());

  const APInt *C;
  if (match(V, m_APInt(C)))
    return ConstantRange(*C);

  InstrInfoQuery IIQ(UseInstrInfo);
  unsigned BitWidth = V->getType()->getScalarSizeInBits();
  APInt Lower = APInt(BitWidth, 0);
  APInt Upper = APInt(BitWidth, 0);
  if (auto *BO = dyn_cast<BinaryOperator>(V))
    setLimitsForBinOp(*BO, Lower, Upper, IIQ, ForSigned);
  else if (auto *II = dyn_cast<IntrinsicInst>(V))
    setLimitsForIntrinsic(*II, Lower, Upper);
  else if (auto *SI = dyn_cast<SelectInst>(V))
    setLimitsForSelectPattern(*SI, Lower, Upper, IIQ);
  else if (isa<FPToUIInst>(V) || isa<FPToSIInst>(V))
    setLimitForFPToI(cast<Instruction>(V), Lower, Upper);

  ConstantRange CR = ConstantRange::getNonEmpty(Lower, Upper);

  if (auto *I = dyn_cast<Instruction>(V))
    if (auto *Range = IIQ.getMetadata(I, LLVMContext::MD_range))
      CR = CR.intersectWith(getConstantRangeFromMetadata(*Range));

  if (CtxI && AC) {
    // Try to restrict the range based on information from assumptions.
    for (auto &AssumeVH : AC->assumptionsFor(V)) {
      if (!AssumeVH)
        continue;
      CallInst *I = cast<CallInst>(AssumeVH);
      assert(I->getParent()->getParent() == CtxI->getParent()->getParent() &&
             "Got assumption for the wrong function!");
      assert(I->getCalledFunction()->getIntrinsicID() == Intrinsic::assume &&
             "must be an assume intrinsic");

      if (!isValidAssumeForContext(I, CtxI, DT))
        continue;
      Value *Arg = I->getArgOperand(0);
      ICmpInst *Cmp = dyn_cast<ICmpInst>(Arg);
      // Currently we just use information from comparisons.
      if (!Cmp || Cmp->getOperand(0) != V)
        continue;
      // TODO: Set "ForSigned" parameter via Cmp->isSigned()?
      ConstantRange RHS =
          computeConstantRange(Cmp->getOperand(1), /* ForSigned */ false,
                               UseInstrInfo, AC, I, DT, Depth + 1);
      CR = CR.intersectWith(
          ConstantRange::makeAllowedICmpRegion(Cmp->getPredicate(), RHS));
    }
  }

  return CR;
}

static Optional<int64_t>
getOffsetFromIndex(const GEPOperator *GEP, unsigned Idx, const DataLayout &DL) {
  // Skip over the first indices.
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1; i != Idx; ++i, ++GTI)
    /*skip along*/;

  // Compute the offset implied by the rest of the indices.
  int64_t Offset = 0;
  for (unsigned i = Idx, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!OpC)
      return None;
    if (OpC->isZero())
      continue; // No offset.

    // Handle struct indices, which add their field offset to the pointer.
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      Offset += DL.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
      continue;
    }

    // Otherwise, we have a sequential type like an array or fixed-length
    // vector. Multiply the index by the ElementSize.
    TypeSize Size = DL.getTypeAllocSize(GTI.getIndexedType());
    if (Size.isScalable())
      return None;
    Offset += Size.getFixedSize() * OpC->getSExtValue();
  }

  return Offset;
}

Optional<int64_t> llvm::isPointerOffset(const Value *Ptr1, const Value *Ptr2,
                                        const DataLayout &DL) {
  APInt Offset1(DL.getIndexTypeSizeInBits(Ptr1->getType()), 0);
  APInt Offset2(DL.getIndexTypeSizeInBits(Ptr2->getType()), 0);
  Ptr1 = Ptr1->stripAndAccumulateConstantOffsets(DL, Offset1, true);
  Ptr2 = Ptr2->stripAndAccumulateConstantOffsets(DL, Offset2, true);

  // Handle the trivial case first.
  if (Ptr1 == Ptr2)
    return Offset2.getSExtValue() - Offset1.getSExtValue();

  const GEPOperator *GEP1 = dyn_cast<GEPOperator>(Ptr1);
  const GEPOperator *GEP2 = dyn_cast<GEPOperator>(Ptr2);

  // Right now we handle the case when Ptr1/Ptr2 are both GEPs with an identical
  // base.  After that base, they may have some number of common (and
  // potentially variable) indices.  After that they handle some constant
  // offset, which determines their offset from each other.  At this point, we
  // handle no other case.
  if (!GEP1 || !GEP2 || GEP1->getOperand(0) != GEP2->getOperand(0) ||
      GEP1->getSourceElementType() != GEP2->getSourceElementType())
    return None;

  // Skip any common indices and track the GEP types.
  unsigned Idx = 1;
  for (; Idx != GEP1->getNumOperands() && Idx != GEP2->getNumOperands(); ++Idx)
    if (GEP1->getOperand(Idx) != GEP2->getOperand(Idx))
      break;

  auto IOffset1 = getOffsetFromIndex(GEP1, Idx, DL);
  auto IOffset2 = getOffsetFromIndex(GEP2, Idx, DL);
  if (!IOffset1 || !IOffset2)
    return None;
  return *IOffset2 - *IOffset1 + Offset2.getSExtValue() -
         Offset1.getSExtValue();
}
