//===- KnownBitsAnalysis.cpp - An analysis that caches KnownBits ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/KnownBitsAnalysis.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/ValueTrackingHelper.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace llvm;
using namespace vthelper;
using namespace PatternMatch;
using namespace std::placeholders;

template <typename NodeRef, typename ChildIteratorType>
struct NodeGraphTraitsBase {
  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { // NOLINT
    return N->user_begin();
  }
  static ChildIteratorType child_end(NodeRef N) { // NOLINT
    return N->user_end();
  }
};

template <>
struct GraphTraits<Value *>
    : public NodeGraphTraitsBase<Value *, Value::user_iterator> {
  using NodeRef = Value *;
  using ChildIteratorType = Value::user_iterator;
};

template <>
struct GraphTraits<const Value *>
    : public NodeGraphTraitsBase<const Value *, Value::const_user_iterator> {
  using NodeRef = const Value *;
  using ChildIteratorType = Value::const_user_iterator;
};

/// A wrapper around make_filter_range, that filters \p R for integer, pointer,
/// or vector thereof, types, excluding values that match \p ExcludeFn.
template <typename RangeT>
static auto filter_range( // NOLINT
    RangeT R, std::function<bool(const Value *)> ExcludeFn = [](const Value *) {
      return false;
    }) {
  return make_filter_range(R, [&](const Value *V) {
    return !ExcludeFn(V) && V->getType()->getScalarType()->isIntOrPtrTy();
  });
}

static bool isLeaf(const Value *V) { return filter_range(V->users()).empty(); }

ArrayRef<const Value *> KnownBitsDataflow::getRoots() const { return Roots; }

SmallVector<const Value *>
KnownBitsDataflow::getLeaves(ArrayRef<const Value *> Roots) const {
  SetVector<const Value *> Leaves;
  for (const Value *R : Roots)
    for (const Value *N : filter_range(depth_first(R)))
      if (isLeaf(N))
        Leaves.insert(N);
  return Leaves.takeVector();
}

SmallVector<const Value *> KnownBitsDataflow::getLeaves() const {
  return getLeaves(getRoots());
}

void KnownBitsDataflow::emplace_all_conflict(const Value *V) {
  emplace_or_assign(V, KnownBits::getAllConflict(DL.getTypeSizeInBits(
                           V->getType()->getScalarType())));
}

template <typename RangeT>
SmallVector<Value *> KnownBitsDataflow::insert_range(RangeT R) {
  SmallVector<Value *> Filtered(
      filter_range(R, std::bind(&KnownBitsDataflow::contains, this, _1)));
  for (Value *V : Filtered)
    emplace_all_conflict(V);
  return Filtered;
}

void KnownBitsDataflow::recurseInsertChildren(ArrayRef<Value *> R) {
  for (Value *V : R)
    recurseInsertChildren(insert_range(
        map_range(V->users(), [](User *U) -> Value * { return U; })));
}

template <typename RangeT> void KnownBitsDataflow::initialize(RangeT R) {
  for (auto *V : R) {
    emplace_all_conflict(V);
    recurseInsertChildren(V);
  }
  append_range(Roots, R);
}

void KnownBitsDataflow::initialize(Function &F) {
  // First, initialize with function arguments.
  initialize(filter_range(llvm::make_pointer_range(F.args())));
  for (BasicBlock &BB : F) {
    // Now initialize with all Instructions in the BB that weren't seen.
    initialize(filter_range(llvm::make_pointer_range(BB),
                            std::bind(&KnownBitsDataflow::contains, this, _1)));
  }
}

KnownBitsDataflow::KnownBitsDataflow(Function &F) : DL(F.getDataLayout()) {
  initialize(F);
}

SmallVector<const Value *> KnownBitsDataflow::invalidate(const Value *V) {
  SmallVector<const Value *> Leaves;
  for (const Value *N : filter_range(depth_first(V))) {
    setAllConflict(N);
    if (isLeaf(N))
      Leaves.push_back(N);
  }
  return Leaves;
}

void KnownBitsDataflow::print(raw_ostream &OS) const {
  for (const Value *R : getRoots()) {
    OS << "^ ";
    R->print(OS);
    OS << " | ";
    at(R).print(OS);
    OS << "\n";
    for (const Value *V : filter_range(drop_begin(depth_first(R)))) {
      if (isLeaf(V))
        OS << "$ ";
      else
        OS << "  ";
      V->print(OS);
      OS << " | ";
      at(V).print(OS);
      OS << "\n";
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void KnownBitsDataflow::dump() const { print(dbgs()); }
#endif

constexpr unsigned MaxAnalysisRecursionDepth = 6;

KnownBits KnownBitsCache::computeKnownBitsForHorizontalOperation(
    const Operator *I, const APInt &DemandedElts, const SimplifyQuery &Q,
    const function_ref<KnownBits(const KnownBits &, const KnownBits &)>
        KnownBitsFunc) {
  APInt DemandedEltsLHS, DemandedEltsRHS;
  getHorizDemandedEltsForFirstOperand(Q.DL.getTypeSizeInBits(I->getType()),
                                      DemandedElts, DemandedEltsLHS,
                                      DemandedEltsRHS);

  const auto ComputeForSingleOpFunc = [&](const Value *Op,
                                          APInt &DemandedEltsOp) {
    return KnownBitsFunc(computeKnownBits(Op, DemandedEltsOp, Q),
                         computeKnownBits(Op, DemandedEltsOp << 1, Q));
  };

  if (DemandedEltsRHS.isZero())
    return ComputeForSingleOpFunc(I->getOperand(0), DemandedEltsLHS);
  if (DemandedEltsLHS.isZero())
    return ComputeForSingleOpFunc(I->getOperand(1), DemandedEltsRHS);

  return ComputeForSingleOpFunc(I->getOperand(0), DemandedEltsLHS)
      .intersectWith(ComputeForSingleOpFunc(I->getOperand(1), DemandedEltsRHS));
}

void KnownBitsCache::computeKnownBitsFromLerpPattern(const Value *Op0,
                                                     const Value *Op1,
                                                     const APInt &DemandedElts,
                                                     KnownBits &KnownOut,
                                                     const SimplifyQuery &Q) {

  Type *Ty = Op0->getType();
  const unsigned BitWidth = Ty->getScalarSizeInBits();

  // Only handle scalar types for now
  if (Ty->isVectorTy())
    return;

  // Try to match: a * (b - c) + c * d.
  // When a == 1 => A == nullptr, the same applies to d/D as well.
  const Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
  const Instruction *SubBC = nullptr;

  const auto MatchSubBC = [&]() {
    // (b - c) can have two forms that interest us:
    //
    //   1. sub nuw %b, %c
    //   2. xor %c, %b
    //
    // For the first case, nuw flag guarantees our requirement b >= c.
    //
    // The second case might happen when the analysis can infer that b is a mask
    // for c and we can transform sub operation into xor (that is usually true
    // for constant b's). Even though xor is symmetrical, canonicalization
    // ensures that the constant will be the RHS. We have additional checks
    // later on to ensure that this xor operation is equivalent to subtraction.
    return m_Instruction(SubBC, m_CombineOr(m_NUWSub(m_Value(B), m_Value(C)),
                                            m_Xor(m_Value(C), m_Value(B))));
  };

  const auto MatchASubBC = [&]() {
    // Cases:
    //   - a * (b - c)
    //   - (b - c) * a
    //   - (b - c) <- a implicitly equals 1
    return m_CombineOr(m_c_Mul(m_Value(A), MatchSubBC()), MatchSubBC());
  };

  const auto MatchCD = [&]() {
    // Cases:
    //   - d * c
    //   - c * d
    //   - c <- d implicitly equals 1
    return m_CombineOr(m_c_Mul(m_Value(D), m_Specific(C)), m_Specific(C));
  };

  const auto Match = [&](const Value *LHS, const Value *RHS) {
    // We do use m_Specific(C) in MatchCD, so we have to make sure that
    // it's bound to anything and match(LHS, MatchASubBC()) absolutely
    // has to evaluate first and return true.
    //
    // If Match returns true, it is guaranteed that B != nullptr, C != nullptr.
    return match(LHS, MatchASubBC()) && match(RHS, MatchCD());
  };

  if (!Match(Op0, Op1) && !Match(Op1, Op0))
    return;

  const auto ComputeKnownBitsOrOne = [&](const Value *V) {
    // For some of the values we use the convention of leaving
    // it nullptr to signify an implicit constant 1.
    return V ? computeKnownBits(V, DemandedElts, Q)
             : KnownBits::makeConstant(APInt(BitWidth, 1));
  };

  // Check that all operands are non-negative
  const KnownBits KnownA = ComputeKnownBitsOrOne(A);
  if (!KnownA.isNonNegative())
    return;

  const KnownBits KnownD = ComputeKnownBitsOrOne(D);
  if (!KnownD.isNonNegative())
    return;

  const KnownBits KnownB = computeKnownBits(B, DemandedElts, Q);
  if (!KnownB.isNonNegative())
    return;

  const KnownBits KnownC = computeKnownBits(C, DemandedElts, Q);
  if (!KnownC.isNonNegative())
    return;

  // If we matched subtraction as xor, we need to actually check that xor
  // is semantically equivalent to subtraction.
  //
  // For that to be true, b has to be a mask for c or that b's known
  // ones cover all known and possible ones of c.
  if (SubBC->getOpcode() == Instruction::Xor &&
      !KnownC.getMaxValue().isSubsetOf(KnownB.getMinValue()))
    return;

  const APInt MaxA = KnownA.getMaxValue();
  const APInt MaxD = KnownD.getMaxValue();
  const APInt MaxAD = APIntOps::umax(MaxA, MaxD);
  const APInt MaxB = KnownB.getMaxValue();

  // We can't infer leading zeros info if the upper-bound estimate wraps.
  bool Overflow;
  const APInt UpperBound = MaxAD.umul_ov(MaxB, Overflow);

  if (Overflow)
    return;

  // If we know that x <= y and both are positive than x has at least the same
  // number of leading zeros as y.
  const unsigned MinimumNumberOfLeadingZeros = UpperBound.countl_zero();
  KnownOut.Zero.setHighBits(MinimumNumberOfLeadingZeros);
}

void KnownBitsCache::computeKnownBitsAddSub(bool Add, const Value *Op0,
                                            const Value *Op1, bool NSW,
                                            bool NUW, const APInt &DemandedElts,
                                            KnownBits &KnownOut,
                                            KnownBits &Known2,
                                            const SimplifyQuery &Q) {
  computeKnownBits(Op1, DemandedElts, KnownOut, Q);

  // If one operand is unknown and we have no nowrap information,
  // the result will be unknown independently of the second operand.
  if (KnownOut.isUnknown() && !NSW && !NUW)
    return;

  computeKnownBits(Op0, DemandedElts, Known2, Q);
  KnownOut = KnownBits::computeForAddSub(Add, NSW, NUW, Known2, KnownOut);

  if (!Add && NSW && !KnownOut.isNonNegative() &&
      isImpliedByDomCondition(ICmpInst::ICMP_SLE, Op1, Op0, Q.CxtI, Q.DL)
          .value_or(false))
    KnownOut.makeNonNegative();

  if (Add)
    // Try to match lerp pattern and combine results
    computeKnownBitsFromLerpPattern(Op0, Op1, DemandedElts, KnownOut, Q);
}

void KnownBitsCache::computeKnownBitsMul(const Value *Op0, const Value *Op1,
                                         bool NSW, bool NUW,
                                         const APInt &DemandedElts,
                                         KnownBits &Known, KnownBits &Known2,
                                         const SimplifyQuery &Q) {
  computeKnownBits(Op1, DemandedElts, Known, Q);
  computeKnownBits(Op0, DemandedElts, Known2, Q);

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
      if (!isKnownNonNegative && NUW) {
        // mul nuw nsw with a factor > 1 is non-negative.
        KnownBits One = KnownBits::makeConstant(APInt(Known.getBitWidth(), 1));
        isKnownNonNegative = KnownBits::sgt(Known, One).value_or(false) ||
                             KnownBits::sgt(Known2, One).value_or(false);
      }

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
  if (SelfMultiply)
    SelfMultiply &= isGuaranteedNotToBeUndef(Op0, Q.AC, Q.CxtI, Q.DT);

  // MISSING: computeNumSignBits in case of SelfMultiply.

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

void KnownBitsCache::computeKnownBitsFromShiftOperator(
    const Operator *I, const APInt &DemandedElts, KnownBits &Known,
    KnownBits &Known2, const SimplifyQuery &Q,
    function_ref<KnownBits(const KnownBits &, const KnownBits &, bool)> KF) {
  computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
  computeKnownBits(I->getOperand(1), DemandedElts, Known, Q);
  // To limit compile-time impact, only query isKnownNonZero() if we know at
  // least something about the shift amount.
  bool ShAmtNonZero =
      Known.isNonZero() || (Known.getMaxValue().ult(Known.getBitWidth()) &&
                            isKnownNonZero(I->getOperand(1), DemandedElts, Q));
  Known = KF(Known2, Known, ShAmtNonZero);
}

KnownBits KnownBitsCache::getKnownBitsFromAndXorOr(const Operator *I,
                                                   const APInt &DemandedElts,
                                                   const KnownBits &KnownLHS,
                                                   const KnownBits &KnownRHS,
                                                   const SimplifyQuery &Q) {
  unsigned BitWidth = KnownLHS.getBitWidth();
  KnownBits KnownOut(BitWidth);
  bool IsAnd = false;
  bool HasKnownOne = !KnownLHS.One.isZero() || !KnownRHS.One.isZero();
  Value *X = nullptr, *Y = nullptr;

  switch (I->getOpcode()) {
  case Instruction::And:
    KnownOut = KnownLHS & KnownRHS;
    IsAnd = true;
    // and(x, -x) is common idioms that will clear all but lowest set
    // bit. If we have a single known bit in x, we can clear all bits
    // above it.
    // TODO: instcombine often reassociates independent `and` which can hide
    // this pattern. Try to match and(x, and(-x, y)) / and(and(x, y), -x).
    if (HasKnownOne && match(I, m_c_And(m_Value(X), m_Neg(m_Deferred(X))))) {
      // -(-x) == x so using whichever (LHS/RHS) gets us a better result.
      if (KnownLHS.countMaxTrailingZeros() <= KnownRHS.countMaxTrailingZeros())
        KnownOut = KnownLHS.blsi();
      else
        KnownOut = KnownRHS.blsi();
    }
    break;
  case Instruction::Or:
    KnownOut = KnownLHS | KnownRHS;
    break;
  case Instruction::Xor:
    KnownOut = KnownLHS ^ KnownRHS;
    // xor(x, x-1) is common idioms that will clear all but lowest set
    // bit. If we have a single known bit in x, we can clear all bits
    // above it.
    // TODO: xor(x, x-1) is often rewritting as xor(x, x-C) where C !=
    // -1 but for the purpose of demanded bits (xor(x, x-C) &
    // Demanded) == (xor(x, x-1) & Demanded). Extend the xor pattern
    // to use arbitrary C if xor(x, x-C) as the same as xor(x, x-1).
    if (HasKnownOne &&
        match(I, m_c_Xor(m_Value(X), m_Add(m_Deferred(X), m_AllOnes())))) {
      const KnownBits &XBits = I->getOperand(0) == X ? KnownLHS : KnownRHS;
      KnownOut = XBits.blsmsk();
    }
    break;
  default:
    llvm_unreachable("Invalid Op used in 'analyzeKnownBitsFromAndXorOr'");
  }

  // and(x, add (x, -1)) is a common idiom that always clears the low bit;
  // xor/or(x, add (x, -1)) is an idiom that will always set the low bit.
  // here we handle the more general case of adding any odd number by
  // matching the form and/xor/or(x, add(x, y)) where y is odd.
  // TODO: This could be generalized to clearing any bit set in y where the
  // following bit is known to be unset in y.
  if (!KnownOut.Zero[0] && !KnownOut.One[0] &&
      (match(I, m_c_BinOp(m_Value(X), m_c_Add(m_Deferred(X), m_Value(Y)))) ||
       match(I, m_c_BinOp(m_Value(X), m_Sub(m_Deferred(X), m_Value(Y)))) ||
       match(I, m_c_BinOp(m_Value(X), m_Sub(m_Value(Y), m_Deferred(X)))))) {
    KnownBits KnownY(BitWidth);
    computeKnownBits(Y, DemandedElts, KnownY, Q);
    if (KnownY.countMinTrailingOnes() > 0) {
      if (IsAnd)
        KnownOut.Zero.setBit(0);
      else
        KnownOut.One.setBit(0);
    }
  }
  return KnownOut;
}

void KnownBitsCache::computeKnownBitsFromOperator(const Operator *I,
                                                  const APInt &DemandedElts,
                                                  KnownBits &Known,
                                                  const SimplifyQuery &Q) {
  unsigned BitWidth = Known.getBitWidth();

  KnownBits Known2(BitWidth);
  switch (I->getOpcode()) {
  default:
    break;
  case Instruction::Load:
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<LoadInst>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);
    break;
  case Instruction::And:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q);
    break;
  case Instruction::Or:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q);
    break;
  case Instruction::Xor:
    computeKnownBits(I->getOperand(1), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);

    Known = getKnownBitsFromAndXorOr(I, DemandedElts, Known2, Known, Q);
    break;
  case Instruction::Mul: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsMul(I->getOperand(0), I->getOperand(1), NSW, NUW,
                        DemandedElts, Known, Known2, Q);
    break;
  }
  case Instruction::UDiv: {
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
    Known =
        KnownBits::udiv(Known, Known2, Q.IIQ.isExact(cast<BinaryOperator>(I)));
    break;
  }
  case Instruction::SDiv: {
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
    Known =
        KnownBits::sdiv(Known, Known2, Q.IIQ.isExact(cast<BinaryOperator>(I)));
    break;
  }
  case Instruction::Select: {
    auto ComputeForArm = [&](Value *Arm, bool Invert) {
      KnownBits Res(Known.getBitWidth());
      computeKnownBits(Arm, DemandedElts, Res, Q);
      adjustKnownBitsForSelectArm(Res, I->getOperand(0), Arm, Invert, Q);
      return Res;
    };
    // Only known if known in both the LHS and RHS.
    Known =
        ComputeForArm(I->getOperand(1), /*Invert=*/false)
            .intersectWith(ComputeForArm(I->getOperand(2), /*Invert=*/true));
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
  case Instruction::PtrToAddr:
  case Instruction::IntToPtr:
    // Fall through and handle them the same as zext/trunc.
    [[fallthrough]];
  case Instruction::ZExt:
  case Instruction::Trunc: {
    Type *SrcTy = I->getOperand(0)->getType();

    unsigned SrcBitWidth;
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    Type *ScalarTy = SrcTy->getScalarType();
    SrcBitWidth = ScalarTy->isPointerTy()
                      ? Q.DL.getPointerTypeSizeInBits(ScalarTy)
                      : Q.DL.getTypeSizeInBits(ScalarTy);

    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    Known = Known.anyextOrTrunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    if (auto *Inst = dyn_cast<PossiblyNonNegInst>(I);
        Inst && Inst->hasNonNeg() && !Known.isNegative())
      Known.makeNonNegative();
    Known = Known.zextOrTrunc(BitWidth);
    break;
  }
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    if (SrcTy->isIntOrPtrTy() &&
        // TODO: For now, not handling conversions like:
        // (bitcast i64 %x to <2 x i32>)
        !I->getType()->isVectorTy()) {
      computeKnownBits(I->getOperand(0), Known, Q);
      break;
    }

    // MISSING: computeKnownFPClass to handle bitcast from floating-point to
    // integer.

    // Handle cast from vector integer type to scalar or vector integer.
    auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcTy);
    if (!SrcVecTy || !SrcVecTy->getElementType()->isIntegerTy() ||
        !I->getType()->isIntOrIntVectorTy() ||
        isa<ScalableVectorType>(I->getType()))
      break;

    unsigned NumElts = DemandedElts.getBitWidth();
    bool IsLE = Q.DL.isLittleEndian();
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
      unsigned SubScale = BitWidth / SubBitWidth;
      APInt SubDemandedElts = APInt::getZero(NumElts * SubScale);
      for (unsigned i = 0; i != NumElts; ++i) {
        if (DemandedElts[i])
          SubDemandedElts.setBit(i * SubScale);
      }

      KnownBits KnownSrc(SubBitWidth);
      for (unsigned i = 0; i != SubScale; ++i) {
        computeKnownBits(I->getOperand(0), SubDemandedElts.shl(i), KnownSrc, Q);
        unsigned ShiftElt = IsLE ? i : SubScale - 1 - i;
        Known.insertBits(KnownSrc, ShiftElt * SubBitWidth);
      }
    }
    // Look through a cast from wider vector elements to narrow type.
    // Examples: v2i64 -> v4i32
    if (SubBitWidth % BitWidth == 0) {
      unsigned SubScale = SubBitWidth / BitWidth;
      KnownBits KnownSrc(SubBitWidth);
      APInt SubDemandedElts =
          APIntOps::ScaleBitMask(DemandedElts, NumElts / SubScale);
      computeKnownBits(I->getOperand(0), SubDemandedElts, KnownSrc, Q);

      Known.setAllConflict();
      for (unsigned i = 0; i != NumElts; ++i) {
        if (DemandedElts[i]) {
          unsigned Shifts = IsLE ? i : NumElts - 1 - i;
          unsigned Offset = (Shifts % SubScale) * BitWidth;
          Known = Known.intersectWith(KnownSrc.extractBits(BitWidth, Offset));
          if (Known.isUnknown())
            break;
        }
      }
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    Known = Known.trunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    Known = Known.sext(BitWidth);
    break;
  }
  case Instruction::Shl: {
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    auto KF = [NUW, NSW](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                         bool ShAmtNonZero) {
      return KnownBits::shl(KnownVal, KnownAmt, NUW, NSW, ShAmtNonZero);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, KF);
    // Trailing zeros of a right-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setLowBits(C->countr_zero());
    break;
  }
  case Instruction::LShr: {
    bool Exact = Q.IIQ.isExact(cast<BinaryOperator>(I));
    auto KF = [Exact](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                      bool ShAmtNonZero) {
      return KnownBits::lshr(KnownVal, KnownAmt, ShAmtNonZero, Exact);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, KF);
    // Leading zeros of a left-shifted constant never decrease.
    const APInt *C;
    if (match(I->getOperand(0), m_APInt(C)))
      Known.Zero.setHighBits(C->countl_zero());
    break;
  }
  case Instruction::AShr: {
    bool Exact = Q.IIQ.isExact(cast<BinaryOperator>(I));
    auto KF = [Exact](const KnownBits &KnownVal, const KnownBits &KnownAmt,
                      bool ShAmtNonZero) {
      return KnownBits::ashr(KnownVal, KnownAmt, ShAmtNonZero, Exact);
    };
    computeKnownBitsFromShiftOperator(I, DemandedElts, Known, Known2, Q, KF);
    break;
  }
  case Instruction::Sub: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(false, I->getOperand(0), I->getOperand(1), NSW, NUW,
                           DemandedElts, Known, Known2, Q);
    break;
  }
  case Instruction::Add: {
    bool NSW = Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(I));
    bool NUW = Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(I));
    computeKnownBitsAddSub(true, I->getOperand(0), I->getOperand(1), NSW, NUW,
                           DemandedElts, Known, Known2, Q);
    break;
  }
  case Instruction::SRem:
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
    Known = KnownBits::srem(Known, Known2);
    break;

  case Instruction::URem:
    computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
    computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
    Known = KnownBits::urem(Known, Known2);
    break;
  case Instruction::Alloca:
    Known.Zero.setLowBits(Log2(cast<AllocaInst>(I)->getAlign()));
    break;
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    computeKnownBits(I->getOperand(0), Known, Q);
    // Accumulate the constant indices in a separate variable
    // to minimize the number of calls to computeForAddSub.
    unsigned IndexWidth = Q.DL.getIndexTypeSizeInBits(I->getType());
    APInt AccConstIndices(IndexWidth, 0);

    auto AddIndexToKnown = [&](KnownBits IndexBits) {
      if (IndexWidth == BitWidth) {
        // Note that inbounds does *not* guarantee nsw for the addition, as only
        // the offset is signed, while the base address is unsigned.
        Known = KnownBits::add(Known, IndexBits);
      } else {
        // If the index width is smaller than the pointer width, only add the
        // value to the low bits.
        assert(IndexWidth < BitWidth &&
               "Index width can't be larger than pointer width");
        Known.insertBits(KnownBits::add(Known.trunc(IndexWidth), IndexBits), 0);
      }
    };

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      // TrailZ can only become smaller, short-circuit if we hit zero.
      if (Known.isUnknown())
        break;

      Value *Index = I->getOperand(i);

      // Handle case when index is zero.
      Constant *CIndex = dyn_cast<Constant>(Index);
      if (CIndex && CIndex->isNullValue())
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

      TypeSize Stride = GTI.getSequentialElementStride(Q.DL);
      uint64_t StrideInBytes = Stride.getKnownMinValue();
      if (!Stride.isScalable()) {
        // Fast path for constant offset.
        if (auto *CI = dyn_cast<ConstantInt>(Index)) {
          AccConstIndices +=
              CI->getValue().sextOrTrunc(IndexWidth) * StrideInBytes;
          continue;
        }
      }

      KnownBits IndexBits = computeKnownBits(Index, Q).sextOrTrunc(IndexWidth);
      KnownBits ScalingFactor(IndexWidth);
      // Multiply by current sizeof type.
      // &A[i] == A + i * sizeof(*A[i]).
      if (Stride.isScalable()) {
        // For scalable types the only thing we know about sizeof is
        // that this is a multiple of the minimum size.
        ScalingFactor.Zero.setLowBits(llvm::countr_zero(StrideInBytes));
      } else {
        ScalingFactor =
            KnownBits::makeConstant(APInt(IndexWidth, StrideInBytes));
      }
      AddIndexToKnown(KnownBits::mul(IndexBits, ScalingFactor));
    }
    if (!Known.isUnknown() && !AccConstIndices.isZero())
      AddIndexToKnown(KnownBits::makeConstant(AccConstIndices));
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

      switch (Opcode) {
      // If this is a shift recurrence, we know the bits being shifted in. We
      // can combine that with information about the start value of the
      // recurrence to conclude facts about the result. If this is a udiv
      // recurrence, we know that the result can never exceed either the
      // numerator or the start value, whichever is greater.
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::Shl:
      case Instruction::UDiv:
        if (BO->getOperand(0) != I)
          break;
        [[fallthrough]];

      // For a urem recurrence, the result can never exceed the start value. The
      // phi could either be the numerator or the denominator.
      case Instruction::URem: {
        // We have matched a recurrence of the form:
        // %iv = [R, %entry], [%iv.next, %backedge]
        // %iv.next = shift_op %iv, L

        // Recurse with the phi context to avoid concern about whether facts
        // inferred hold at original context instruction.  TODO: It may be
        // correct to use the original context.  IF warranted, explore and
        // add sufficient tests to cover.
        SimplifyQuery RecQ = Q.getWithoutCondContext();
        RecQ.CxtI = P;
        computeKnownBits(R, DemandedElts, Known2, RecQ);
        switch (Opcode) {
        case Instruction::Shl:
          // A shl recurrence will only increase the tailing zeros
          Known.Zero.setLowBits(Known2.countMinTrailingZeros());
          break;
        case Instruction::LShr:
        case Instruction::UDiv:
        case Instruction::URem:
          // lshr, udiv, and urem recurrences will preserve the leading zeros of
          // the start value.
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          break;
        case Instruction::AShr:
          // An ashr recurrence will extend the initial sign bit
          Known.Zero.setHighBits(Known2.countMinLeadingZeros());
          Known.One.setHighBits(Known2.countMinLeadingOnes());
          break;
        }
        break;
      }

      // Check for operations that have the property that if
      // both their operands have low zero bits, the result
      // will have low zero bits.
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Mul: {
        // Change the context instruction to the "edge" that flows into the
        // phi. This is important because that is where the value is actually
        // "evaluated" even though it is used later somewhere else. (see also
        // D69571).
        SimplifyQuery RecQ = Q.getWithoutCondContext();

        unsigned OpNum = P->getOperand(0) == R ? 0 : 1;
        Instruction *RInst = P->getIncomingBlock(OpNum)->getTerminator();
        Instruction *LInst = P->getIncomingBlock(1 - OpNum)->getTerminator();

        // Ok, we have a PHI of the form L op= R. Check for low
        // zero bits.
        RecQ.CxtI = RInst;
        computeKnownBits(R, DemandedElts, Known2, RecQ);

        // We need to take the minimum number of known bits
        KnownBits Known3(BitWidth);
        RecQ.CxtI = LInst;
        computeKnownBits(L, DemandedElts, Known3, RecQ);

        Known.Zero.setLowBits(std::min(Known2.countMinTrailingZeros(),
                                       Known3.countMinTrailingZeros()));

        auto *OverflowOp = dyn_cast<OverflowingBinaryOperator>(BO);
        if (!OverflowOp || !Q.IIQ.hasNoSignedWrap(OverflowOp))
          break;

        switch (Opcode) {
        // If initial value of recurrence is nonnegative, and we are adding
        // a nonnegative number with nsw, the result can only be nonnegative
        // or poison value regardless of the number of times we execute the
        // add in phi recurrence. If initial value is negative and we are
        // adding a negative number with nsw, the result can only be
        // negative or poison value. Similar arguments apply to sub and mul.
        //
        // (add non-negative, non-negative) --> non-negative
        // (add negative, negative) --> negative
        case Instruction::Add: {
          if (Known2.isNonNegative() && Known3.isNonNegative())
            Known.makeNonNegative();
          else if (Known2.isNegative() && Known3.isNegative())
            Known.makeNegative();
          break;
        }

        // (sub nsw non-negative, negative) --> non-negative
        // (sub nsw negative, non-negative) --> negative
        case Instruction::Sub: {
          if (BO->getOperand(0) != I)
            break;
          if (Known2.isNonNegative() && Known3.isNegative())
            Known.makeNonNegative();
          else if (Known2.isNegative() && Known3.isNonNegative())
            Known.makeNegative();
          break;
        }

        // (mul nsw non-negative, non-negative) --> non-negative
        case Instruction::Mul:
          if (Known2.isNonNegative() && Known3.isNonNegative())
            Known.makeNonNegative();
          break;

        default:
          break;
        }
        break;
      }

      default:
        break;
      }
    }

    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      break;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    // Skip if every incoming value references to ourself.
    if (isa_and_nonnull<UndefValue>(P->hasConstantValue()))
      break;

    Known.setAllConflict();
    for (const Use &U : P->operands()) {
      Value *IncValue;
      const PHINode *CxtPhi;
      Instruction *CxtI;
      breakSelfRecursivePHI(&U, P, IncValue, CxtI, &CxtPhi);
      // Skip direct self references.
      if (IncValue == P)
        continue;

      // Change the context instruction to the "edge" that flows into the
      // phi. This is important because that is where the value is actually
      // "evaluated" even though it is used later somewhere else. (see also
      // D69571).
      SimplifyQuery RecQ = Q.getWithoutCondContext().getWithInstruction(CxtI);

      Known2 = KnownBits(BitWidth);

      // Recurse, but cap the recursion to one level, because we don't
      // want to waste time spinning around in loops.
      // TODO: See if we can base recursion limiter on number of incoming phi
      // edges so we don't overly clamp analysis.
      vthelper::computeKnownBits(IncValue, DemandedElts, Known2, RecQ,
                                 MaxAnalysisRecursionDepth - 1);

      // See if we can further use a conditional branch into the phi
      // to help us determine the range of the value.
      if (!Known2.isConstant()) {
        CmpPredicate Pred;
        const APInt *RHSC;
        BasicBlock *TrueSucc, *FalseSucc;
        // TODO: Use RHS Value and compute range from its known bits.
        if (match(RecQ.CxtI,
                  m_Br(m_c_ICmp(Pred, m_Specific(IncValue), m_APInt(RHSC)),
                       m_BasicBlock(TrueSucc), m_BasicBlock(FalseSucc)))) {
          // Check for cases of duplicate successors.
          if ((TrueSucc == CxtPhi->getParent()) !=
              (FalseSucc == CxtPhi->getParent())) {
            // If we're using the false successor, invert the predicate.
            if (FalseSucc == CxtPhi->getParent())
              Pred = CmpInst::getInversePredicate(Pred);
            // Get the knownbits implied by the incoming phi condition.
            auto CR = ConstantRange::makeExactICmpRegion(Pred, *RHSC);
            KnownBits KnownUnion = Known2.unionWith(CR.toKnownBits());
            // We can have conflicts here if we are analyzing deadcode (its
            // impossible for us reach this BB based the icmp).
            if (KnownUnion.hasConflict()) {
              // No reason to continue analyzing in a known dead region, so
              // just resetAll and break. This will cause us to also exit the
              // outer loop.
              Known.resetAll();
              break;
            }
            Known2 = KnownUnion;
          }
        }
      }

      // Update KnownBits of IncValue, since we're calling VT's
      // computeKnownBits.
      emplace_or_assign(IncValue, Known2);

      Known = Known.intersectWith(Known2);
      // If all bits have been ruled out, there's no need to check
      // more operands.
      if (Known.isUnknown())
        break;
    }
  } break;
  case Instruction::Call:
  case Instruction::Invoke: {
    // If range metadata is attached to this call, set known bits from that,
    // and then intersect with known bits based on other properties of the
    // function.
    if (MDNode *MD =
            Q.IIQ.getMetadata(cast<Instruction>(I), LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, Known);

    const auto *CB = cast<CallBase>(I);

    if (std::optional<ConstantRange> Range = CB->getRange())
      Known = Known.unionWith(Range->toKnownBits());

    if (const Value *RV = CB->getReturnedArgOperand()) {
      if (RV->getType() == I->getType()) {
        computeKnownBits(RV, Known2, Q);
        Known = Known.unionWith(Known2);
        // If the function doesn't return properly for all input values
        // (e.g. unreachable exits) then there might be conflicts between the
        // argument value and the range metadata. Simply discard the known bits
        // in case of conflicts.
        if (Known.hasConflict())
          Known.resetAll();
      }
    }
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default:
        break;
      case Intrinsic::abs: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        bool IntMinIsPoison = match(II->getArgOperand(1), m_One());
        Known = Known.unionWith(Known2.abs(IntMinIsPoison));
        break;
      }
      case Intrinsic::bitreverse:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        Known = Known.unionWith(Known2.reverseBits());
        break;
      case Intrinsic::bswap:
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        Known = Known.unionWith(Known2.byteSwap());
        break;
      case Intrinsic::ctlz: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleLZ = Known2.countMaxLeadingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleLZ = std::min(PossibleLZ, BitWidth - 1);
        unsigned LowBits = llvm::bit_width(PossibleLZ);
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::cttz: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        // If we have a known 1, its position is our upper bound.
        unsigned PossibleTZ = Known2.countMaxTrailingZeros();
        // If this call is poison for 0 input, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          PossibleTZ = std::min(PossibleTZ, BitWidth - 1);
        unsigned LowBits = llvm::bit_width(PossibleTZ);
        Known.Zero.setBitsFrom(LowBits);
        break;
      }
      case Intrinsic::ctpop: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        // We can bound the space the count needs.  Also, bits known to be zero
        // can't contribute to the population.
        unsigned BitsPossiblySet = Known2.countMaxPopulation();
        unsigned LowBits = llvm::bit_width(BitsPossiblySet);
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
        computeKnownBits(I->getOperand(0), DemandedElts, Known2, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known3, Q);

        Known2 <<= ShiftAmt;
        Known3 >>= BitWidth - ShiftAmt;
        Known = Known2.unionWith(Known3);
        break;
      }
      case Intrinsic::clmul:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::clmul(Known, Known2);
        break;
      case Intrinsic::uadd_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::uadd_sat(Known, Known2);
        break;
      case Intrinsic::usub_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::usub_sat(Known, Known2);
        break;
      case Intrinsic::sadd_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::sadd_sat(Known, Known2);
        break;
      case Intrinsic::ssub_sat:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::ssub_sat(Known, Known2);
        break;
        // Vec reverse preserves bits from input vec.
      case Intrinsic::vector_reverse:
        computeKnownBits(I->getOperand(0), DemandedElts.reverseBits(), Known,
                         Q);
        break;
        // for min/max/and/or reduce, any bit common to each element in the
        // input vec is set in the output.
      case Intrinsic::vector_reduce_and:
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
        computeKnownBits(I->getOperand(0), Known, Q);
        break;
      case Intrinsic::vector_reduce_xor: {
        computeKnownBits(I->getOperand(0), Known, Q);
        // The zeros common to all vecs are zero in the output.
        // If the number of elements is odd, then the common ones remain. If the
        // number of elements is even, then the common ones becomes zeros.
        auto *VecTy = cast<VectorType>(I->getOperand(0)->getType());
        // Even, so the ones become zeros.
        bool EvenCnt = VecTy->getElementCount().isKnownEven();
        if (EvenCnt)
          Known.Zero |= Known.One;
        // Maybe even element count so need to clear ones.
        if (VecTy->isScalableTy() || EvenCnt)
          Known.One.clearAllBits();
        break;
      }
      case Intrinsic::vector_reduce_add: {
        auto *VecTy = dyn_cast<FixedVectorType>(I->getOperand(0)->getType());
        if (!VecTy)
          break;
        computeKnownBits(I->getOperand(0), Known, Q);
        Known = Known.reduceAdd(VecTy->getNumElements());
        break;
      }
      case Intrinsic::umin:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::umin(Known, Known2);
        break;
      case Intrinsic::umax:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::umax(Known, Known2);
        break;
      case Intrinsic::smin:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::smin(Known, Known2);
        unionWithMinMaxIntrinsicClamp(II, Known);
        break;
      case Intrinsic::smax:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::smax(Known, Known2);
        unionWithMinMaxIntrinsicClamp(II, Known);
        break;
      case Intrinsic::ptrmask: {
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);

        const Value *Mask = I->getOperand(1);
        Known2 = KnownBits(Mask->getType()->getScalarSizeInBits());
        computeKnownBits(Mask, DemandedElts, Known2, Q);
        // TODO: 1-extend would be more precise.
        Known &= Known2.anyextOrTrunc(BitWidth);
        break;
      }
      case Intrinsic::x86_sse2_pmulh_w:
      case Intrinsic::x86_avx2_pmulh_w:
      case Intrinsic::x86_avx512_pmulh_w_512:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::mulhs(Known, Known2);
        break;
      case Intrinsic::x86_sse2_pmulhu_w:
      case Intrinsic::x86_avx2_pmulhu_w:
      case Intrinsic::x86_avx512_pmulhu_w_512:
        computeKnownBits(I->getOperand(0), DemandedElts, Known, Q);
        computeKnownBits(I->getOperand(1), DemandedElts, Known2, Q);
        Known = KnownBits::mulhu(Known, Known2);
        break;
      case Intrinsic::x86_sse42_crc32_64_64:
        Known.Zero.setBitsFrom(32);
        break;
      case Intrinsic::x86_ssse3_phadd_d_128:
      case Intrinsic::x86_ssse3_phadd_w_128:
      case Intrinsic::x86_avx2_phadd_d:
      case Intrinsic::x86_avx2_phadd_w: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q,
            [](const KnownBits &KnownLHS, const KnownBits &KnownRHS) {
              return KnownBits::add(KnownLHS, KnownRHS);
            });
        break;
      }
      case Intrinsic::x86_ssse3_phadd_sw_128:
      case Intrinsic::x86_avx2_phadd_sw: {
        Known = computeKnownBitsForHorizontalOperation(I, DemandedElts, Q,
                                                       KnownBits::sadd_sat);
        break;
      }
      case Intrinsic::x86_ssse3_phsub_d_128:
      case Intrinsic::x86_ssse3_phsub_w_128:
      case Intrinsic::x86_avx2_phsub_d:
      case Intrinsic::x86_avx2_phsub_w: {
        Known = computeKnownBitsForHorizontalOperation(
            I, DemandedElts, Q,
            [](const KnownBits &KnownLHS, const KnownBits &KnownRHS) {
              return KnownBits::sub(KnownLHS, KnownRHS);
            });
        break;
      }
      case Intrinsic::x86_ssse3_phsub_sw_128:
      case Intrinsic::x86_avx2_phsub_sw: {
        Known = computeKnownBitsForHorizontalOperation(I, DemandedElts, Q,
                                                       KnownBits::ssub_sat);
        break;
      }
      case Intrinsic::riscv_vsetvli:
      case Intrinsic::riscv_vsetvlimax: {
        bool HasAVL = II->getIntrinsicID() == Intrinsic::riscv_vsetvli;
        const ConstantRange Range = getVScaleRange(II->getFunction(), BitWidth);
        uint64_t SEW = RISCVVType::decodeVSEW(
            cast<ConstantInt>(II->getArgOperand(HasAVL))->getZExtValue());
        RISCVVType::VLMUL VLMUL = static_cast<RISCVVType::VLMUL>(
            cast<ConstantInt>(II->getArgOperand(1 + HasAVL))->getZExtValue());
        uint64_t MaxVLEN =
            Range.getUnsignedMax().getZExtValue() * RISCV::RVVBitsPerBlock;
        uint64_t MaxVL = MaxVLEN / RISCVVType::getSEWLMULRatio(SEW, VLMUL);

        // Result of vsetvli must be not larger than AVL.
        if (HasAVL)
          if (auto *CI = dyn_cast<ConstantInt>(II->getArgOperand(0)))
            MaxVL = std::min(MaxVL, CI->getZExtValue());

        unsigned KnownZeroFirstBit = Log2_32(MaxVL) + 1;
        if (BitWidth > KnownZeroFirstBit)
          Known.Zero.setBitsFrom(KnownZeroFirstBit);
        break;
      }
      case Intrinsic::vscale: {
        if (!II->getParent() || !II->getFunction())
          break;

        Known = getVScaleRange(II->getFunction(), BitWidth).toKnownBits();
        break;
      }
      }
    }
    break;
  }
  case Instruction::ShuffleVector: {
    if (auto *Splat = getSplatValue(I)) {
      computeKnownBits(Splat, Known, Q);
      break;
    }

    auto *Shuf = dyn_cast<ShuffleVectorInst>(I);
    // FIXME: Do we need to handle ConstantExpr involving shufflevectors?
    if (!Shuf) {
      Known.resetAll();
      return;
    }
    // For undef elements, we don't know anything about the common state of
    // the shuffle result.
    APInt DemandedLHS, DemandedRHS;
    if (!vthelper::getShuffleDemandedElts(Shuf, DemandedElts, DemandedLHS,
                                          DemandedRHS)) {
      Known.resetAll();
      return;
    }
    Known.setAllConflict();
    if (!!DemandedLHS) {
      const Value *LHS = Shuf->getOperand(0);
      computeKnownBits(LHS, DemandedLHS, Known, Q);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }
    if (!!DemandedRHS) {
      const Value *RHS = Shuf->getOperand(1);
      computeKnownBits(RHS, DemandedRHS, Known2, Q);
      Known = Known.intersectWith(Known2);
    }
    break;
  }
  case Instruction::InsertElement: {
    if (isa<ScalableVectorType>(I->getType())) {
      Known.resetAll();
      return;
    }
    const Value *Vec = I->getOperand(0);
    const Value *Elt = I->getOperand(1);
    auto *CIdx = dyn_cast<ConstantInt>(I->getOperand(2));
    unsigned NumElts = DemandedElts.getBitWidth();
    APInt DemandedVecElts = DemandedElts;
    bool NeedsElt = true;
    // If we know the index we are inserting too, clear it from Vec check.
    if (CIdx && CIdx->getValue().ult(NumElts)) {
      DemandedVecElts.clearBit(CIdx->getZExtValue());
      NeedsElt = DemandedElts[CIdx->getZExtValue()];
    }

    Known.setAllConflict();
    if (NeedsElt) {
      computeKnownBits(Elt, Known, Q);
      // If we don't know any bits, early out.
      if (Known.isUnknown())
        break;
    }

    if (!DemandedVecElts.isZero()) {
      computeKnownBits(Vec, DemandedVecElts, Known2, Q);
      Known = Known.intersectWith(Known2);
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
    computeKnownBits(Vec, DemandedVecElts, Known, Q);
    break;
  }
  case Instruction::ExtractValue:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->getOperand(0))) {
      const ExtractValueInst *EVI = cast<ExtractValueInst>(I);
      if (EVI->getNumIndices() != 1)
        break;
      if (EVI->getIndices()[0] == 0) {
        switch (II->getIntrinsicID()) {
        default:
          break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          computeKnownBitsAddSub(
              true, II->getArgOperand(0), II->getArgOperand(1), /*NSW=*/false,
              /* NUW=*/false, DemandedElts, Known, Known2, Q);
          break;
        case Intrinsic::usub_with_overflow:
        case Intrinsic::ssub_with_overflow:
          computeKnownBitsAddSub(
              false, II->getArgOperand(0), II->getArgOperand(1), /*NSW=*/false,
              /* NUW=*/false, DemandedElts, Known, Known2, Q);
          break;
        case Intrinsic::umul_with_overflow:
        case Intrinsic::smul_with_overflow:
          computeKnownBitsMul(II->getArgOperand(0), II->getArgOperand(1), false,
                              false, DemandedElts, Known, Known2, Q);
          break;
        }
      }
    }
    break;
  case Instruction::Freeze:
    if (isGuaranteedNotToBePoison(I->getOperand(0), Q.AC, Q.CxtI, Q.DT))
      computeKnownBits(I->getOperand(0), Known, Q);
    break;
  }
}

void KnownBitsCache::computeKnownBits(const Value *V, const APInt &DemandedElts,
                                      KnownBits &Known,
                                      const SimplifyQuery &Q) {
  if (!Known.isAllConflict())
    return;

  if (!DemandedElts) {
    // No demanded elts, better to assume we don't know anything.
    Known.resetAll();
    return;
  }

  assert(V && "No Value?");

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
           "DemandedElt width should be 1 for scalars or scalable vectors");
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
    assert(!isa<ScalableVectorType>(V->getType()));
    // We know that CDV must be a vector of integers. Take the intersection of
    // each element.
    Known.setAllConflict();
    for (unsigned i = 0, e = CDV->getNumElements(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      APInt Elt = CDV->getElementAsAPInt(i);
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    if (Known.hasConflict())
      Known.resetAll();
    return;
  }

  if (const auto *CV = dyn_cast<ConstantVector>(V)) {
    assert(!isa<ScalableVectorType>(V->getType()));
    // We know that CV must be a vector of integers. Take the intersection of
    // each element.
    Known.setAllConflict();
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      if (!DemandedElts[i])
        continue;
      Constant *Element = CV->getAggregateElement(i);
      if (isa<PoisonValue>(Element))
        continue;
      auto *ElementCI = dyn_cast_or_null<ConstantInt>(Element);
      if (!ElementCI) {
        Known.resetAll();
        return;
      }
      const APInt &Elt = ElementCI->getValue();
      Known.Zero &= ~Elt;
      Known.One &= Elt;
    }
    if (Known.hasConflict())
      Known.resetAll();
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

  if (const auto *A = dyn_cast<Argument>(V))
    if (std::optional<ConstantRange> Range = A->getRange())
      Known = Range->toKnownBits();

  // A weak GlobalAlias is totally unknown. A non-weak GlobalAlias has
  // the bits of its aliasee.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
    if (!GA->isInterposable())
      computeKnownBits(GA->getAliasee(), Known, Q);
    return;
  }

  if (const Operator *I = dyn_cast<Operator>(V))
    computeKnownBitsFromOperator(I, DemandedElts, Known, Q);
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (std::optional<ConstantRange> CR = GV->getAbsoluteSymbolRange())
      Known = CR->toKnownBits();
  }

  // Aligned pointers have trailing zeros - refine Known.Zero set
  if (isa<PointerType>(V->getType())) {
    Align Alignment = V->getPointerAlignment(Q.DL);
    Known.Zero.setLowBits(Log2(Alignment));
  }

  // computeKnownBitsFromContext strictly refines Known.
  // Therefore, we run them after computeKnownBitsFromOperator.

  // Check whether we can determine known bits from context such as assumes.
  computeKnownBitsFromContext(V, Known, Q);

  // Make sure we recursively compute KnownBits of all operands, for cases not
  // already handled.
  if (auto *I = dyn_cast<Instruction>(V))
    for (Value *Op : I->operands())
      if (contains(Op))
        computeKnownBits(Op, Q);
}

KnownBits KnownBitsCache::computeKnownBits(const Value *V,
                                           const SimplifyQuery &Q) {
  assert(contains(V) && "KnownBits information for Value not tracked");
  KnownBits &Known = getKnownBits(V);
  computeKnownBits(V, Known, Q);
  return Known;
}

KnownBits KnownBitsCache::computeKnownBits(const Value *V,
                                           const APInt &DemandedElts,
                                           const SimplifyQuery &Q) {
  assert(contains(V) && "KnownBits information for Value not tracked");
  KnownBits &Known = getKnownBits(V);
  computeKnownBits(V, DemandedElts, Known, Q);
  return Known;
}

void KnownBitsCache::computeKnownBits(const Value *V, KnownBits &Known,
                                      const SimplifyQuery &Q) {
  // Since the number of lanes in a scalable vector is unknown at compile time,
  // we track one bit which is implicitly broadcast to all lanes.  This means
  // that all lanes in a scalable vector are considered demanded.
  auto *FVTy = dyn_cast<FixedVectorType>(V->getType());
  APInt DemandedElts =
      FVTy ? APInt::getAllOnes(FVTy->getNumElements()) : APInt(1, 1);
  computeKnownBits(V, DemandedElts, Known, Q);
}

void KnownBitsCache::compute(ArrayRef<const Value *> Leaves) {
  for (const Value *V : Leaves) {
    assert(contains(V) && "KnownBits information for Value not tracked");
    computeKnownBits(V, getKnownBits(V), DL);
  }
}

KnownBits KnownBitsCache::getOrCompute(const Value *V) {
  assert(contains(V) && "KnownBits information for Value not tracked");
  KnownBits &Known = getKnownBits(V);
  if (Known.isAllConflict())
    compute(V);
  return Known;
}

KnownBitsCache::KnownBitsCache(Function &F) : KnownBitsDataflow(F) {
  compute(getLeaves());
}
