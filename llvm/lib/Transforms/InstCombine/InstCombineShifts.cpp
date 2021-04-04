//===- InstCombineShifts.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitShl, visitLShr, and visitAShr functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

bool canTryToConstantAddTwoShiftAmounts(Value *Sh0, Value *ShAmt0, Value *Sh1,
                                        Value *ShAmt1) {
  // We have two shift amounts from two different shifts. The types of those
  // shift amounts may not match. If that's the case let's bailout now..
  if (ShAmt0->getType() != ShAmt1->getType())
    return false;

  // As input, we have the following pattern:
  //   Sh0 (Sh1 X, Q), K
  // We want to rewrite that as:
  //   Sh x, (Q+K)  iff (Q+K) u< bitwidth(x)
  // While we know that originally (Q+K) would not overflow
  // (because  2 * (N-1) u<= iN -1), we have looked past extensions of
  // shift amounts. so it may now overflow in smaller bitwidth.
  // To ensure that does not happen, we need to ensure that the total maximal
  // shift amount is still representable in that smaller bit width.
  unsigned MaximalPossibleTotalShiftAmount =
      (Sh0->getType()->getScalarSizeInBits() - 1) +
      (Sh1->getType()->getScalarSizeInBits() - 1);
  APInt MaximalRepresentableShiftAmount =
      APInt::getAllOnesValue(ShAmt0->getType()->getScalarSizeInBits());
  return MaximalRepresentableShiftAmount.uge(MaximalPossibleTotalShiftAmount);
}

// Given pattern:
//   (x shiftopcode Q) shiftopcode K
// we should rewrite it as
//   x shiftopcode (Q+K)  iff (Q+K) u< bitwidth(x) and
//
// This is valid for any shift, but they must be identical, and we must be
// careful in case we have (zext(Q)+zext(K)) and look past extensions,
// (Q+K) must not overflow or else (Q+K) u< bitwidth(x) is bogus.
//
// AnalyzeForSignBitExtraction indicates that we will only analyze whether this
// pattern has any 2 right-shifts that sum to 1 less than original bit width.
Value *InstCombinerImpl::reassociateShiftAmtsOfTwoSameDirectionShifts(
    BinaryOperator *Sh0, const SimplifyQuery &SQ,
    bool AnalyzeForSignBitExtraction) {
  // Look for a shift of some instruction, ignore zext of shift amount if any.
  Instruction *Sh0Op0;
  Value *ShAmt0;
  if (!match(Sh0,
             m_Shift(m_Instruction(Sh0Op0), m_ZExtOrSelf(m_Value(ShAmt0)))))
    return nullptr;

  // If there is a truncation between the two shifts, we must make note of it
  // and look through it. The truncation imposes additional constraints on the
  // transform.
  Instruction *Sh1;
  Value *Trunc = nullptr;
  match(Sh0Op0,
        m_CombineOr(m_CombineAnd(m_Trunc(m_Instruction(Sh1)), m_Value(Trunc)),
                    m_Instruction(Sh1)));

  // Inner shift: (x shiftopcode ShAmt1)
  // Like with other shift, ignore zext of shift amount if any.
  Value *X, *ShAmt1;
  if (!match(Sh1, m_Shift(m_Value(X), m_ZExtOrSelf(m_Value(ShAmt1)))))
    return nullptr;

  // Verify that it would be safe to try to add those two shift amounts.
  if (!canTryToConstantAddTwoShiftAmounts(Sh0, ShAmt0, Sh1, ShAmt1))
    return nullptr;

  // We are only looking for signbit extraction if we have two right shifts.
  bool HadTwoRightShifts = match(Sh0, m_Shr(m_Value(), m_Value())) &&
                           match(Sh1, m_Shr(m_Value(), m_Value()));
  // ... and if it's not two right-shifts, we know the answer already.
  if (AnalyzeForSignBitExtraction && !HadTwoRightShifts)
    return nullptr;

  // The shift opcodes must be identical, unless we are just checking whether
  // this pattern can be interpreted as a sign-bit-extraction.
  Instruction::BinaryOps ShiftOpcode = Sh0->getOpcode();
  bool IdenticalShOpcodes = Sh0->getOpcode() == Sh1->getOpcode();
  if (!IdenticalShOpcodes && !AnalyzeForSignBitExtraction)
    return nullptr;

  // If we saw truncation, we'll need to produce extra instruction,
  // and for that one of the operands of the shift must be one-use,
  // unless of course we don't actually plan to produce any instructions here.
  if (Trunc && !AnalyzeForSignBitExtraction &&
      !match(Sh0, m_c_BinOp(m_OneUse(m_Value()), m_Value())))
    return nullptr;

  // Can we fold (ShAmt0+ShAmt1) ?
  auto *NewShAmt = dyn_cast_or_null<Constant>(
      SimplifyAddInst(ShAmt0, ShAmt1, /*isNSW=*/false, /*isNUW=*/false,
                      SQ.getWithInstruction(Sh0)));
  if (!NewShAmt)
    return nullptr; // Did not simplify.
  unsigned NewShAmtBitWidth = NewShAmt->getType()->getScalarSizeInBits();
  unsigned XBitWidth = X->getType()->getScalarSizeInBits();
  // Is the new shift amount smaller than the bit width of inner/new shift?
  if (!match(NewShAmt, m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT,
                                          APInt(NewShAmtBitWidth, XBitWidth))))
    return nullptr; // FIXME: could perform constant-folding.

  // If there was a truncation, and we have a right-shift, we can only fold if
  // we are left with the original sign bit. Likewise, if we were just checking
  // that this is a sighbit extraction, this is the place to check it.
  // FIXME: zero shift amount is also legal here, but we can't *easily* check
  // more than one predicate so it's not really worth it.
  if (HadTwoRightShifts && (Trunc || AnalyzeForSignBitExtraction)) {
    // If it's not a sign bit extraction, then we're done.
    if (!match(NewShAmt,
               m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ,
                                  APInt(NewShAmtBitWidth, XBitWidth - 1))))
      return nullptr;
    // If it is, and that was the question, return the base value.
    if (AnalyzeForSignBitExtraction)
      return X;
  }

  assert(IdenticalShOpcodes && "Should not get here with different shifts.");

  // All good, we can do this fold.
  NewShAmt = ConstantExpr::getZExtOrBitCast(NewShAmt, X->getType());

  BinaryOperator *NewShift = BinaryOperator::Create(ShiftOpcode, X, NewShAmt);

  // The flags can only be propagated if there wasn't a trunc.
  if (!Trunc) {
    // If the pattern did not involve trunc, and both of the original shifts
    // had the same flag set, preserve the flag.
    if (ShiftOpcode == Instruction::BinaryOps::Shl) {
      NewShift->setHasNoUnsignedWrap(Sh0->hasNoUnsignedWrap() &&
                                     Sh1->hasNoUnsignedWrap());
      NewShift->setHasNoSignedWrap(Sh0->hasNoSignedWrap() &&
                                   Sh1->hasNoSignedWrap());
    } else {
      NewShift->setIsExact(Sh0->isExact() && Sh1->isExact());
    }
  }

  Instruction *Ret = NewShift;
  if (Trunc) {
    Builder.Insert(NewShift);
    Ret = CastInst::Create(Instruction::Trunc, NewShift, Sh0->getType());
  }

  return Ret;
}

// If we have some pattern that leaves only some low bits set, and then performs
// left-shift of those bits, if none of the bits that are left after the final
// shift are modified by the mask, we can omit the mask.
//
// There are many variants to this pattern:
//   a)  (x & ((1 << MaskShAmt) - 1)) << ShiftShAmt
//   b)  (x & (~(-1 << MaskShAmt))) << ShiftShAmt
//   c)  (x & (-1 >> MaskShAmt)) << ShiftShAmt
//   d)  (x & ((-1 << MaskShAmt) >> MaskShAmt)) << ShiftShAmt
//   e)  ((x << MaskShAmt) l>> MaskShAmt) << ShiftShAmt
//   f)  ((x << MaskShAmt) a>> MaskShAmt) << ShiftShAmt
// All these patterns can be simplified to just:
//   x << ShiftShAmt
// iff:
//   a,b)     (MaskShAmt+ShiftShAmt) u>= bitwidth(x)
//   c,d,e,f) (ShiftShAmt-MaskShAmt) s>= 0 (i.e. ShiftShAmt u>= MaskShAmt)
static Instruction *
dropRedundantMaskingOfLeftShiftInput(BinaryOperator *OuterShift,
                                     const SimplifyQuery &Q,
                                     InstCombiner::BuilderTy &Builder) {
  assert(OuterShift->getOpcode() == Instruction::BinaryOps::Shl &&
         "The input must be 'shl'!");

  Value *Masked, *ShiftShAmt;
  match(OuterShift,
        m_Shift(m_Value(Masked), m_ZExtOrSelf(m_Value(ShiftShAmt))));

  // *If* there is a truncation between an outer shift and a possibly-mask,
  // then said truncation *must* be one-use, else we can't perform the fold.
  Value *Trunc;
  if (match(Masked, m_CombineAnd(m_Trunc(m_Value(Masked)), m_Value(Trunc))) &&
      !Trunc->hasOneUse())
    return nullptr;

  Type *NarrowestTy = OuterShift->getType();
  Type *WidestTy = Masked->getType();
  bool HadTrunc = WidestTy != NarrowestTy;

  // The mask must be computed in a type twice as wide to ensure
  // that no bits are lost if the sum-of-shifts is wider than the base type.
  Type *ExtendedTy = WidestTy->getExtendedType();

  Value *MaskShAmt;

  // ((1 << MaskShAmt) - 1)
  auto MaskA = m_Add(m_Shl(m_One(), m_Value(MaskShAmt)), m_AllOnes());
  // (~(-1 << maskNbits))
  auto MaskB = m_Xor(m_Shl(m_AllOnes(), m_Value(MaskShAmt)), m_AllOnes());
  // (-1 >> MaskShAmt)
  auto MaskC = m_Shr(m_AllOnes(), m_Value(MaskShAmt));
  // ((-1 << MaskShAmt) >> MaskShAmt)
  auto MaskD =
      m_Shr(m_Shl(m_AllOnes(), m_Value(MaskShAmt)), m_Deferred(MaskShAmt));

  Value *X;
  Constant *NewMask;

  if (match(Masked, m_c_And(m_CombineOr(MaskA, MaskB), m_Value(X)))) {
    // Peek through an optional zext of the shift amount.
    match(MaskShAmt, m_ZExtOrSelf(m_Value(MaskShAmt)));

    // Verify that it would be safe to try to add those two shift amounts.
    if (!canTryToConstantAddTwoShiftAmounts(OuterShift, ShiftShAmt, Masked,
                                            MaskShAmt))
      return nullptr;

    // Can we simplify (MaskShAmt+ShiftShAmt) ?
    auto *SumOfShAmts = dyn_cast_or_null<Constant>(SimplifyAddInst(
        MaskShAmt, ShiftShAmt, /*IsNSW=*/false, /*IsNUW=*/false, Q));
    if (!SumOfShAmts)
      return nullptr; // Did not simplify.
    // In this pattern SumOfShAmts correlates with the number of low bits
    // that shall remain in the root value (OuterShift).

    // An extend of an undef value becomes zero because the high bits are never
    // completely unknown. Replace the the `undef` shift amounts with final
    // shift bitwidth to ensure that the value remains undef when creating the
    // subsequent shift op.
    SumOfShAmts = Constant::replaceUndefsWith(
        SumOfShAmts, ConstantInt::get(SumOfShAmts->getType()->getScalarType(),
                                      ExtendedTy->getScalarSizeInBits()));
    auto *ExtendedSumOfShAmts = ConstantExpr::getZExt(SumOfShAmts, ExtendedTy);
    // And compute the mask as usual: ~(-1 << (SumOfShAmts))
    auto *ExtendedAllOnes = ConstantExpr::getAllOnesValue(ExtendedTy);
    auto *ExtendedInvertedMask =
        ConstantExpr::getShl(ExtendedAllOnes, ExtendedSumOfShAmts);
    NewMask = ConstantExpr::getNot(ExtendedInvertedMask);
  } else if (match(Masked, m_c_And(m_CombineOr(MaskC, MaskD), m_Value(X))) ||
             match(Masked, m_Shr(m_Shl(m_Value(X), m_Value(MaskShAmt)),
                                 m_Deferred(MaskShAmt)))) {
    // Peek through an optional zext of the shift amount.
    match(MaskShAmt, m_ZExtOrSelf(m_Value(MaskShAmt)));

    // Verify that it would be safe to try to add those two shift amounts.
    if (!canTryToConstantAddTwoShiftAmounts(OuterShift, ShiftShAmt, Masked,
                                            MaskShAmt))
      return nullptr;

    // Can we simplify (ShiftShAmt-MaskShAmt) ?
    auto *ShAmtsDiff = dyn_cast_or_null<Constant>(SimplifySubInst(
        ShiftShAmt, MaskShAmt, /*IsNSW=*/false, /*IsNUW=*/false, Q));
    if (!ShAmtsDiff)
      return nullptr; // Did not simplify.
    // In this pattern ShAmtsDiff correlates with the number of high bits that
    // shall be unset in the root value (OuterShift).

    // An extend of an undef value becomes zero because the high bits are never
    // completely unknown. Replace the the `undef` shift amounts with negated
    // bitwidth of innermost shift to ensure that the value remains undef when
    // creating the subsequent shift op.
    unsigned WidestTyBitWidth = WidestTy->getScalarSizeInBits();
    ShAmtsDiff = Constant::replaceUndefsWith(
        ShAmtsDiff, ConstantInt::get(ShAmtsDiff->getType()->getScalarType(),
                                     -WidestTyBitWidth));
    auto *ExtendedNumHighBitsToClear = ConstantExpr::getZExt(
        ConstantExpr::getSub(ConstantInt::get(ShAmtsDiff->getType(),
                                              WidestTyBitWidth,
                                              /*isSigned=*/false),
                             ShAmtsDiff),
        ExtendedTy);
    // And compute the mask as usual: (-1 l>> (NumHighBitsToClear))
    auto *ExtendedAllOnes = ConstantExpr::getAllOnesValue(ExtendedTy);
    NewMask =
        ConstantExpr::getLShr(ExtendedAllOnes, ExtendedNumHighBitsToClear);
  } else
    return nullptr; // Don't know anything about this pattern.

  NewMask = ConstantExpr::getTrunc(NewMask, NarrowestTy);

  // Does this mask has any unset bits? If not then we can just not apply it.
  bool NeedMask = !match(NewMask, m_AllOnes());

  // If we need to apply a mask, there are several more restrictions we have.
  if (NeedMask) {
    // The old masking instruction must go away.
    if (!Masked->hasOneUse())
      return nullptr;
    // The original "masking" instruction must not have been`ashr`.
    if (match(Masked, m_AShr(m_Value(), m_Value())))
      return nullptr;
  }

  // If we need to apply truncation, let's do it first, since we can.
  // We have already ensured that the old truncation will go away.
  if (HadTrunc)
    X = Builder.CreateTrunc(X, NarrowestTy);

  // No 'NUW'/'NSW'! We no longer know that we won't shift-out non-0 bits.
  // We didn't change the Type of this outermost shift, so we can just do it.
  auto *NewShift = BinaryOperator::Create(OuterShift->getOpcode(), X,
                                          OuterShift->getOperand(1));
  if (!NeedMask)
    return NewShift;

  Builder.Insert(NewShift);
  return BinaryOperator::Create(Instruction::And, NewShift, NewMask);
}

/// If we have a shift-by-constant of a bitwise logic op that itself has a
/// shift-by-constant operand with identical opcode, we may be able to convert
/// that into 2 independent shifts followed by the logic op. This eliminates a
/// a use of an intermediate value (reduces dependency chain).
static Instruction *foldShiftOfShiftedLogic(BinaryOperator &I,
                                            InstCombiner::BuilderTy &Builder) {
  assert(I.isShift() && "Expected a shift as input");
  auto *LogicInst = dyn_cast<BinaryOperator>(I.getOperand(0));
  if (!LogicInst || !LogicInst->isBitwiseLogicOp() || !LogicInst->hasOneUse())
    return nullptr;

  Constant *C0, *C1;
  if (!match(I.getOperand(1), m_Constant(C1)))
    return nullptr;

  Instruction::BinaryOps ShiftOpcode = I.getOpcode();
  Type *Ty = I.getType();

  // Find a matching one-use shift by constant. The fold is not valid if the sum
  // of the shift values equals or exceeds bitwidth.
  // TODO: Remove the one-use check if the other logic operand (Y) is constant.
  Value *X, *Y;
  auto matchFirstShift = [&](Value *V) {
    BinaryOperator *BO;
    APInt Threshold(Ty->getScalarSizeInBits(), Ty->getScalarSizeInBits());
    return match(V, m_BinOp(BO)) && BO->getOpcode() == ShiftOpcode &&
           match(V, m_OneUse(m_Shift(m_Value(X), m_Constant(C0)))) &&
           match(ConstantExpr::getAdd(C0, C1),
                 m_SpecificInt_ICMP(ICmpInst::ICMP_ULT, Threshold));
  };

  // Logic ops are commutative, so check each operand for a match.
  if (matchFirstShift(LogicInst->getOperand(0)))
    Y = LogicInst->getOperand(1);
  else if (matchFirstShift(LogicInst->getOperand(1)))
    Y = LogicInst->getOperand(0);
  else
    return nullptr;

  // shift (logic (shift X, C0), Y), C1 -> logic (shift X, C0+C1), (shift Y, C1)
  Constant *ShiftSumC = ConstantExpr::getAdd(C0, C1);
  Value *NewShift1 = Builder.CreateBinOp(ShiftOpcode, X, ShiftSumC);
  Value *NewShift2 = Builder.CreateBinOp(ShiftOpcode, Y, I.getOperand(1));
  return BinaryOperator::Create(LogicInst->getOpcode(), NewShift1, NewShift2);
}

Instruction *InstCombinerImpl::commonShiftTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  assert(Op0->getType() == Op1->getType());

  // If the shift amount is a one-use `sext`, we can demote it to `zext`.
  Value *Y;
  if (match(Op1, m_OneUse(m_SExt(m_Value(Y))))) {
    Value *NewExt = Builder.CreateZExt(Y, I.getType(), Op1->getName());
    return BinaryOperator::Create(I.getOpcode(), Op0, NewExt);
  }

  // See if we can fold away this shift.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (Constant *CUI = dyn_cast<Constant>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;

  if (auto *NewShift = cast_or_null<Instruction>(
          reassociateShiftAmtsOfTwoSameDirectionShifts(&I, SQ)))
    return NewShift;

  // (C1 shift (A add C2)) -> (C1 shift C2) shift A)
  // iff A and C2 are both positive.
  Value *A;
  Constant *C;
  if (match(Op0, m_Constant()) && match(Op1, m_Add(m_Value(A), m_Constant(C))))
    if (isKnownNonNegative(A, DL, 0, &AC, &I, &DT) &&
        isKnownNonNegative(C, DL, 0, &AC, &I, &DT))
      return BinaryOperator::Create(
          I.getOpcode(), Builder.CreateBinOp(I.getOpcode(), Op0, C), A);

  // X shift (A srem C) -> X shift (A and (C - 1)) iff C is a power of 2.
  // Because shifts by negative values (which could occur if A were negative)
  // are undefined.
  if (Op1->hasOneUse() && match(Op1, m_SRem(m_Value(A), m_Constant(C))) &&
      match(C, m_Power2())) {
    // FIXME: Should this get moved into SimplifyDemandedBits by saying we don't
    // demand the sign bit (and many others) here??
    Constant *Mask = ConstantExpr::getSub(C, ConstantInt::get(I.getType(), 1));
    Value *Rem = Builder.CreateAnd(A, Mask, Op1->getName());
    return replaceOperand(I, 1, Rem);
  }

  if (Instruction *Logic = foldShiftOfShiftedLogic(I, Builder))
    return Logic;

  return nullptr;
}

/// Return true if we can simplify two logical (either left or right) shifts
/// that have constant shift amounts: OuterShift (InnerShift X, C1), C2.
static bool canEvaluateShiftedShift(unsigned OuterShAmt, bool IsOuterShl,
                                    Instruction *InnerShift,
                                    InstCombinerImpl &IC, Instruction *CxtI) {
  assert(InnerShift->isLogicalShift() && "Unexpected instruction type");

  // We need constant scalar or constant splat shifts.
  const APInt *InnerShiftConst;
  if (!match(InnerShift->getOperand(1), m_APInt(InnerShiftConst)))
    return false;

  // Two logical shifts in the same direction:
  // shl (shl X, C1), C2 -->  shl X, C1 + C2
  // lshr (lshr X, C1), C2 --> lshr X, C1 + C2
  bool IsInnerShl = InnerShift->getOpcode() == Instruction::Shl;
  if (IsInnerShl == IsOuterShl)
    return true;

  // Equal shift amounts in opposite directions become bitwise 'and':
  // lshr (shl X, C), C --> and X, C'
  // shl (lshr X, C), C --> and X, C'
  if (*InnerShiftConst == OuterShAmt)
    return true;

  // If the 2nd shift is bigger than the 1st, we can fold:
  // lshr (shl X, C1), C2 -->  and (shl X, C1 - C2), C3
  // shl (lshr X, C1), C2 --> and (lshr X, C1 - C2), C3
  // but it isn't profitable unless we know the and'd out bits are already zero.
  // Also, check that the inner shift is valid (less than the type width) or
  // we'll crash trying to produce the bit mask for the 'and'.
  unsigned TypeWidth = InnerShift->getType()->getScalarSizeInBits();
  if (InnerShiftConst->ugt(OuterShAmt) && InnerShiftConst->ult(TypeWidth)) {
    unsigned InnerShAmt = InnerShiftConst->getZExtValue();
    unsigned MaskShift =
        IsInnerShl ? TypeWidth - InnerShAmt : InnerShAmt - OuterShAmt;
    APInt Mask = APInt::getLowBitsSet(TypeWidth, OuterShAmt) << MaskShift;
    if (IC.MaskedValueIsZero(InnerShift->getOperand(0), Mask, 0, CxtI))
      return true;
  }

  return false;
}

/// See if we can compute the specified value, but shifted logically to the left
/// or right by some number of bits. This should return true if the expression
/// can be computed for the same cost as the current expression tree. This is
/// used to eliminate extraneous shifting from things like:
///      %C = shl i128 %A, 64
///      %D = shl i128 %B, 96
///      %E = or i128 %C, %D
///      %F = lshr i128 %E, 64
/// where the client will ask if E can be computed shifted right by 64-bits. If
/// this succeeds, getShiftedValue() will be called to produce the value.
static bool canEvaluateShifted(Value *V, unsigned NumBits, bool IsLeftShift,
                               InstCombinerImpl &IC, Instruction *CxtI) {
  // We can always evaluate constants shifted.
  if (isa<Constant>(V))
    return true;

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // We can't mutate something that has multiple uses: doing so would
  // require duplicating the instruction in general, which isn't profitable.
  if (!I->hasOneUse()) return false;

  switch (I->getOpcode()) {
  default: return false;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // Bitwise operators can all arbitrarily be arbitrarily evaluated shifted.
    return canEvaluateShifted(I->getOperand(0), NumBits, IsLeftShift, IC, I) &&
           canEvaluateShifted(I->getOperand(1), NumBits, IsLeftShift, IC, I);

  case Instruction::Shl:
  case Instruction::LShr:
    return canEvaluateShiftedShift(NumBits, IsLeftShift, I, IC, CxtI);

  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    Value *TrueVal = SI->getTrueValue();
    Value *FalseVal = SI->getFalseValue();
    return canEvaluateShifted(TrueVal, NumBits, IsLeftShift, IC, SI) &&
           canEvaluateShifted(FalseVal, NumBits, IsLeftShift, IC, SI);
  }
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (Value *IncValue : PN->incoming_values())
      if (!canEvaluateShifted(IncValue, NumBits, IsLeftShift, IC, PN))
        return false;
    return true;
  }
  }
}

/// Fold OuterShift (InnerShift X, C1), C2.
/// See canEvaluateShiftedShift() for the constraints on these instructions.
static Value *foldShiftedShift(BinaryOperator *InnerShift, unsigned OuterShAmt,
                               bool IsOuterShl,
                               InstCombiner::BuilderTy &Builder) {
  bool IsInnerShl = InnerShift->getOpcode() == Instruction::Shl;
  Type *ShType = InnerShift->getType();
  unsigned TypeWidth = ShType->getScalarSizeInBits();

  // We only accept shifts-by-a-constant in canEvaluateShifted().
  const APInt *C1;
  match(InnerShift->getOperand(1), m_APInt(C1));
  unsigned InnerShAmt = C1->getZExtValue();

  // Change the shift amount and clear the appropriate IR flags.
  auto NewInnerShift = [&](unsigned ShAmt) {
    InnerShift->setOperand(1, ConstantInt::get(ShType, ShAmt));
    if (IsInnerShl) {
      InnerShift->setHasNoUnsignedWrap(false);
      InnerShift->setHasNoSignedWrap(false);
    } else {
      InnerShift->setIsExact(false);
    }
    return InnerShift;
  };

  // Two logical shifts in the same direction:
  // shl (shl X, C1), C2 -->  shl X, C1 + C2
  // lshr (lshr X, C1), C2 --> lshr X, C1 + C2
  if (IsInnerShl == IsOuterShl) {
    // If this is an oversized composite shift, then unsigned shifts get 0.
    if (InnerShAmt + OuterShAmt >= TypeWidth)
      return Constant::getNullValue(ShType);

    return NewInnerShift(InnerShAmt + OuterShAmt);
  }

  // Equal shift amounts in opposite directions become bitwise 'and':
  // lshr (shl X, C), C --> and X, C'
  // shl (lshr X, C), C --> and X, C'
  if (InnerShAmt == OuterShAmt) {
    APInt Mask = IsInnerShl
                     ? APInt::getLowBitsSet(TypeWidth, TypeWidth - OuterShAmt)
                     : APInt::getHighBitsSet(TypeWidth, TypeWidth - OuterShAmt);
    Value *And = Builder.CreateAnd(InnerShift->getOperand(0),
                                   ConstantInt::get(ShType, Mask));
    if (auto *AndI = dyn_cast<Instruction>(And)) {
      AndI->moveBefore(InnerShift);
      AndI->takeName(InnerShift);
    }
    return And;
  }

  assert(InnerShAmt > OuterShAmt &&
         "Unexpected opposite direction logical shift pair");

  // In general, we would need an 'and' for this transform, but
  // canEvaluateShiftedShift() guarantees that the masked-off bits are not used.
  // lshr (shl X, C1), C2 -->  shl X, C1 - C2
  // shl (lshr X, C1), C2 --> lshr X, C1 - C2
  return NewInnerShift(InnerShAmt - OuterShAmt);
}

/// When canEvaluateShifted() returns true for an expression, this function
/// inserts the new computation that produces the shifted value.
static Value *getShiftedValue(Value *V, unsigned NumBits, bool isLeftShift,
                              InstCombinerImpl &IC, const DataLayout &DL) {
  // We can always evaluate constants shifted.
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (isLeftShift)
      return IC.Builder.CreateShl(C, NumBits);
    else
      return IC.Builder.CreateLShr(C, NumBits);
  }

  Instruction *I = cast<Instruction>(V);
  IC.addToWorklist(I);

  switch (I->getOpcode()) {
  default: llvm_unreachable("Inconsistency with CanEvaluateShifted");
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // Bitwise operators can all arbitrarily be arbitrarily evaluated shifted.
    I->setOperand(
        0, getShiftedValue(I->getOperand(0), NumBits, isLeftShift, IC, DL));
    I->setOperand(
        1, getShiftedValue(I->getOperand(1), NumBits, isLeftShift, IC, DL));
    return I;

  case Instruction::Shl:
  case Instruction::LShr:
    return foldShiftedShift(cast<BinaryOperator>(I), NumBits, isLeftShift,
                            IC.Builder);

  case Instruction::Select:
    I->setOperand(
        1, getShiftedValue(I->getOperand(1), NumBits, isLeftShift, IC, DL));
    I->setOperand(
        2, getShiftedValue(I->getOperand(2), NumBits, isLeftShift, IC, DL));
    return I;
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      PN->setIncomingValue(i, getShiftedValue(PN->getIncomingValue(i), NumBits,
                                              isLeftShift, IC, DL));
    return PN;
  }
  }
}

// If this is a bitwise operator or add with a constant RHS we might be able
// to pull it through a shift.
static bool canShiftBinOpWithConstantRHS(BinaryOperator &Shift,
                                         BinaryOperator *BO) {
  switch (BO->getOpcode()) {
  default:
    return false; // Do not perform transform!
  case Instruction::Add:
    return Shift.getOpcode() == Instruction::Shl;
  case Instruction::Or:
  case Instruction::And:
    return true;
  case Instruction::Xor:
    // Do not change a 'not' of logical shift because that would create a normal
    // 'xor'. The 'not' is likely better for analysis, SCEV, and codegen.
    return !(Shift.isLogicalShift() && match(BO, m_Not(m_Value())));
  }
}

Instruction *InstCombinerImpl::FoldShiftByConstant(Value *Op0, Constant *Op1,
                                                   BinaryOperator &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  const APInt *Op1C;
  if (!match(Op1, m_APInt(Op1C)))
    return nullptr;

  // See if we can propagate this shift into the input, this covers the trivial
  // cast of lshr(shl(x,c1),c2) as well as other more complex cases.
  if (I.getOpcode() != Instruction::AShr &&
      canEvaluateShifted(Op0, Op1C->getZExtValue(), isLeftShift, *this, &I)) {
    LLVM_DEBUG(
        dbgs() << "ICE: GetShiftedValue propagating shift through expression"
                  " to eliminate shift:\n  IN: "
               << *Op0 << "\n  SH: " << I << "\n");

    return replaceInstUsesWith(
        I, getShiftedValue(Op0, Op1C->getZExtValue(), isLeftShift, *this, DL));
  }

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  Type *Ty = I.getType();
  unsigned TypeBits = Ty->getScalarSizeInBits();
  assert(!Op1C->uge(TypeBits) &&
         "Shift over the type width should have been removed already");

  if (Instruction *FoldedShift = foldBinOpIntoSelectOrPhi(I))
    return FoldedShift;

  // Fold shift2(trunc(shift1(x,c1)), c2) -> trunc(shift2(shift1(x,c1),c2))
  if (auto *TI = dyn_cast<TruncInst>(Op0)) {
    // If 'shift2' is an ashr, we would have to get the sign bit into a funny
    // place.  Don't try to do this transformation in this case.  Also, we
    // require that the input operand is a shift-by-constant so that we have
    // confidence that the shifts will get folded together.  We could do this
    // xform in more cases, but it is unlikely to be profitable.
    const APInt *TrShiftAmt;
    if (I.isLogicalShift() &&
        match(TI->getOperand(0), m_Shift(m_Value(), m_APInt(TrShiftAmt)))) {
      auto *TrOp = cast<Instruction>(TI->getOperand(0));
      Type *SrcTy = TrOp->getType();

      // Okay, we'll do this xform.  Make the shift of shift.
      Constant *ShAmt = ConstantExpr::getZExt(Op1, SrcTy);
      // (shift2 (shift1 & 0x00FF), c2)
      Value *NSh = Builder.CreateBinOp(I.getOpcode(), TrOp, ShAmt, I.getName());

      // For logical shifts, the truncation has the effect of making the high
      // part of the register be zeros.  Emulate this by inserting an AND to
      // clear the top bits as needed.  This 'and' will usually be zapped by
      // other xforms later if dead.
      unsigned SrcSize = SrcTy->getScalarSizeInBits();
      Constant *MaskV =
          ConstantInt::get(SrcTy, APInt::getLowBitsSet(SrcSize, TypeBits));

      // The mask we constructed says what the trunc would do if occurring
      // between the shifts.  We want to know the effect *after* the second
      // shift.  We know that it is a logical shift by a constant, so adjust the
      // mask as appropriate.
      MaskV = ConstantExpr::get(I.getOpcode(), MaskV, ShAmt);
      // shift1 & 0x00FF
      Value *And = Builder.CreateAnd(NSh, MaskV, TI->getName());
      // Return the value truncated to the interesting size.
      return new TruncInst(And, Ty);
    }
  }

  if (Op0->hasOneUse()) {
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1;
      const APInt *CC;
      switch (Op0BO->getOpcode()) {
      default: break;
      case Instruction::Add:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor: {
        // These operators commute.
        // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
            match(Op0BO->getOperand(1), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =         // (Y << C)
            Builder.CreateShl(Op0BO->getOperand(0), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder.CreateBinOp(Op0BO->getOpcode(), YS, V1,
                                         Op0BO->getOperand(1)->getName());
          unsigned Op1Val = Op1C->getLimitedValue(TypeBits);
          APInt Bits = APInt::getHighBitsSet(TypeBits, TypeBits - Op1Val);
          Constant *Mask = ConstantInt::get(Ty, Bits);
          return BinaryOperator::CreateAnd(X, Mask);
        }

        // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
        Value *Op0BOOp1 = Op0BO->getOperand(1);
        if (isLeftShift && Op0BOOp1->hasOneUse() &&
            match(Op0BOOp1, m_And(m_OneUse(m_Shr(m_Value(V1), m_Specific(Op1))),
                                  m_APInt(CC)))) {
          Value *YS = // (Y << C)
              Builder.CreateShl(Op0BO->getOperand(0), Op1, Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder.CreateAnd(
              V1, ConstantExpr::getShl(ConstantInt::get(Ty, *CC), Op1),
              V1->getName() + ".mask");
          return BinaryOperator::Create(Op0BO->getOpcode(), YS, XM);
        }
        LLVM_FALLTHROUGH;
      }

      case Instruction::Sub: {
        // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =  // (Y << C)
            Builder.CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder.CreateBinOp(Op0BO->getOpcode(), V1, YS,
                                         Op0BO->getOperand(0)->getName());
          unsigned Op1Val = Op1C->getLimitedValue(TypeBits);
          APInt Bits = APInt::getHighBitsSet(TypeBits, TypeBits - Op1Val);
          Constant *Mask = ConstantInt::get(Ty, Bits);
          return BinaryOperator::CreateAnd(X, Mask);
        }

        // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0),
                  m_And(m_OneUse(m_Shr(m_Value(V1), m_Specific(Op1))),
                        m_APInt(CC)))) {
          Value *YS = // (Y << C)
              Builder.CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder.CreateAnd(
              V1, ConstantExpr::getShl(ConstantInt::get(Ty, *CC), Op1),
              V1->getName() + ".mask");
          return BinaryOperator::Create(Op0BO->getOpcode(), XM, YS);
        }

        break;
      }
      }

      // If the operand is a bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      const APInt *Op0C;
      if (match(Op0BO->getOperand(1), m_APInt(Op0C))) {
        if (canShiftBinOpWithConstantRHS(I, Op0BO)) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(),
                                     cast<Constant>(Op0BO->getOperand(1)), Op1);

          Value *NewShift =
            Builder.CreateBinOp(I.getOpcode(), Op0BO->getOperand(0), Op1);
          NewShift->takeName(Op0BO);

          return BinaryOperator::Create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }

      // If the operand is a subtract with a constant LHS, and the shift
      // is the only use, we can pull it out of the shift.
      // This folds (shl (sub C1, X), C2) -> (sub (C1 << C2), (shl X, C2))
      if (isLeftShift && Op0BO->getOpcode() == Instruction::Sub &&
          match(Op0BO->getOperand(0), m_APInt(Op0C))) {
        Constant *NewRHS = ConstantExpr::get(I.getOpcode(),
                                   cast<Constant>(Op0BO->getOperand(0)), Op1);

        Value *NewShift = Builder.CreateShl(Op0BO->getOperand(1), Op1);
        NewShift->takeName(Op0BO);

        return BinaryOperator::CreateSub(NewRHS, NewShift);
      }
    }

    // If we have a select that conditionally executes some binary operator,
    // see if we can pull it the select and operator through the shift.
    //
    // For example, turning:
    //   shl (select C, (add X, C1), X), C2
    // Into:
    //   Y = shl X, C2
    //   select C, (add Y, C1 << C2), Y
    Value *Cond;
    BinaryOperator *TBO;
    Value *FalseVal;
    if (match(Op0, m_Select(m_Value(Cond), m_OneUse(m_BinOp(TBO)),
                            m_Value(FalseVal)))) {
      const APInt *C;
      if (!isa<Constant>(FalseVal) && TBO->getOperand(0) == FalseVal &&
          match(TBO->getOperand(1), m_APInt(C)) &&
          canShiftBinOpWithConstantRHS(I, TBO)) {
        Constant *NewRHS = ConstantExpr::get(I.getOpcode(),
                                       cast<Constant>(TBO->getOperand(1)), Op1);

        Value *NewShift =
          Builder.CreateBinOp(I.getOpcode(), FalseVal, Op1);
        Value *NewOp = Builder.CreateBinOp(TBO->getOpcode(), NewShift,
                                           NewRHS);
        return SelectInst::Create(Cond, NewOp, NewShift);
      }
    }

    BinaryOperator *FBO;
    Value *TrueVal;
    if (match(Op0, m_Select(m_Value(Cond), m_Value(TrueVal),
                            m_OneUse(m_BinOp(FBO))))) {
      const APInt *C;
      if (!isa<Constant>(TrueVal) && FBO->getOperand(0) == TrueVal &&
          match(FBO->getOperand(1), m_APInt(C)) &&
          canShiftBinOpWithConstantRHS(I, FBO)) {
        Constant *NewRHS = ConstantExpr::get(I.getOpcode(),
                                       cast<Constant>(FBO->getOperand(1)), Op1);

        Value *NewShift =
          Builder.CreateBinOp(I.getOpcode(), TrueVal, Op1);
        Value *NewOp = Builder.CreateBinOp(FBO->getOpcode(), NewShift,
                                           NewRHS);
        return SelectInst::Create(Cond, NewShift, NewOp);
      }
    }
  }

  return nullptr;
}

Instruction *InstCombinerImpl::visitShl(BinaryOperator &I) {
  const SimplifyQuery Q = SQ.getWithInstruction(&I);

  if (Value *V = SimplifyShlInst(I.getOperand(0), I.getOperand(1),
                                 I.hasNoSignedWrap(), I.hasNoUnsignedWrap(), Q))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *V = commonShiftTransforms(I))
    return V;

  if (Instruction *V = dropRedundantMaskingOfLeftShiftInput(&I, Q, Builder))
    return V;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();

  const APInt *ShAmtAPInt;
  if (match(Op1, m_APInt(ShAmtAPInt))) {
    unsigned ShAmt = ShAmtAPInt->getZExtValue();

    // shl (zext X), ShAmt --> zext (shl X, ShAmt)
    // This is only valid if X would have zeros shifted out.
    Value *X;
    if (match(Op0, m_OneUse(m_ZExt(m_Value(X))))) {
      unsigned SrcWidth = X->getType()->getScalarSizeInBits();
      if (ShAmt < SrcWidth &&
          MaskedValueIsZero(X, APInt::getHighBitsSet(SrcWidth, ShAmt), 0, &I))
        return new ZExtInst(Builder.CreateShl(X, ShAmt), Ty);
    }

    // (X >> C) << C --> X & (-1 << C)
    if (match(Op0, m_Shr(m_Value(X), m_Specific(Op1)))) {
      APInt Mask(APInt::getHighBitsSet(BitWidth, BitWidth - ShAmt));
      return BinaryOperator::CreateAnd(X, ConstantInt::get(Ty, Mask));
    }

    const APInt *ShOp1;
    if (match(Op0, m_Exact(m_Shr(m_Value(X), m_APInt(ShOp1)))) &&
        ShOp1->ult(BitWidth)) {
      unsigned ShrAmt = ShOp1->getZExtValue();
      if (ShrAmt < ShAmt) {
        // If C1 < C2: (X >>?,exact C1) << C2 --> X << (C2 - C1)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShAmt - ShrAmt);
        auto *NewShl = BinaryOperator::CreateShl(X, ShiftDiff);
        NewShl->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
        NewShl->setHasNoSignedWrap(I.hasNoSignedWrap());
        return NewShl;
      }
      if (ShrAmt > ShAmt) {
        // If C1 > C2: (X >>?exact C1) << C2 --> X >>?exact (C1 - C2)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShrAmt - ShAmt);
        auto *NewShr = BinaryOperator::Create(
            cast<BinaryOperator>(Op0)->getOpcode(), X, ShiftDiff);
        NewShr->setIsExact(true);
        return NewShr;
      }
    }

    if (match(Op0, m_OneUse(m_Shr(m_Value(X), m_APInt(ShOp1)))) &&
        ShOp1->ult(BitWidth)) {
      unsigned ShrAmt = ShOp1->getZExtValue();
      if (ShrAmt < ShAmt) {
        // If C1 < C2: (X >>? C1) << C2 --> X << (C2 - C1) & (-1 << C2)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShAmt - ShrAmt);
        auto *NewShl = BinaryOperator::CreateShl(X, ShiftDiff);
        NewShl->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
        NewShl->setHasNoSignedWrap(I.hasNoSignedWrap());
        Builder.Insert(NewShl);
        APInt Mask(APInt::getHighBitsSet(BitWidth, BitWidth - ShAmt));
        return BinaryOperator::CreateAnd(NewShl, ConstantInt::get(Ty, Mask));
      }
      if (ShrAmt > ShAmt) {
        // If C1 > C2: (X >>? C1) << C2 --> X >>? (C1 - C2) & (-1 << C2)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShrAmt - ShAmt);
        auto *OldShr = cast<BinaryOperator>(Op0);
        auto *NewShr =
            BinaryOperator::Create(OldShr->getOpcode(), X, ShiftDiff);
        NewShr->setIsExact(OldShr->isExact());
        Builder.Insert(NewShr);
        APInt Mask(APInt::getHighBitsSet(BitWidth, BitWidth - ShAmt));
        return BinaryOperator::CreateAnd(NewShr, ConstantInt::get(Ty, Mask));
      }
    }

    if (match(Op0, m_Shl(m_Value(X), m_APInt(ShOp1))) && ShOp1->ult(BitWidth)) {
      unsigned AmtSum = ShAmt + ShOp1->getZExtValue();
      // Oversized shifts are simplified to zero in InstSimplify.
      if (AmtSum < BitWidth)
        // (X << C1) << C2 --> X << (C1 + C2)
        return BinaryOperator::CreateShl(X, ConstantInt::get(Ty, AmtSum));
    }

    // If the shifted-out value is known-zero, then this is a NUW shift.
    if (!I.hasNoUnsignedWrap() &&
        MaskedValueIsZero(Op0, APInt::getHighBitsSet(BitWidth, ShAmt), 0, &I)) {
      I.setHasNoUnsignedWrap();
      return &I;
    }

    // If the shifted-out value is all signbits, then this is a NSW shift.
    if (!I.hasNoSignedWrap() && ComputeNumSignBits(Op0, 0, &I) > ShAmt) {
      I.setHasNoSignedWrap();
      return &I;
    }
  }

  // Transform  (x >> y) << y  to  x & (-1 << y)
  // Valid for any type of right-shift.
  Value *X;
  if (match(Op0, m_OneUse(m_Shr(m_Value(X), m_Specific(Op1))))) {
    Constant *AllOnes = ConstantInt::getAllOnesValue(Ty);
    Value *Mask = Builder.CreateShl(AllOnes, Op1);
    return BinaryOperator::CreateAnd(Mask, X);
  }

  Constant *C1;
  if (match(Op1, m_Constant(C1))) {
    Constant *C2;
    Value *X;
    // (C2 << X) << C1 --> (C2 << C1) << X
    if (match(Op0, m_OneUse(m_Shl(m_Constant(C2), m_Value(X)))))
      return BinaryOperator::CreateShl(ConstantExpr::getShl(C2, C1), X);

    // (X * C2) << C1 --> X * (C2 << C1)
    if (match(Op0, m_Mul(m_Value(X), m_Constant(C2))))
      return BinaryOperator::CreateMul(X, ConstantExpr::getShl(C2, C1));

    // shl (zext i1 X), C1 --> select (X, 1 << C1, 0)
    if (match(Op0, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)) {
      auto *NewC = ConstantExpr::getShl(ConstantInt::get(Ty, 1), C1);
      return SelectInst::Create(X, NewC, ConstantInt::getNullValue(Ty));
    }
  }

  // (1 << (C - x)) -> ((1 << C) >> x) if C is bitwidth - 1
  if (match(Op0, m_One()) &&
      match(Op1, m_Sub(m_SpecificInt(BitWidth - 1), m_Value(X))))
    return BinaryOperator::CreateLShr(
        ConstantInt::get(Ty, APInt::getSignMask(BitWidth)), X);

  return nullptr;
}

Instruction *InstCombinerImpl::visitLShr(BinaryOperator &I) {
  if (Value *V = SimplifyLShrInst(I.getOperand(0), I.getOperand(1), I.isExact(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *R = commonShiftTransforms(I))
    return R;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  const APInt *ShAmtAPInt;
  if (match(Op1, m_APInt(ShAmtAPInt))) {
    unsigned ShAmt = ShAmtAPInt->getZExtValue();
    unsigned BitWidth = Ty->getScalarSizeInBits();
    auto *II = dyn_cast<IntrinsicInst>(Op0);
    if (II && isPowerOf2_32(BitWidth) && Log2_32(BitWidth) == ShAmt &&
        (II->getIntrinsicID() == Intrinsic::ctlz ||
         II->getIntrinsicID() == Intrinsic::cttz ||
         II->getIntrinsicID() == Intrinsic::ctpop)) {
      // ctlz.i32(x)>>5  --> zext(x == 0)
      // cttz.i32(x)>>5  --> zext(x == 0)
      // ctpop.i32(x)>>5 --> zext(x == -1)
      bool IsPop = II->getIntrinsicID() == Intrinsic::ctpop;
      Constant *RHS = ConstantInt::getSigned(Ty, IsPop ? -1 : 0);
      Value *Cmp = Builder.CreateICmpEQ(II->getArgOperand(0), RHS);
      return new ZExtInst(Cmp, Ty);
    }

    Value *X;
    const APInt *ShOp1;
    if (match(Op0, m_Shl(m_Value(X), m_APInt(ShOp1))) && ShOp1->ult(BitWidth)) {
      if (ShOp1->ult(ShAmt)) {
        unsigned ShlAmt = ShOp1->getZExtValue();
        Constant *ShiftDiff = ConstantInt::get(Ty, ShAmt - ShlAmt);
        if (cast<BinaryOperator>(Op0)->hasNoUnsignedWrap()) {
          // (X <<nuw C1) >>u C2 --> X >>u (C2 - C1)
          auto *NewLShr = BinaryOperator::CreateLShr(X, ShiftDiff);
          NewLShr->setIsExact(I.isExact());
          return NewLShr;
        }
        // (X << C1) >>u C2  --> (X >>u (C2 - C1)) & (-1 >> C2)
        Value *NewLShr = Builder.CreateLShr(X, ShiftDiff, "", I.isExact());
        APInt Mask(APInt::getLowBitsSet(BitWidth, BitWidth - ShAmt));
        return BinaryOperator::CreateAnd(NewLShr, ConstantInt::get(Ty, Mask));
      }
      if (ShOp1->ugt(ShAmt)) {
        unsigned ShlAmt = ShOp1->getZExtValue();
        Constant *ShiftDiff = ConstantInt::get(Ty, ShlAmt - ShAmt);
        if (cast<BinaryOperator>(Op0)->hasNoUnsignedWrap()) {
          // (X <<nuw C1) >>u C2 --> X <<nuw (C1 - C2)
          auto *NewShl = BinaryOperator::CreateShl(X, ShiftDiff);
          NewShl->setHasNoUnsignedWrap(true);
          return NewShl;
        }
        // (X << C1) >>u C2  --> X << (C1 - C2) & (-1 >> C2)
        Value *NewShl = Builder.CreateShl(X, ShiftDiff);
        APInt Mask(APInt::getLowBitsSet(BitWidth, BitWidth - ShAmt));
        return BinaryOperator::CreateAnd(NewShl, ConstantInt::get(Ty, Mask));
      }
      assert(*ShOp1 == ShAmt);
      // (X << C) >>u C --> X & (-1 >>u C)
      APInt Mask(APInt::getLowBitsSet(BitWidth, BitWidth - ShAmt));
      return BinaryOperator::CreateAnd(X, ConstantInt::get(Ty, Mask));
    }

    if (match(Op0, m_OneUse(m_ZExt(m_Value(X)))) &&
        (!Ty->isIntegerTy() || shouldChangeType(Ty, X->getType()))) {
      assert(ShAmt < X->getType()->getScalarSizeInBits() &&
             "Big shift not simplified to zero?");
      // lshr (zext iM X to iN), C --> zext (lshr X, C) to iN
      Value *NewLShr = Builder.CreateLShr(X, ShAmt);
      return new ZExtInst(NewLShr, Ty);
    }

    if (match(Op0, m_SExt(m_Value(X))) &&
        (!Ty->isIntegerTy() || shouldChangeType(Ty, X->getType()))) {
      // Are we moving the sign bit to the low bit and widening with high zeros?
      unsigned SrcTyBitWidth = X->getType()->getScalarSizeInBits();
      if (ShAmt == BitWidth - 1) {
        // lshr (sext i1 X to iN), N-1 --> zext X to iN
        if (SrcTyBitWidth == 1)
          return new ZExtInst(X, Ty);

        // lshr (sext iM X to iN), N-1 --> zext (lshr X, M-1) to iN
        if (Op0->hasOneUse()) {
          Value *NewLShr = Builder.CreateLShr(X, SrcTyBitWidth - 1);
          return new ZExtInst(NewLShr, Ty);
        }
      }

      // lshr (sext iM X to iN), N-M --> zext (ashr X, min(N-M, M-1)) to iN
      if (ShAmt == BitWidth - SrcTyBitWidth && Op0->hasOneUse()) {
        // The new shift amount can't be more than the narrow source type.
        unsigned NewShAmt = std::min(ShAmt, SrcTyBitWidth - 1);
        Value *AShr = Builder.CreateAShr(X, NewShAmt);
        return new ZExtInst(AShr, Ty);
      }
    }

    // lshr i32 (X -nsw Y), 31 --> zext (X < Y)
    Value *Y;
    if (ShAmt == BitWidth - 1 &&
        match(Op0, m_OneUse(m_NSWSub(m_Value(X), m_Value(Y)))))
      return new ZExtInst(Builder.CreateICmpSLT(X, Y), Ty);

    if (match(Op0, m_LShr(m_Value(X), m_APInt(ShOp1)))) {
      unsigned AmtSum = ShAmt + ShOp1->getZExtValue();
      // Oversized shifts are simplified to zero in InstSimplify.
      if (AmtSum < BitWidth)
        // (X >>u C1) >>u C2 --> X >>u (C1 + C2)
        return BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, AmtSum));
    }

    // If the shifted-out value is known-zero, then this is an exact shift.
    if (!I.isExact() &&
        MaskedValueIsZero(Op0, APInt::getLowBitsSet(BitWidth, ShAmt), 0, &I)) {
      I.setIsExact();
      return &I;
    }
  }

  // Transform  (x << y) >> y  to  x & (-1 >> y)
  Value *X;
  if (match(Op0, m_OneUse(m_Shl(m_Value(X), m_Specific(Op1))))) {
    Constant *AllOnes = ConstantInt::getAllOnesValue(Ty);
    Value *Mask = Builder.CreateLShr(AllOnes, Op1);
    return BinaryOperator::CreateAnd(Mask, X);
  }

  return nullptr;
}

Instruction *
InstCombinerImpl::foldVariableSignZeroExtensionOfVariableHighBitExtract(
    BinaryOperator &OldAShr) {
  assert(OldAShr.getOpcode() == Instruction::AShr &&
         "Must be called with arithmetic right-shift instruction only.");

  // Check that constant C is a splat of the element-wise bitwidth of V.
  auto BitWidthSplat = [](Constant *C, Value *V) {
    return match(
        C, m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ,
                              APInt(C->getType()->getScalarSizeInBits(),
                                    V->getType()->getScalarSizeInBits())));
  };

  // It should look like variable-length sign-extension on the outside:
  //   (Val << (bitwidth(Val)-Nbits)) a>> (bitwidth(Val)-Nbits)
  Value *NBits;
  Instruction *MaybeTrunc;
  Constant *C1, *C2;
  if (!match(&OldAShr,
             m_AShr(m_Shl(m_Instruction(MaybeTrunc),
                          m_ZExtOrSelf(m_Sub(m_Constant(C1),
                                             m_ZExtOrSelf(m_Value(NBits))))),
                    m_ZExtOrSelf(m_Sub(m_Constant(C2),
                                       m_ZExtOrSelf(m_Deferred(NBits)))))) ||
      !BitWidthSplat(C1, &OldAShr) || !BitWidthSplat(C2, &OldAShr))
    return nullptr;

  // There may or may not be a truncation after outer two shifts.
  Instruction *HighBitExtract;
  match(MaybeTrunc, m_TruncOrSelf(m_Instruction(HighBitExtract)));
  bool HadTrunc = MaybeTrunc != HighBitExtract;

  // And finally, the innermost part of the pattern must be a right-shift.
  Value *X, *NumLowBitsToSkip;
  if (!match(HighBitExtract, m_Shr(m_Value(X), m_Value(NumLowBitsToSkip))))
    return nullptr;

  // Said right-shift must extract high NBits bits - C0 must be it's bitwidth.
  Constant *C0;
  if (!match(NumLowBitsToSkip,
             m_ZExtOrSelf(
                 m_Sub(m_Constant(C0), m_ZExtOrSelf(m_Specific(NBits))))) ||
      !BitWidthSplat(C0, HighBitExtract))
    return nullptr;

  // Since the NBits is identical for all shifts, if the outermost and
  // innermost shifts are identical, then outermost shifts are redundant.
  // If we had truncation, do keep it though.
  if (HighBitExtract->getOpcode() == OldAShr.getOpcode())
    return replaceInstUsesWith(OldAShr, MaybeTrunc);

  // Else, if there was a truncation, then we need to ensure that one
  // instruction will go away.
  if (HadTrunc && !match(&OldAShr, m_c_BinOp(m_OneUse(m_Value()), m_Value())))
    return nullptr;

  // Finally, bypass two innermost shifts, and perform the outermost shift on
  // the operands of the innermost shift.
  Instruction *NewAShr =
      BinaryOperator::Create(OldAShr.getOpcode(), X, NumLowBitsToSkip);
  NewAShr->copyIRFlags(HighBitExtract); // We can preserve 'exact'-ness.
  if (!HadTrunc)
    return NewAShr;

  Builder.Insert(NewAShr);
  return TruncInst::CreateTruncOrBitCast(NewAShr, OldAShr.getType());
}

Instruction *InstCombinerImpl::visitAShr(BinaryOperator &I) {
  if (Value *V = SimplifyAShrInst(I.getOperand(0), I.getOperand(1), I.isExact(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *R = commonShiftTransforms(I))
    return R;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  unsigned BitWidth = Ty->getScalarSizeInBits();
  const APInt *ShAmtAPInt;
  if (match(Op1, m_APInt(ShAmtAPInt)) && ShAmtAPInt->ult(BitWidth)) {
    unsigned ShAmt = ShAmtAPInt->getZExtValue();

    // If the shift amount equals the difference in width of the destination
    // and source scalar types:
    // ashr (shl (zext X), C), C --> sext X
    Value *X;
    if (match(Op0, m_Shl(m_ZExt(m_Value(X)), m_Specific(Op1))) &&
        ShAmt == BitWidth - X->getType()->getScalarSizeInBits())
      return new SExtInst(X, Ty);

    // We can't handle (X << C1) >>s C2. It shifts arbitrary bits in. However,
    // we can handle (X <<nsw C1) >>s C2 since it only shifts in sign bits.
    const APInt *ShOp1;
    if (match(Op0, m_NSWShl(m_Value(X), m_APInt(ShOp1))) &&
        ShOp1->ult(BitWidth)) {
      unsigned ShlAmt = ShOp1->getZExtValue();
      if (ShlAmt < ShAmt) {
        // (X <<nsw C1) >>s C2 --> X >>s (C2 - C1)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShAmt - ShlAmt);
        auto *NewAShr = BinaryOperator::CreateAShr(X, ShiftDiff);
        NewAShr->setIsExact(I.isExact());
        return NewAShr;
      }
      if (ShlAmt > ShAmt) {
        // (X <<nsw C1) >>s C2 --> X <<nsw (C1 - C2)
        Constant *ShiftDiff = ConstantInt::get(Ty, ShlAmt - ShAmt);
        auto *NewShl = BinaryOperator::Create(Instruction::Shl, X, ShiftDiff);
        NewShl->setHasNoSignedWrap(true);
        return NewShl;
      }
    }

    if (match(Op0, m_AShr(m_Value(X), m_APInt(ShOp1))) &&
        ShOp1->ult(BitWidth)) {
      unsigned AmtSum = ShAmt + ShOp1->getZExtValue();
      // Oversized arithmetic shifts replicate the sign bit.
      AmtSum = std::min(AmtSum, BitWidth - 1);
      // (X >>s C1) >>s C2 --> X >>s (C1 + C2)
      return BinaryOperator::CreateAShr(X, ConstantInt::get(Ty, AmtSum));
    }

    if (match(Op0, m_OneUse(m_SExt(m_Value(X)))) &&
        (Ty->isVectorTy() || shouldChangeType(Ty, X->getType()))) {
      // ashr (sext X), C --> sext (ashr X, C')
      Type *SrcTy = X->getType();
      ShAmt = std::min(ShAmt, SrcTy->getScalarSizeInBits() - 1);
      Value *NewSh = Builder.CreateAShr(X, ConstantInt::get(SrcTy, ShAmt));
      return new SExtInst(NewSh, Ty);
    }

    // ashr i32 (X -nsw Y), 31 --> sext (X < Y)
    Value *Y;
    if (ShAmt == BitWidth - 1 &&
        match(Op0, m_OneUse(m_NSWSub(m_Value(X), m_Value(Y)))))
      return new SExtInst(Builder.CreateICmpSLT(X, Y), Ty);

    // If the shifted-out value is known-zero, then this is an exact shift.
    if (!I.isExact() &&
        MaskedValueIsZero(Op0, APInt::getLowBitsSet(BitWidth, ShAmt), 0, &I)) {
      I.setIsExact();
      return &I;
    }
  }

  if (Instruction *R = foldVariableSignZeroExtensionOfVariableHighBitExtract(I))
    return R;

  // See if we can turn a signed shr into an unsigned shr.
  if (MaskedValueIsZero(Op0, APInt::getSignMask(BitWidth), 0, &I))
    return BinaryOperator::CreateLShr(Op0, Op1);

  return nullptr;
}
