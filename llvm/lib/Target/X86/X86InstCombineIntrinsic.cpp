//===-- X86InstCombineIntrinsic.cpp - X86 specific InstCombine pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// X86 target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "X86TargetTransformInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "x86tti"

/// Return a constant boolean vector that has true elements in all positions
/// where the input constant data vector has an element with the sign bit set.
static Constant *getNegativeIsTrueBoolVec(ConstantDataVector *V) {
  SmallVector<Constant *, 32> BoolVec;
  IntegerType *BoolTy = Type::getInt1Ty(V->getContext());
  for (unsigned I = 0, E = V->getNumElements(); I != E; ++I) {
    Constant *Elt = V->getElementAsConstant(I);
    assert((isa<ConstantInt>(Elt) || isa<ConstantFP>(Elt)) &&
           "Unexpected constant data vector element type");
    bool Sign = V->getElementType()->isIntegerTy()
                    ? cast<ConstantInt>(Elt)->isNegative()
                    : cast<ConstantFP>(Elt)->isNegative();
    BoolVec.push_back(ConstantInt::get(BoolTy, Sign));
  }
  return ConstantVector::get(BoolVec);
}

// TODO: If the x86 backend knew how to convert a bool vector mask back to an
// XMM register mask efficiently, we could transform all x86 masked intrinsics
// to LLVM masked intrinsics and remove the x86 masked intrinsic defs.
static Instruction *simplifyX86MaskedLoad(IntrinsicInst &II, InstCombiner &IC) {
  Value *Ptr = II.getOperand(0);
  Value *Mask = II.getOperand(1);
  Constant *ZeroVec = Constant::getNullValue(II.getType());

  // Special case a zero mask since that's not a ConstantDataVector.
  // This masked load instruction creates a zero vector.
  if (isa<ConstantAggregateZero>(Mask))
    return IC.replaceInstUsesWith(II, ZeroVec);

  auto *ConstMask = dyn_cast<ConstantDataVector>(Mask);
  if (!ConstMask)
    return nullptr;

  // The mask is constant. Convert this x86 intrinsic to the LLVM instrinsic
  // to allow target-independent optimizations.

  // First, cast the x86 intrinsic scalar pointer to a vector pointer to match
  // the LLVM intrinsic definition for the pointer argument.
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  PointerType *VecPtrTy = PointerType::get(II.getType(), AddrSpace);
  Value *PtrCast = IC.Builder.CreateBitCast(Ptr, VecPtrTy, "castvec");

  // Second, convert the x86 XMM integer vector mask to a vector of bools based
  // on each element's most significant bit (the sign bit).
  Constant *BoolMask = getNegativeIsTrueBoolVec(ConstMask);

  // The pass-through vector for an x86 masked load is a zero vector.
  CallInst *NewMaskedLoad =
      IC.Builder.CreateMaskedLoad(PtrCast, Align(1), BoolMask, ZeroVec);
  return IC.replaceInstUsesWith(II, NewMaskedLoad);
}

// TODO: If the x86 backend knew how to convert a bool vector mask back to an
// XMM register mask efficiently, we could transform all x86 masked intrinsics
// to LLVM masked intrinsics and remove the x86 masked intrinsic defs.
static bool simplifyX86MaskedStore(IntrinsicInst &II, InstCombiner &IC) {
  Value *Ptr = II.getOperand(0);
  Value *Mask = II.getOperand(1);
  Value *Vec = II.getOperand(2);

  // Special case a zero mask since that's not a ConstantDataVector:
  // this masked store instruction does nothing.
  if (isa<ConstantAggregateZero>(Mask)) {
    IC.eraseInstFromFunction(II);
    return true;
  }

  // The SSE2 version is too weird (eg, unaligned but non-temporal) to do
  // anything else at this level.
  if (II.getIntrinsicID() == Intrinsic::x86_sse2_maskmov_dqu)
    return false;

  auto *ConstMask = dyn_cast<ConstantDataVector>(Mask);
  if (!ConstMask)
    return false;

  // The mask is constant. Convert this x86 intrinsic to the LLVM instrinsic
  // to allow target-independent optimizations.

  // First, cast the x86 intrinsic scalar pointer to a vector pointer to match
  // the LLVM intrinsic definition for the pointer argument.
  unsigned AddrSpace = cast<PointerType>(Ptr->getType())->getAddressSpace();
  PointerType *VecPtrTy = PointerType::get(Vec->getType(), AddrSpace);
  Value *PtrCast = IC.Builder.CreateBitCast(Ptr, VecPtrTy, "castvec");

  // Second, convert the x86 XMM integer vector mask to a vector of bools based
  // on each element's most significant bit (the sign bit).
  Constant *BoolMask = getNegativeIsTrueBoolVec(ConstMask);

  IC.Builder.CreateMaskedStore(Vec, PtrCast, Align(1), BoolMask);

  // 'Replace uses' doesn't work for stores. Erase the original masked store.
  IC.eraseInstFromFunction(II);
  return true;
}

static Value *simplifyX86immShift(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  bool LogicalShift = false;
  bool ShiftLeft = false;
  bool IsImm = false;

  switch (II.getIntrinsicID()) {
  default:
    llvm_unreachable("Unexpected intrinsic!");
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_avx2_psrai_d:
  case Intrinsic::x86_avx2_psrai_w:
  case Intrinsic::x86_avx512_psrai_q_128:
  case Intrinsic::x86_avx512_psrai_q_256:
  case Intrinsic::x86_avx512_psrai_d_512:
  case Intrinsic::x86_avx512_psrai_q_512:
  case Intrinsic::x86_avx512_psrai_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psra_d:
  case Intrinsic::x86_sse2_psra_w:
  case Intrinsic::x86_avx2_psra_d:
  case Intrinsic::x86_avx2_psra_w:
  case Intrinsic::x86_avx512_psra_q_128:
  case Intrinsic::x86_avx512_psra_q_256:
  case Intrinsic::x86_avx512_psra_d_512:
  case Intrinsic::x86_avx512_psra_q_512:
  case Intrinsic::x86_avx512_psra_w_512:
    LogicalShift = false;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w:
  case Intrinsic::x86_avx512_psrli_d_512:
  case Intrinsic::x86_avx512_psrli_q_512:
  case Intrinsic::x86_avx512_psrli_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx512_psrl_d_512:
  case Intrinsic::x86_avx512_psrl_q_512:
  case Intrinsic::x86_avx512_psrl_w_512:
    LogicalShift = true;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_avx512_pslli_d_512:
  case Intrinsic::x86_avx512_pslli_q_512:
  case Intrinsic::x86_avx512_pslli_w_512:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx512_psll_d_512:
  case Intrinsic::x86_avx512_psll_q_512:
  case Intrinsic::x86_avx512_psll_w_512:
    LogicalShift = true;
    ShiftLeft = true;
    break;
  }
  assert((LogicalShift || !ShiftLeft) && "Only logical shifts can shift left");

  auto Vec = II.getArgOperand(0);
  auto Amt = II.getArgOperand(1);
  auto VT = cast<VectorType>(Vec->getType());
  auto SVT = VT->getElementType();
  auto AmtVT = Amt->getType();
  unsigned VWidth = VT->getNumElements();
  unsigned BitWidth = SVT->getPrimitiveSizeInBits();

  // If the shift amount is guaranteed to be in-range we can replace it with a
  // generic shift. If its guaranteed to be out of range, logical shifts combine
  // to zero and arithmetic shifts are clamped to (BitWidth - 1).
  if (IsImm) {
    assert(AmtVT->isIntegerTy(32) && "Unexpected shift-by-immediate type");
    KnownBits KnownAmtBits =
        llvm::computeKnownBits(Amt, II.getModule()->getDataLayout());
    if (KnownAmtBits.getMaxValue().ult(BitWidth)) {
      Amt = Builder.CreateZExtOrTrunc(Amt, SVT);
      Amt = Builder.CreateVectorSplat(VWidth, Amt);
      return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                        : Builder.CreateLShr(Vec, Amt))
                           : Builder.CreateAShr(Vec, Amt));
    }
    if (KnownAmtBits.getMinValue().uge(BitWidth)) {
      if (LogicalShift)
        return ConstantAggregateZero::get(VT);
      Amt = ConstantInt::get(SVT, BitWidth - 1);
      return Builder.CreateAShr(Vec, Builder.CreateVectorSplat(VWidth, Amt));
    }
  } else {
    // Ensure the first element has an in-range value and the rest of the
    // elements in the bottom 64 bits are zero.
    assert(AmtVT->isVectorTy() && AmtVT->getPrimitiveSizeInBits() == 128 &&
           cast<VectorType>(AmtVT)->getElementType() == SVT &&
           "Unexpected shift-by-scalar type");
    unsigned NumAmtElts = cast<VectorType>(AmtVT)->getNumElements();
    APInt DemandedLower = APInt::getOneBitSet(NumAmtElts, 0);
    APInt DemandedUpper = APInt::getBitsSet(NumAmtElts, 1, NumAmtElts / 2);
    KnownBits KnownLowerBits = llvm::computeKnownBits(
        Amt, DemandedLower, II.getModule()->getDataLayout());
    KnownBits KnownUpperBits = llvm::computeKnownBits(
        Amt, DemandedUpper, II.getModule()->getDataLayout());
    if (KnownLowerBits.getMaxValue().ult(BitWidth) &&
        (DemandedUpper.isNullValue() || KnownUpperBits.isZero())) {
      SmallVector<int, 16> ZeroSplat(VWidth, 0);
      Amt = Builder.CreateShuffleVector(Amt, Amt, ZeroSplat);
      return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                        : Builder.CreateLShr(Vec, Amt))
                           : Builder.CreateAShr(Vec, Amt));
    }
  }

  // Simplify if count is constant vector.
  auto CDV = dyn_cast<ConstantDataVector>(Amt);
  if (!CDV)
    return nullptr;

  // SSE2/AVX2 uses all the first 64-bits of the 128-bit vector
  // operand to compute the shift amount.
  assert(AmtVT->isVectorTy() && AmtVT->getPrimitiveSizeInBits() == 128 &&
         cast<VectorType>(AmtVT)->getElementType() == SVT &&
         "Unexpected shift-by-scalar type");

  // Concatenate the sub-elements to create the 64-bit value.
  APInt Count(64, 0);
  for (unsigned i = 0, NumSubElts = 64 / BitWidth; i != NumSubElts; ++i) {
    unsigned SubEltIdx = (NumSubElts - 1) - i;
    auto SubElt = cast<ConstantInt>(CDV->getElementAsConstant(SubEltIdx));
    Count <<= BitWidth;
    Count |= SubElt->getValue().zextOrTrunc(64);
  }

  // If shift-by-zero then just return the original value.
  if (Count.isNullValue())
    return Vec;

  // Handle cases when Shift >= BitWidth.
  if (Count.uge(BitWidth)) {
    // If LogicalShift - just return zero.
    if (LogicalShift)
      return ConstantAggregateZero::get(VT);

    // If ArithmeticShift - clamp Shift to (BitWidth - 1).
    Count = APInt(64, BitWidth - 1);
  }

  // Get a constant vector of the same type as the first operand.
  auto ShiftAmt = ConstantInt::get(SVT, Count.zextOrTrunc(BitWidth));
  auto ShiftVec = Builder.CreateVectorSplat(VWidth, ShiftAmt);

  if (ShiftLeft)
    return Builder.CreateShl(Vec, ShiftVec);

  if (LogicalShift)
    return Builder.CreateLShr(Vec, ShiftVec);

  return Builder.CreateAShr(Vec, ShiftVec);
}

// Attempt to simplify AVX2 per-element shift intrinsics to a generic IR shift.
// Unlike the generic IR shifts, the intrinsics have defined behaviour for out
// of range shift amounts (logical - set to zero, arithmetic - splat sign bit).
static Value *simplifyX86varShift(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  bool LogicalShift = false;
  bool ShiftLeft = false;

  switch (II.getIntrinsicID()) {
  default:
    llvm_unreachable("Unexpected intrinsic!");
  case Intrinsic::x86_avx2_psrav_d:
  case Intrinsic::x86_avx2_psrav_d_256:
  case Intrinsic::x86_avx512_psrav_q_128:
  case Intrinsic::x86_avx512_psrav_q_256:
  case Intrinsic::x86_avx512_psrav_d_512:
  case Intrinsic::x86_avx512_psrav_q_512:
  case Intrinsic::x86_avx512_psrav_w_128:
  case Intrinsic::x86_avx512_psrav_w_256:
  case Intrinsic::x86_avx512_psrav_w_512:
    LogicalShift = false;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_avx2_psrlv_d:
  case Intrinsic::x86_avx2_psrlv_d_256:
  case Intrinsic::x86_avx2_psrlv_q:
  case Intrinsic::x86_avx2_psrlv_q_256:
  case Intrinsic::x86_avx512_psrlv_d_512:
  case Intrinsic::x86_avx512_psrlv_q_512:
  case Intrinsic::x86_avx512_psrlv_w_128:
  case Intrinsic::x86_avx512_psrlv_w_256:
  case Intrinsic::x86_avx512_psrlv_w_512:
    LogicalShift = true;
    ShiftLeft = false;
    break;
  case Intrinsic::x86_avx2_psllv_d:
  case Intrinsic::x86_avx2_psllv_d_256:
  case Intrinsic::x86_avx2_psllv_q:
  case Intrinsic::x86_avx2_psllv_q_256:
  case Intrinsic::x86_avx512_psllv_d_512:
  case Intrinsic::x86_avx512_psllv_q_512:
  case Intrinsic::x86_avx512_psllv_w_128:
  case Intrinsic::x86_avx512_psllv_w_256:
  case Intrinsic::x86_avx512_psllv_w_512:
    LogicalShift = true;
    ShiftLeft = true;
    break;
  }
  assert((LogicalShift || !ShiftLeft) && "Only logical shifts can shift left");

  auto Vec = II.getArgOperand(0);
  auto Amt = II.getArgOperand(1);
  auto VT = cast<VectorType>(II.getType());
  auto SVT = VT->getElementType();
  int NumElts = VT->getNumElements();
  int BitWidth = SVT->getIntegerBitWidth();

  // If the shift amount is guaranteed to be in-range we can replace it with a
  // generic shift.
  APInt UpperBits =
      APInt::getHighBitsSet(BitWidth, BitWidth - Log2_32(BitWidth));
  if (llvm::MaskedValueIsZero(Amt, UpperBits,
                              II.getModule()->getDataLayout())) {
    return (LogicalShift ? (ShiftLeft ? Builder.CreateShl(Vec, Amt)
                                      : Builder.CreateLShr(Vec, Amt))
                         : Builder.CreateAShr(Vec, Amt));
  }

  // Simplify if all shift amounts are constant/undef.
  auto *CShift = dyn_cast<Constant>(Amt);
  if (!CShift)
    return nullptr;

  // Collect each element's shift amount.
  // We also collect special cases: UNDEF = -1, OUT-OF-RANGE = BitWidth.
  bool AnyOutOfRange = false;
  SmallVector<int, 8> ShiftAmts;
  for (int I = 0; I < NumElts; ++I) {
    auto *CElt = CShift->getAggregateElement(I);
    if (CElt && isa<UndefValue>(CElt)) {
      ShiftAmts.push_back(-1);
      continue;
    }

    auto *COp = dyn_cast_or_null<ConstantInt>(CElt);
    if (!COp)
      return nullptr;

    // Handle out of range shifts.
    // If LogicalShift - set to BitWidth (special case).
    // If ArithmeticShift - set to (BitWidth - 1) (sign splat).
    APInt ShiftVal = COp->getValue();
    if (ShiftVal.uge(BitWidth)) {
      AnyOutOfRange = LogicalShift;
      ShiftAmts.push_back(LogicalShift ? BitWidth : BitWidth - 1);
      continue;
    }

    ShiftAmts.push_back((int)ShiftVal.getZExtValue());
  }

  // If all elements out of range or UNDEF, return vector of zeros/undefs.
  // ArithmeticShift should only hit this if they are all UNDEF.
  auto OutOfRange = [&](int Idx) { return (Idx < 0) || (BitWidth <= Idx); };
  if (llvm::all_of(ShiftAmts, OutOfRange)) {
    SmallVector<Constant *, 8> ConstantVec;
    for (int Idx : ShiftAmts) {
      if (Idx < 0) {
        ConstantVec.push_back(UndefValue::get(SVT));
      } else {
        assert(LogicalShift && "Logical shift expected");
        ConstantVec.push_back(ConstantInt::getNullValue(SVT));
      }
    }
    return ConstantVector::get(ConstantVec);
  }

  // We can't handle only some out of range values with generic logical shifts.
  if (AnyOutOfRange)
    return nullptr;

  // Build the shift amount constant vector.
  SmallVector<Constant *, 8> ShiftVecAmts;
  for (int Idx : ShiftAmts) {
    if (Idx < 0)
      ShiftVecAmts.push_back(UndefValue::get(SVT));
    else
      ShiftVecAmts.push_back(ConstantInt::get(SVT, Idx));
  }
  auto ShiftVec = ConstantVector::get(ShiftVecAmts);

  if (ShiftLeft)
    return Builder.CreateShl(Vec, ShiftVec);

  if (LogicalShift)
    return Builder.CreateLShr(Vec, ShiftVec);

  return Builder.CreateAShr(Vec, ShiftVec);
}

static Value *simplifyX86pack(IntrinsicInst &II,
                              InstCombiner::BuilderTy &Builder, bool IsSigned) {
  Value *Arg0 = II.getArgOperand(0);
  Value *Arg1 = II.getArgOperand(1);
  Type *ResTy = II.getType();

  // Fast all undef handling.
  if (isa<UndefValue>(Arg0) && isa<UndefValue>(Arg1))
    return UndefValue::get(ResTy);

  auto *ArgTy = cast<VectorType>(Arg0->getType());
  unsigned NumLanes = ResTy->getPrimitiveSizeInBits() / 128;
  unsigned NumSrcElts = ArgTy->getNumElements();
  assert(cast<VectorType>(ResTy)->getNumElements() == (2 * NumSrcElts) &&
         "Unexpected packing types");

  unsigned NumSrcEltsPerLane = NumSrcElts / NumLanes;
  unsigned DstScalarSizeInBits = ResTy->getScalarSizeInBits();
  unsigned SrcScalarSizeInBits = ArgTy->getScalarSizeInBits();
  assert(SrcScalarSizeInBits == (2 * DstScalarSizeInBits) &&
         "Unexpected packing types");

  // Constant folding.
  if (!isa<Constant>(Arg0) || !isa<Constant>(Arg1))
    return nullptr;

  // Clamp Values - signed/unsigned both use signed clamp values, but they
  // differ on the min/max values.
  APInt MinValue, MaxValue;
  if (IsSigned) {
    // PACKSS: Truncate signed value with signed saturation.
    // Source values less than dst minint are saturated to minint.
    // Source values greater than dst maxint are saturated to maxint.
    MinValue =
        APInt::getSignedMinValue(DstScalarSizeInBits).sext(SrcScalarSizeInBits);
    MaxValue =
        APInt::getSignedMaxValue(DstScalarSizeInBits).sext(SrcScalarSizeInBits);
  } else {
    // PACKUS: Truncate signed value with unsigned saturation.
    // Source values less than zero are saturated to zero.
    // Source values greater than dst maxuint are saturated to maxuint.
    MinValue = APInt::getNullValue(SrcScalarSizeInBits);
    MaxValue = APInt::getLowBitsSet(SrcScalarSizeInBits, DstScalarSizeInBits);
  }

  auto *MinC = Constant::getIntegerValue(ArgTy, MinValue);
  auto *MaxC = Constant::getIntegerValue(ArgTy, MaxValue);
  Arg0 = Builder.CreateSelect(Builder.CreateICmpSLT(Arg0, MinC), MinC, Arg0);
  Arg1 = Builder.CreateSelect(Builder.CreateICmpSLT(Arg1, MinC), MinC, Arg1);
  Arg0 = Builder.CreateSelect(Builder.CreateICmpSGT(Arg0, MaxC), MaxC, Arg0);
  Arg1 = Builder.CreateSelect(Builder.CreateICmpSGT(Arg1, MaxC), MaxC, Arg1);

  // Shuffle clamped args together at the lane level.
  SmallVector<int, 32> PackMask;
  for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
    for (unsigned Elt = 0; Elt != NumSrcEltsPerLane; ++Elt)
      PackMask.push_back(Elt + (Lane * NumSrcEltsPerLane));
    for (unsigned Elt = 0; Elt != NumSrcEltsPerLane; ++Elt)
      PackMask.push_back(Elt + (Lane * NumSrcEltsPerLane) + NumSrcElts);
  }
  auto *Shuffle = Builder.CreateShuffleVector(Arg0, Arg1, PackMask);

  // Truncate to dst size.
  return Builder.CreateTrunc(Shuffle, ResTy);
}

static Value *simplifyX86movmsk(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  Value *Arg = II.getArgOperand(0);
  Type *ResTy = II.getType();

  // movmsk(undef) -> zero as we must ensure the upper bits are zero.
  if (isa<UndefValue>(Arg))
    return Constant::getNullValue(ResTy);

  auto *ArgTy = dyn_cast<VectorType>(Arg->getType());
  // We can't easily peek through x86_mmx types.
  if (!ArgTy)
    return nullptr;

  // Expand MOVMSK to compare/bitcast/zext:
  // e.g. PMOVMSKB(v16i8 x):
  // %cmp = icmp slt <16 x i8> %x, zeroinitializer
  // %int = bitcast <16 x i1> %cmp to i16
  // %res = zext i16 %int to i32
  unsigned NumElts = ArgTy->getNumElements();
  Type *IntegerVecTy = VectorType::getInteger(ArgTy);
  Type *IntegerTy = Builder.getIntNTy(NumElts);

  Value *Res = Builder.CreateBitCast(Arg, IntegerVecTy);
  Res = Builder.CreateICmpSLT(Res, Constant::getNullValue(IntegerVecTy));
  Res = Builder.CreateBitCast(Res, IntegerTy);
  Res = Builder.CreateZExtOrTrunc(Res, ResTy);
  return Res;
}

static Value *simplifyX86addcarry(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  Value *CarryIn = II.getArgOperand(0);
  Value *Op1 = II.getArgOperand(1);
  Value *Op2 = II.getArgOperand(2);
  Type *RetTy = II.getType();
  Type *OpTy = Op1->getType();
  assert(RetTy->getStructElementType(0)->isIntegerTy(8) &&
         RetTy->getStructElementType(1) == OpTy && OpTy == Op2->getType() &&
         "Unexpected types for x86 addcarry");

  // If carry-in is zero, this is just an unsigned add with overflow.
  if (match(CarryIn, PatternMatch::m_ZeroInt())) {
    Value *UAdd = Builder.CreateIntrinsic(Intrinsic::uadd_with_overflow, OpTy,
                                          {Op1, Op2});
    // The types have to be adjusted to match the x86 call types.
    Value *UAddResult = Builder.CreateExtractValue(UAdd, 0);
    Value *UAddOV = Builder.CreateZExt(Builder.CreateExtractValue(UAdd, 1),
                                       Builder.getInt8Ty());
    Value *Res = UndefValue::get(RetTy);
    Res = Builder.CreateInsertValue(Res, UAddOV, 0);
    return Builder.CreateInsertValue(Res, UAddResult, 1);
  }

  return nullptr;
}

static Value *simplifyX86insertps(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  auto *CInt = dyn_cast<ConstantInt>(II.getArgOperand(2));
  if (!CInt)
    return nullptr;

  VectorType *VecTy = cast<VectorType>(II.getType());
  assert(VecTy->getNumElements() == 4 && "insertps with wrong vector type");

  // The immediate permute control byte looks like this:
  //    [3:0] - zero mask for each 32-bit lane
  //    [5:4] - select one 32-bit destination lane
  //    [7:6] - select one 32-bit source lane

  uint8_t Imm = CInt->getZExtValue();
  uint8_t ZMask = Imm & 0xf;
  uint8_t DestLane = (Imm >> 4) & 0x3;
  uint8_t SourceLane = (Imm >> 6) & 0x3;

  ConstantAggregateZero *ZeroVector = ConstantAggregateZero::get(VecTy);

  // If all zero mask bits are set, this was just a weird way to
  // generate a zero vector.
  if (ZMask == 0xf)
    return ZeroVector;

  // Initialize by passing all of the first source bits through.
  int ShuffleMask[4] = {0, 1, 2, 3};

  // We may replace the second operand with the zero vector.
  Value *V1 = II.getArgOperand(1);

  if (ZMask) {
    // If the zero mask is being used with a single input or the zero mask
    // overrides the destination lane, this is a shuffle with the zero vector.
    if ((II.getArgOperand(0) == II.getArgOperand(1)) ||
        (ZMask & (1 << DestLane))) {
      V1 = ZeroVector;
      // We may still move 32-bits of the first source vector from one lane
      // to another.
      ShuffleMask[DestLane] = SourceLane;
      // The zero mask may override the previous insert operation.
      for (unsigned i = 0; i < 4; ++i)
        if ((ZMask >> i) & 0x1)
          ShuffleMask[i] = i + 4;
    } else {
      // TODO: Model this case as 2 shuffles or a 'logical and' plus shuffle?
      return nullptr;
    }
  } else {
    // Replace the selected destination lane with the selected source lane.
    ShuffleMask[DestLane] = SourceLane + 4;
  }

  return Builder.CreateShuffleVector(II.getArgOperand(0), V1, ShuffleMask);
}

/// Attempt to simplify SSE4A EXTRQ/EXTRQI instructions using constant folding
/// or conversion to a shuffle vector.
static Value *simplifyX86extrq(IntrinsicInst &II, Value *Op0,
                               ConstantInt *CILength, ConstantInt *CIIndex,
                               InstCombiner::BuilderTy &Builder) {
  auto LowConstantHighUndef = [&](uint64_t Val) {
    Type *IntTy64 = Type::getInt64Ty(II.getContext());
    Constant *Args[] = {ConstantInt::get(IntTy64, Val),
                        UndefValue::get(IntTy64)};
    return ConstantVector::get(Args);
  };

  // See if we're dealing with constant values.
  Constant *C0 = dyn_cast<Constant>(Op0);
  ConstantInt *CI0 =
      C0 ? dyn_cast_or_null<ConstantInt>(C0->getAggregateElement((unsigned)0))
         : nullptr;

  // Attempt to constant fold.
  if (CILength && CIIndex) {
    // From AMD documentation: "The bit index and field length are each six
    // bits in length other bits of the field are ignored."
    APInt APIndex = CIIndex->getValue().zextOrTrunc(6);
    APInt APLength = CILength->getValue().zextOrTrunc(6);

    unsigned Index = APIndex.getZExtValue();

    // From AMD documentation: "a value of zero in the field length is
    // defined as length of 64".
    unsigned Length = APLength == 0 ? 64 : APLength.getZExtValue();

    // From AMD documentation: "If the sum of the bit index + length field
    // is greater than 64, the results are undefined".
    unsigned End = Index + Length;

    // Note that both field index and field length are 8-bit quantities.
    // Since variables 'Index' and 'Length' are unsigned values
    // obtained from zero-extending field index and field length
    // respectively, their sum should never wrap around.
    if (End > 64)
      return UndefValue::get(II.getType());

    // If we are inserting whole bytes, we can convert this to a shuffle.
    // Lowering can recognize EXTRQI shuffle masks.
    if ((Length % 8) == 0 && (Index % 8) == 0) {
      // Convert bit indices to byte indices.
      Length /= 8;
      Index /= 8;

      Type *IntTy8 = Type::getInt8Ty(II.getContext());
      auto *ShufTy = FixedVectorType::get(IntTy8, 16);

      SmallVector<int, 16> ShuffleMask;
      for (int i = 0; i != (int)Length; ++i)
        ShuffleMask.push_back(i + Index);
      for (int i = Length; i != 8; ++i)
        ShuffleMask.push_back(i + 16);
      for (int i = 8; i != 16; ++i)
        ShuffleMask.push_back(-1);

      Value *SV = Builder.CreateShuffleVector(
          Builder.CreateBitCast(Op0, ShufTy),
          ConstantAggregateZero::get(ShufTy), ShuffleMask);
      return Builder.CreateBitCast(SV, II.getType());
    }

    // Constant Fold - shift Index'th bit to lowest position and mask off
    // Length bits.
    if (CI0) {
      APInt Elt = CI0->getValue();
      Elt.lshrInPlace(Index);
      Elt = Elt.zextOrTrunc(Length);
      return LowConstantHighUndef(Elt.getZExtValue());
    }

    // If we were an EXTRQ call, we'll save registers if we convert to EXTRQI.
    if (II.getIntrinsicID() == Intrinsic::x86_sse4a_extrq) {
      Value *Args[] = {Op0, CILength, CIIndex};
      Module *M = II.getModule();
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::x86_sse4a_extrqi);
      return Builder.CreateCall(F, Args);
    }
  }

  // Constant Fold - extraction from zero is always {zero, undef}.
  if (CI0 && CI0->isZero())
    return LowConstantHighUndef(0);

  return nullptr;
}

/// Attempt to simplify SSE4A INSERTQ/INSERTQI instructions using constant
/// folding or conversion to a shuffle vector.
static Value *simplifyX86insertq(IntrinsicInst &II, Value *Op0, Value *Op1,
                                 APInt APLength, APInt APIndex,
                                 InstCombiner::BuilderTy &Builder) {
  // From AMD documentation: "The bit index and field length are each six bits
  // in length other bits of the field are ignored."
  APIndex = APIndex.zextOrTrunc(6);
  APLength = APLength.zextOrTrunc(6);

  // Attempt to constant fold.
  unsigned Index = APIndex.getZExtValue();

  // From AMD documentation: "a value of zero in the field length is
  // defined as length of 64".
  unsigned Length = APLength == 0 ? 64 : APLength.getZExtValue();

  // From AMD documentation: "If the sum of the bit index + length field
  // is greater than 64, the results are undefined".
  unsigned End = Index + Length;

  // Note that both field index and field length are 8-bit quantities.
  // Since variables 'Index' and 'Length' are unsigned values
  // obtained from zero-extending field index and field length
  // respectively, their sum should never wrap around.
  if (End > 64)
    return UndefValue::get(II.getType());

  // If we are inserting whole bytes, we can convert this to a shuffle.
  // Lowering can recognize INSERTQI shuffle masks.
  if ((Length % 8) == 0 && (Index % 8) == 0) {
    // Convert bit indices to byte indices.
    Length /= 8;
    Index /= 8;

    Type *IntTy8 = Type::getInt8Ty(II.getContext());
    auto *ShufTy = FixedVectorType::get(IntTy8, 16);

    SmallVector<int, 16> ShuffleMask;
    for (int i = 0; i != (int)Index; ++i)
      ShuffleMask.push_back(i);
    for (int i = 0; i != (int)Length; ++i)
      ShuffleMask.push_back(i + 16);
    for (int i = Index + Length; i != 8; ++i)
      ShuffleMask.push_back(i);
    for (int i = 8; i != 16; ++i)
      ShuffleMask.push_back(-1);

    Value *SV = Builder.CreateShuffleVector(Builder.CreateBitCast(Op0, ShufTy),
                                            Builder.CreateBitCast(Op1, ShufTy),
                                            ShuffleMask);
    return Builder.CreateBitCast(SV, II.getType());
  }

  // See if we're dealing with constant values.
  Constant *C0 = dyn_cast<Constant>(Op0);
  Constant *C1 = dyn_cast<Constant>(Op1);
  ConstantInt *CI00 =
      C0 ? dyn_cast_or_null<ConstantInt>(C0->getAggregateElement((unsigned)0))
         : nullptr;
  ConstantInt *CI10 =
      C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)0))
         : nullptr;

  // Constant Fold - insert bottom Length bits starting at the Index'th bit.
  if (CI00 && CI10) {
    APInt V00 = CI00->getValue();
    APInt V10 = CI10->getValue();
    APInt Mask = APInt::getLowBitsSet(64, Length).shl(Index);
    V00 = V00 & ~Mask;
    V10 = V10.zextOrTrunc(Length).zextOrTrunc(64).shl(Index);
    APInt Val = V00 | V10;
    Type *IntTy64 = Type::getInt64Ty(II.getContext());
    Constant *Args[] = {ConstantInt::get(IntTy64, Val.getZExtValue()),
                        UndefValue::get(IntTy64)};
    return ConstantVector::get(Args);
  }

  // If we were an INSERTQ call, we'll save demanded elements if we convert to
  // INSERTQI.
  if (II.getIntrinsicID() == Intrinsic::x86_sse4a_insertq) {
    Type *IntTy8 = Type::getInt8Ty(II.getContext());
    Constant *CILength = ConstantInt::get(IntTy8, Length, false);
    Constant *CIIndex = ConstantInt::get(IntTy8, Index, false);

    Value *Args[] = {Op0, Op1, CILength, CIIndex};
    Module *M = II.getModule();
    Function *F = Intrinsic::getDeclaration(M, Intrinsic::x86_sse4a_insertqi);
    return Builder.CreateCall(F, Args);
  }

  return nullptr;
}

/// Attempt to convert pshufb* to shufflevector if the mask is constant.
static Value *simplifyX86pshufb(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  Constant *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  assert((NumElts == 16 || NumElts == 32 || NumElts == 64) &&
         "Unexpected number of elements in shuffle mask!");

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[64];

  // Each byte in the shuffle control mask forms an index to permute the
  // corresponding byte in the destination operand.
  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    int8_t Index = cast<ConstantInt>(COp)->getValue().getZExtValue();

    // If the most significant bit (bit[7]) of each byte of the shuffle
    // control mask is set, then zero is written in the result byte.
    // The zero vector is in the right-hand side of the resulting
    // shufflevector.

    // The value of each index for the high 128-bit lane is the least
    // significant 4 bits of the respective shuffle control byte.
    Index = ((Index < 0) ? NumElts : Index & 0x0F) + (I & 0xF0);
    Indexes[I] = Index;
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = Constant::getNullValue(VecTy);
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, NumElts));
}

/// Attempt to convert vpermilvar* to shufflevector if the mask is constant.
static Value *simplifyX86vpermilvar(const IntrinsicInst &II,
                                    InstCombiner::BuilderTy &Builder) {
  Constant *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned NumElts = VecTy->getNumElements();
  bool IsPD = VecTy->getScalarType()->isDoubleTy();
  unsigned NumLaneElts = IsPD ? 2 : 4;
  assert(NumElts == 16 || NumElts == 8 || NumElts == 4 || NumElts == 2);

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[16];

  // The intrinsics only read one or two bits, clear the rest.
  for (unsigned I = 0; I < NumElts; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    APInt Index = cast<ConstantInt>(COp)->getValue();
    Index = Index.zextOrTrunc(32).getLoBits(2);

    // The PD variants uses bit 1 to select per-lane element index, so
    // shift down to convert to generic shuffle mask index.
    if (IsPD)
      Index.lshrInPlace(1);

    // The _256 variants are a bit trickier since the mask bits always index
    // into the corresponding 128 half. In order to convert to a generic
    // shuffle, we have to make that explicit.
    Index += APInt(32, (I / NumLaneElts) * NumLaneElts);

    Indexes[I] = Index.getZExtValue();
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = UndefValue::get(V1->getType());
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, NumElts));
}

/// Attempt to convert vpermd/vpermps to shufflevector if the mask is constant.
static Value *simplifyX86vpermv(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  auto *V = dyn_cast<Constant>(II.getArgOperand(1));
  if (!V)
    return nullptr;

  auto *VecTy = cast<VectorType>(II.getType());
  unsigned Size = VecTy->getNumElements();
  assert((Size == 4 || Size == 8 || Size == 16 || Size == 32 || Size == 64) &&
         "Unexpected shuffle mask size");

  // Construct a shuffle mask from constant integers or UNDEFs.
  int Indexes[64];

  for (unsigned I = 0; I < Size; ++I) {
    Constant *COp = V->getAggregateElement(I);
    if (!COp || (!isa<UndefValue>(COp) && !isa<ConstantInt>(COp)))
      return nullptr;

    if (isa<UndefValue>(COp)) {
      Indexes[I] = -1;
      continue;
    }

    uint32_t Index = cast<ConstantInt>(COp)->getZExtValue();
    Index &= Size - 1;
    Indexes[I] = Index;
  }

  auto V1 = II.getArgOperand(0);
  auto V2 = UndefValue::get(VecTy);
  return Builder.CreateShuffleVector(V1, V2, makeArrayRef(Indexes, Size));
}

Optional<Instruction *>
X86TTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  auto SimplifyDemandedVectorEltsLow = [&IC](Value *Op, unsigned Width,
                                             unsigned DemandedWidth) {
    APInt UndefElts(Width, 0);
    APInt DemandedElts = APInt::getLowBitsSet(Width, DemandedWidth);
    return IC.SimplifyDemandedVectorElts(Op, DemandedElts, UndefElts);
  };

  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  case Intrinsic::x86_bmi_bextr_32:
  case Intrinsic::x86_bmi_bextr_64:
  case Intrinsic::x86_tbm_bextri_u32:
  case Intrinsic::x86_tbm_bextri_u64:
    // If the RHS is a constant we can try some simplifications.
    if (auto *C = dyn_cast<ConstantInt>(II.getArgOperand(1))) {
      uint64_t Shift = C->getZExtValue();
      uint64_t Length = (Shift >> 8) & 0xff;
      Shift &= 0xff;
      unsigned BitWidth = II.getType()->getIntegerBitWidth();
      // If the length is 0 or the shift is out of range, replace with zero.
      if (Length == 0 || Shift >= BitWidth) {
        return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), 0));
      }
      // If the LHS is also a constant, we can completely constant fold this.
      if (auto *InC = dyn_cast<ConstantInt>(II.getArgOperand(0))) {
        uint64_t Result = InC->getZExtValue() >> Shift;
        if (Length > BitWidth)
          Length = BitWidth;
        Result &= maskTrailingOnes<uint64_t>(Length);
        return IC.replaceInstUsesWith(II,
                                      ConstantInt::get(II.getType(), Result));
      }
      // TODO should we turn this into 'and' if shift is 0? Or 'shl' if we
      // are only masking bits that a shift already cleared?
    }
    break;

  case Intrinsic::x86_bmi_bzhi_32:
  case Intrinsic::x86_bmi_bzhi_64:
    // If the RHS is a constant we can try some simplifications.
    if (auto *C = dyn_cast<ConstantInt>(II.getArgOperand(1))) {
      uint64_t Index = C->getZExtValue() & 0xff;
      unsigned BitWidth = II.getType()->getIntegerBitWidth();
      if (Index >= BitWidth) {
        return IC.replaceInstUsesWith(II, II.getArgOperand(0));
      }
      if (Index == 0) {
        return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), 0));
      }
      // If the LHS is also a constant, we can completely constant fold this.
      if (auto *InC = dyn_cast<ConstantInt>(II.getArgOperand(0))) {
        uint64_t Result = InC->getZExtValue();
        Result &= maskTrailingOnes<uint64_t>(Index);
        return IC.replaceInstUsesWith(II,
                                      ConstantInt::get(II.getType(), Result));
      }
      // TODO should we convert this to an AND if the RHS is constant?
    }
    break;
  case Intrinsic::x86_bmi_pext_32:
  case Intrinsic::x86_bmi_pext_64:
    if (auto *MaskC = dyn_cast<ConstantInt>(II.getArgOperand(1))) {
      if (MaskC->isNullValue()) {
        return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), 0));
      }
      if (MaskC->isAllOnesValue()) {
        return IC.replaceInstUsesWith(II, II.getArgOperand(0));
      }

      if (auto *SrcC = dyn_cast<ConstantInt>(II.getArgOperand(0))) {
        uint64_t Src = SrcC->getZExtValue();
        uint64_t Mask = MaskC->getZExtValue();
        uint64_t Result = 0;
        uint64_t BitToSet = 1;

        while (Mask) {
          // Isolate lowest set bit.
          uint64_t BitToTest = Mask & -Mask;
          if (BitToTest & Src)
            Result |= BitToSet;

          BitToSet <<= 1;
          // Clear lowest set bit.
          Mask &= Mask - 1;
        }

        return IC.replaceInstUsesWith(II,
                                      ConstantInt::get(II.getType(), Result));
      }
    }
    break;
  case Intrinsic::x86_bmi_pdep_32:
  case Intrinsic::x86_bmi_pdep_64:
    if (auto *MaskC = dyn_cast<ConstantInt>(II.getArgOperand(1))) {
      if (MaskC->isNullValue()) {
        return IC.replaceInstUsesWith(II, ConstantInt::get(II.getType(), 0));
      }
      if (MaskC->isAllOnesValue()) {
        return IC.replaceInstUsesWith(II, II.getArgOperand(0));
      }

      if (auto *SrcC = dyn_cast<ConstantInt>(II.getArgOperand(0))) {
        uint64_t Src = SrcC->getZExtValue();
        uint64_t Mask = MaskC->getZExtValue();
        uint64_t Result = 0;
        uint64_t BitToTest = 1;

        while (Mask) {
          // Isolate lowest set bit.
          uint64_t BitToSet = Mask & -Mask;
          if (BitToTest & Src)
            Result |= BitToSet;

          BitToTest <<= 1;
          // Clear lowest set bit;
          Mask &= Mask - 1;
        }

        return IC.replaceInstUsesWith(II,
                                      ConstantInt::get(II.getType(), Result));
      }
    }
    break;

  case Intrinsic::x86_sse_cvtss2si:
  case Intrinsic::x86_sse_cvtss2si64:
  case Intrinsic::x86_sse_cvttss2si:
  case Intrinsic::x86_sse_cvttss2si64:
  case Intrinsic::x86_sse2_cvtsd2si:
  case Intrinsic::x86_sse2_cvtsd2si64:
  case Intrinsic::x86_sse2_cvttsd2si:
  case Intrinsic::x86_sse2_cvttsd2si64:
  case Intrinsic::x86_avx512_vcvtss2si32:
  case Intrinsic::x86_avx512_vcvtss2si64:
  case Intrinsic::x86_avx512_vcvtss2usi32:
  case Intrinsic::x86_avx512_vcvtss2usi64:
  case Intrinsic::x86_avx512_vcvtsd2si32:
  case Intrinsic::x86_avx512_vcvtsd2si64:
  case Intrinsic::x86_avx512_vcvtsd2usi32:
  case Intrinsic::x86_avx512_vcvtsd2usi64:
  case Intrinsic::x86_avx512_cvttss2si:
  case Intrinsic::x86_avx512_cvttss2si64:
  case Intrinsic::x86_avx512_cvttss2usi:
  case Intrinsic::x86_avx512_cvttss2usi64:
  case Intrinsic::x86_avx512_cvttsd2si:
  case Intrinsic::x86_avx512_cvttsd2si64:
  case Intrinsic::x86_avx512_cvttsd2usi:
  case Intrinsic::x86_avx512_cvttsd2usi64: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    Value *Arg = II.getArgOperand(0);
    unsigned VWidth = cast<VectorType>(Arg->getType())->getNumElements();
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg, VWidth, 1)) {
      return IC.replaceOperand(II, 0, V);
    }
    break;
  }

  case Intrinsic::x86_mmx_pmovmskb:
  case Intrinsic::x86_sse_movmsk_ps:
  case Intrinsic::x86_sse2_movmsk_pd:
  case Intrinsic::x86_sse2_pmovmskb_128:
  case Intrinsic::x86_avx_movmsk_pd_256:
  case Intrinsic::x86_avx_movmsk_ps_256:
  case Intrinsic::x86_avx2_pmovmskb:
    if (Value *V = simplifyX86movmsk(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_sse_comieq_ss:
  case Intrinsic::x86_sse_comige_ss:
  case Intrinsic::x86_sse_comigt_ss:
  case Intrinsic::x86_sse_comile_ss:
  case Intrinsic::x86_sse_comilt_ss:
  case Intrinsic::x86_sse_comineq_ss:
  case Intrinsic::x86_sse_ucomieq_ss:
  case Intrinsic::x86_sse_ucomige_ss:
  case Intrinsic::x86_sse_ucomigt_ss:
  case Intrinsic::x86_sse_ucomile_ss:
  case Intrinsic::x86_sse_ucomilt_ss:
  case Intrinsic::x86_sse_ucomineq_ss:
  case Intrinsic::x86_sse2_comieq_sd:
  case Intrinsic::x86_sse2_comige_sd:
  case Intrinsic::x86_sse2_comigt_sd:
  case Intrinsic::x86_sse2_comile_sd:
  case Intrinsic::x86_sse2_comilt_sd:
  case Intrinsic::x86_sse2_comineq_sd:
  case Intrinsic::x86_sse2_ucomieq_sd:
  case Intrinsic::x86_sse2_ucomige_sd:
  case Intrinsic::x86_sse2_ucomigt_sd:
  case Intrinsic::x86_sse2_ucomile_sd:
  case Intrinsic::x86_sse2_ucomilt_sd:
  case Intrinsic::x86_sse2_ucomineq_sd:
  case Intrinsic::x86_avx512_vcomi_ss:
  case Intrinsic::x86_avx512_vcomi_sd:
  case Intrinsic::x86_avx512_mask_cmp_ss:
  case Intrinsic::x86_avx512_mask_cmp_sd: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    bool MadeChange = false;
    Value *Arg0 = II.getArgOperand(0);
    Value *Arg1 = II.getArgOperand(1);
    unsigned VWidth = cast<VectorType>(Arg0->getType())->getNumElements();
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg0, VWidth, 1)) {
      IC.replaceOperand(II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Arg1, VWidth, 1)) {
      IC.replaceOperand(II, 1, V);
      MadeChange = true;
    }
    if (MadeChange) {
      return &II;
    }
    break;
  }
  case Intrinsic::x86_avx512_cmp_pd_128:
  case Intrinsic::x86_avx512_cmp_pd_256:
  case Intrinsic::x86_avx512_cmp_pd_512:
  case Intrinsic::x86_avx512_cmp_ps_128:
  case Intrinsic::x86_avx512_cmp_ps_256:
  case Intrinsic::x86_avx512_cmp_ps_512: {
    // Folding cmp(sub(a,b),0) -> cmp(a,b) and cmp(0,sub(a,b)) -> cmp(b,a)
    Value *Arg0 = II.getArgOperand(0);
    Value *Arg1 = II.getArgOperand(1);
    bool Arg0IsZero = match(Arg0, PatternMatch::m_PosZeroFP());
    if (Arg0IsZero)
      std::swap(Arg0, Arg1);
    Value *A, *B;
    // This fold requires only the NINF(not +/- inf) since inf minus
    // inf is nan.
    // NSZ(No Signed Zeros) is not needed because zeros of any sign are
    // equal for both compares.
    // NNAN is not needed because nans compare the same for both compares.
    // The compare intrinsic uses the above assumptions and therefore
    // doesn't require additional flags.
    if ((match(Arg0,
               PatternMatch::m_OneUse(PatternMatch::m_FSub(
                   PatternMatch::m_Value(A), PatternMatch::m_Value(B)))) &&
         match(Arg1, PatternMatch::m_PosZeroFP()) && isa<Instruction>(Arg0) &&
         cast<Instruction>(Arg0)->getFastMathFlags().noInfs())) {
      if (Arg0IsZero)
        std::swap(A, B);
      IC.replaceOperand(II, 0, A);
      IC.replaceOperand(II, 1, B);
      return &II;
    }
    break;
  }

  case Intrinsic::x86_avx512_add_ps_512:
  case Intrinsic::x86_avx512_div_ps_512:
  case Intrinsic::x86_avx512_mul_ps_512:
  case Intrinsic::x86_avx512_sub_ps_512:
  case Intrinsic::x86_avx512_add_pd_512:
  case Intrinsic::x86_avx512_div_pd_512:
  case Intrinsic::x86_avx512_mul_pd_512:
  case Intrinsic::x86_avx512_sub_pd_512:
    // If the rounding mode is CUR_DIRECTION(4) we can turn these into regular
    // IR operations.
    if (auto *R = dyn_cast<ConstantInt>(II.getArgOperand(2))) {
      if (R->getValue() == 4) {
        Value *Arg0 = II.getArgOperand(0);
        Value *Arg1 = II.getArgOperand(1);

        Value *V;
        switch (IID) {
        default:
          llvm_unreachable("Case stmts out of sync!");
        case Intrinsic::x86_avx512_add_ps_512:
        case Intrinsic::x86_avx512_add_pd_512:
          V = IC.Builder.CreateFAdd(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_sub_ps_512:
        case Intrinsic::x86_avx512_sub_pd_512:
          V = IC.Builder.CreateFSub(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_mul_ps_512:
        case Intrinsic::x86_avx512_mul_pd_512:
          V = IC.Builder.CreateFMul(Arg0, Arg1);
          break;
        case Intrinsic::x86_avx512_div_ps_512:
        case Intrinsic::x86_avx512_div_pd_512:
          V = IC.Builder.CreateFDiv(Arg0, Arg1);
          break;
        }

        return IC.replaceInstUsesWith(II, V);
      }
    }
    break;

  case Intrinsic::x86_avx512_mask_add_ss_round:
  case Intrinsic::x86_avx512_mask_div_ss_round:
  case Intrinsic::x86_avx512_mask_mul_ss_round:
  case Intrinsic::x86_avx512_mask_sub_ss_round:
  case Intrinsic::x86_avx512_mask_add_sd_round:
  case Intrinsic::x86_avx512_mask_div_sd_round:
  case Intrinsic::x86_avx512_mask_mul_sd_round:
  case Intrinsic::x86_avx512_mask_sub_sd_round:
    // If the rounding mode is CUR_DIRECTION(4) we can turn these into regular
    // IR operations.
    if (auto *R = dyn_cast<ConstantInt>(II.getArgOperand(4))) {
      if (R->getValue() == 4) {
        // Extract the element as scalars.
        Value *Arg0 = II.getArgOperand(0);
        Value *Arg1 = II.getArgOperand(1);
        Value *LHS = IC.Builder.CreateExtractElement(Arg0, (uint64_t)0);
        Value *RHS = IC.Builder.CreateExtractElement(Arg1, (uint64_t)0);

        Value *V;
        switch (IID) {
        default:
          llvm_unreachable("Case stmts out of sync!");
        case Intrinsic::x86_avx512_mask_add_ss_round:
        case Intrinsic::x86_avx512_mask_add_sd_round:
          V = IC.Builder.CreateFAdd(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_sub_ss_round:
        case Intrinsic::x86_avx512_mask_sub_sd_round:
          V = IC.Builder.CreateFSub(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_mul_ss_round:
        case Intrinsic::x86_avx512_mask_mul_sd_round:
          V = IC.Builder.CreateFMul(LHS, RHS);
          break;
        case Intrinsic::x86_avx512_mask_div_ss_round:
        case Intrinsic::x86_avx512_mask_div_sd_round:
          V = IC.Builder.CreateFDiv(LHS, RHS);
          break;
        }

        // Handle the masking aspect of the intrinsic.
        Value *Mask = II.getArgOperand(3);
        auto *C = dyn_cast<ConstantInt>(Mask);
        // We don't need a select if we know the mask bit is a 1.
        if (!C || !C->getValue()[0]) {
          // Cast the mask to an i1 vector and then extract the lowest element.
          auto *MaskTy = FixedVectorType::get(
              IC.Builder.getInt1Ty(),
              cast<IntegerType>(Mask->getType())->getBitWidth());
          Mask = IC.Builder.CreateBitCast(Mask, MaskTy);
          Mask = IC.Builder.CreateExtractElement(Mask, (uint64_t)0);
          // Extract the lowest element from the passthru operand.
          Value *Passthru =
              IC.Builder.CreateExtractElement(II.getArgOperand(2), (uint64_t)0);
          V = IC.Builder.CreateSelect(Mask, V, Passthru);
        }

        // Insert the result back into the original argument 0.
        V = IC.Builder.CreateInsertElement(Arg0, V, (uint64_t)0);

        return IC.replaceInstUsesWith(II, V);
      }
    }
    break;

  // Constant fold ashr( <A x Bi>, Ci ).
  // Constant fold lshr( <A x Bi>, Ci ).
  // Constant fold shl( <A x Bi>, Ci ).
  case Intrinsic::x86_sse2_psrai_d:
  case Intrinsic::x86_sse2_psrai_w:
  case Intrinsic::x86_avx2_psrai_d:
  case Intrinsic::x86_avx2_psrai_w:
  case Intrinsic::x86_avx512_psrai_q_128:
  case Intrinsic::x86_avx512_psrai_q_256:
  case Intrinsic::x86_avx512_psrai_d_512:
  case Intrinsic::x86_avx512_psrai_q_512:
  case Intrinsic::x86_avx512_psrai_w_512:
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w:
  case Intrinsic::x86_avx512_psrli_d_512:
  case Intrinsic::x86_avx512_psrli_q_512:
  case Intrinsic::x86_avx512_psrli_w_512:
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_avx512_pslli_d_512:
  case Intrinsic::x86_avx512_pslli_q_512:
  case Intrinsic::x86_avx512_pslli_w_512:
    if (Value *V = simplifyX86immShift(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_sse2_psra_d:
  case Intrinsic::x86_sse2_psra_w:
  case Intrinsic::x86_avx2_psra_d:
  case Intrinsic::x86_avx2_psra_w:
  case Intrinsic::x86_avx512_psra_q_128:
  case Intrinsic::x86_avx512_psra_q_256:
  case Intrinsic::x86_avx512_psra_d_512:
  case Intrinsic::x86_avx512_psra_q_512:
  case Intrinsic::x86_avx512_psra_w_512:
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx512_psrl_d_512:
  case Intrinsic::x86_avx512_psrl_q_512:
  case Intrinsic::x86_avx512_psrl_w_512:
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx512_psll_d_512:
  case Intrinsic::x86_avx512_psll_q_512:
  case Intrinsic::x86_avx512_psll_w_512: {
    if (Value *V = simplifyX86immShift(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }

    // SSE2/AVX2 uses only the first 64-bits of the 128-bit vector
    // operand to compute the shift amount.
    Value *Arg1 = II.getArgOperand(1);
    assert(Arg1->getType()->getPrimitiveSizeInBits() == 128 &&
           "Unexpected packed shift size");
    unsigned VWidth = cast<VectorType>(Arg1->getType())->getNumElements();

    if (Value *V = SimplifyDemandedVectorEltsLow(Arg1, VWidth, VWidth / 2)) {
      return IC.replaceOperand(II, 1, V);
    }
    break;
  }

  case Intrinsic::x86_avx2_psllv_d:
  case Intrinsic::x86_avx2_psllv_d_256:
  case Intrinsic::x86_avx2_psllv_q:
  case Intrinsic::x86_avx2_psllv_q_256:
  case Intrinsic::x86_avx512_psllv_d_512:
  case Intrinsic::x86_avx512_psllv_q_512:
  case Intrinsic::x86_avx512_psllv_w_128:
  case Intrinsic::x86_avx512_psllv_w_256:
  case Intrinsic::x86_avx512_psllv_w_512:
  case Intrinsic::x86_avx2_psrav_d:
  case Intrinsic::x86_avx2_psrav_d_256:
  case Intrinsic::x86_avx512_psrav_q_128:
  case Intrinsic::x86_avx512_psrav_q_256:
  case Intrinsic::x86_avx512_psrav_d_512:
  case Intrinsic::x86_avx512_psrav_q_512:
  case Intrinsic::x86_avx512_psrav_w_128:
  case Intrinsic::x86_avx512_psrav_w_256:
  case Intrinsic::x86_avx512_psrav_w_512:
  case Intrinsic::x86_avx2_psrlv_d:
  case Intrinsic::x86_avx2_psrlv_d_256:
  case Intrinsic::x86_avx2_psrlv_q:
  case Intrinsic::x86_avx2_psrlv_q_256:
  case Intrinsic::x86_avx512_psrlv_d_512:
  case Intrinsic::x86_avx512_psrlv_q_512:
  case Intrinsic::x86_avx512_psrlv_w_128:
  case Intrinsic::x86_avx512_psrlv_w_256:
  case Intrinsic::x86_avx512_psrlv_w_512:
    if (Value *V = simplifyX86varShift(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_sse2_packssdw_128:
  case Intrinsic::x86_sse2_packsswb_128:
  case Intrinsic::x86_avx2_packssdw:
  case Intrinsic::x86_avx2_packsswb:
  case Intrinsic::x86_avx512_packssdw_512:
  case Intrinsic::x86_avx512_packsswb_512:
    if (Value *V = simplifyX86pack(II, IC.Builder, true)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_sse2_packuswb_128:
  case Intrinsic::x86_sse41_packusdw:
  case Intrinsic::x86_avx2_packusdw:
  case Intrinsic::x86_avx2_packuswb:
  case Intrinsic::x86_avx512_packusdw_512:
  case Intrinsic::x86_avx512_packuswb_512:
    if (Value *V = simplifyX86pack(II, IC.Builder, false)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_pclmulqdq:
  case Intrinsic::x86_pclmulqdq_256:
  case Intrinsic::x86_pclmulqdq_512: {
    if (auto *C = dyn_cast<ConstantInt>(II.getArgOperand(2))) {
      unsigned Imm = C->getZExtValue();

      bool MadeChange = false;
      Value *Arg0 = II.getArgOperand(0);
      Value *Arg1 = II.getArgOperand(1);
      unsigned VWidth = cast<VectorType>(Arg0->getType())->getNumElements();

      APInt UndefElts1(VWidth, 0);
      APInt DemandedElts1 =
          APInt::getSplat(VWidth, APInt(2, (Imm & 0x01) ? 2 : 1));
      if (Value *V =
              IC.SimplifyDemandedVectorElts(Arg0, DemandedElts1, UndefElts1)) {
        IC.replaceOperand(II, 0, V);
        MadeChange = true;
      }

      APInt UndefElts2(VWidth, 0);
      APInt DemandedElts2 =
          APInt::getSplat(VWidth, APInt(2, (Imm & 0x10) ? 2 : 1));
      if (Value *V =
              IC.SimplifyDemandedVectorElts(Arg1, DemandedElts2, UndefElts2)) {
        IC.replaceOperand(II, 1, V);
        MadeChange = true;
      }

      // If either input elements are undef, the result is zero.
      if (DemandedElts1.isSubsetOf(UndefElts1) ||
          DemandedElts2.isSubsetOf(UndefElts2)) {
        return IC.replaceInstUsesWith(II,
                                      ConstantAggregateZero::get(II.getType()));
      }

      if (MadeChange) {
        return &II;
      }
    }
    break;
  }

  case Intrinsic::x86_sse41_insertps:
    if (Value *V = simplifyX86insertps(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_sse4a_extrq: {
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);
    unsigned VWidth0 = cast<VectorType>(Op0->getType())->getNumElements();
    unsigned VWidth1 = cast<VectorType>(Op1->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth0 == 2 &&
           VWidth1 == 16 && "Unexpected operand sizes");

    // See if we're dealing with constant values.
    Constant *C1 = dyn_cast<Constant>(Op1);
    ConstantInt *CILength =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)0))
           : nullptr;
    ConstantInt *CIIndex =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)1))
           : nullptr;

    // Attempt to simplify to a constant, shuffle vector or EXTRQI call.
    if (Value *V = simplifyX86extrq(II, Op0, CILength, CIIndex, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }

    // EXTRQ only uses the lowest 64-bits of the first 128-bit vector
    // operands and the lowest 16-bits of the second.
    bool MadeChange = false;
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth0, 1)) {
      IC.replaceOperand(II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Op1, VWidth1, 2)) {
      IC.replaceOperand(II, 1, V);
      MadeChange = true;
    }
    if (MadeChange) {
      return &II;
    }
    break;
  }

  case Intrinsic::x86_sse4a_extrqi: {
    // EXTRQI: Extract Length bits starting from Index. Zero pad the remaining
    // bits of the lower 64-bits. The upper 64-bits are undefined.
    Value *Op0 = II.getArgOperand(0);
    unsigned VWidth = cast<VectorType>(Op0->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 && VWidth == 2 &&
           "Unexpected operand size");

    // See if we're dealing with constant values.
    ConstantInt *CILength = dyn_cast<ConstantInt>(II.getArgOperand(1));
    ConstantInt *CIIndex = dyn_cast<ConstantInt>(II.getArgOperand(2));

    // Attempt to simplify to a constant or shuffle vector.
    if (Value *V = simplifyX86extrq(II, Op0, CILength, CIIndex, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }

    // EXTRQI only uses the lowest 64-bits of the first 128-bit vector
    // operand.
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth, 1)) {
      return IC.replaceOperand(II, 0, V);
    }
    break;
  }

  case Intrinsic::x86_sse4a_insertq: {
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);
    unsigned VWidth = cast<VectorType>(Op0->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth == 2 &&
           cast<VectorType>(Op1->getType())->getNumElements() == 2 &&
           "Unexpected operand size");

    // See if we're dealing with constant values.
    Constant *C1 = dyn_cast<Constant>(Op1);
    ConstantInt *CI11 =
        C1 ? dyn_cast_or_null<ConstantInt>(C1->getAggregateElement((unsigned)1))
           : nullptr;

    // Attempt to simplify to a constant, shuffle vector or INSERTQI call.
    if (CI11) {
      const APInt &V11 = CI11->getValue();
      APInt Len = V11.zextOrTrunc(6);
      APInt Idx = V11.lshr(8).zextOrTrunc(6);
      if (Value *V = simplifyX86insertq(II, Op0, Op1, Len, Idx, IC.Builder)) {
        return IC.replaceInstUsesWith(II, V);
      }
    }

    // INSERTQ only uses the lowest 64-bits of the first 128-bit vector
    // operand.
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth, 1)) {
      return IC.replaceOperand(II, 0, V);
    }
    break;
  }

  case Intrinsic::x86_sse4a_insertqi: {
    // INSERTQI: Extract lowest Length bits from lower half of second source and
    // insert over first source starting at Index bit. The upper 64-bits are
    // undefined.
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);
    unsigned VWidth0 = cast<VectorType>(Op0->getType())->getNumElements();
    unsigned VWidth1 = cast<VectorType>(Op1->getType())->getNumElements();
    assert(Op0->getType()->getPrimitiveSizeInBits() == 128 &&
           Op1->getType()->getPrimitiveSizeInBits() == 128 && VWidth0 == 2 &&
           VWidth1 == 2 && "Unexpected operand sizes");

    // See if we're dealing with constant values.
    ConstantInt *CILength = dyn_cast<ConstantInt>(II.getArgOperand(2));
    ConstantInt *CIIndex = dyn_cast<ConstantInt>(II.getArgOperand(3));

    // Attempt to simplify to a constant or shuffle vector.
    if (CILength && CIIndex) {
      APInt Len = CILength->getValue().zextOrTrunc(6);
      APInt Idx = CIIndex->getValue().zextOrTrunc(6);
      if (Value *V = simplifyX86insertq(II, Op0, Op1, Len, Idx, IC.Builder)) {
        return IC.replaceInstUsesWith(II, V);
      }
    }

    // INSERTQI only uses the lowest 64-bits of the first two 128-bit vector
    // operands.
    bool MadeChange = false;
    if (Value *V = SimplifyDemandedVectorEltsLow(Op0, VWidth0, 1)) {
      IC.replaceOperand(II, 0, V);
      MadeChange = true;
    }
    if (Value *V = SimplifyDemandedVectorEltsLow(Op1, VWidth1, 1)) {
      IC.replaceOperand(II, 1, V);
      MadeChange = true;
    }
    if (MadeChange) {
      return &II;
    }
    break;
  }

  case Intrinsic::x86_sse41_pblendvb:
  case Intrinsic::x86_sse41_blendvps:
  case Intrinsic::x86_sse41_blendvpd:
  case Intrinsic::x86_avx_blendv_ps_256:
  case Intrinsic::x86_avx_blendv_pd_256:
  case Intrinsic::x86_avx2_pblendvb: {
    // fold (blend A, A, Mask) -> A
    Value *Op0 = II.getArgOperand(0);
    Value *Op1 = II.getArgOperand(1);
    Value *Mask = II.getArgOperand(2);
    if (Op0 == Op1) {
      return IC.replaceInstUsesWith(II, Op0);
    }

    // Zero Mask - select 1st argument.
    if (isa<ConstantAggregateZero>(Mask)) {
      return IC.replaceInstUsesWith(II, Op0);
    }

    // Constant Mask - select 1st/2nd argument lane based on top bit of mask.
    if (auto *ConstantMask = dyn_cast<ConstantDataVector>(Mask)) {
      Constant *NewSelector = getNegativeIsTrueBoolVec(ConstantMask);
      return SelectInst::Create(NewSelector, Op1, Op0, "blendv");
    }

    // Convert to a vector select if we can bypass casts and find a boolean
    // vector condition value.
    Value *BoolVec;
    Mask = InstCombiner::peekThroughBitcast(Mask);
    if (match(Mask, PatternMatch::m_SExt(PatternMatch::m_Value(BoolVec))) &&
        BoolVec->getType()->isVectorTy() &&
        BoolVec->getType()->getScalarSizeInBits() == 1) {
      assert(Mask->getType()->getPrimitiveSizeInBits() ==
                 II.getType()->getPrimitiveSizeInBits() &&
             "Not expecting mask and operands with different sizes");

      unsigned NumMaskElts =
          cast<VectorType>(Mask->getType())->getNumElements();
      unsigned NumOperandElts =
          cast<VectorType>(II.getType())->getNumElements();
      if (NumMaskElts == NumOperandElts) {
        return SelectInst::Create(BoolVec, Op1, Op0);
      }

      // If the mask has less elements than the operands, each mask bit maps to
      // multiple elements of the operands. Bitcast back and forth.
      if (NumMaskElts < NumOperandElts) {
        Value *CastOp0 = IC.Builder.CreateBitCast(Op0, Mask->getType());
        Value *CastOp1 = IC.Builder.CreateBitCast(Op1, Mask->getType());
        Value *Sel = IC.Builder.CreateSelect(BoolVec, CastOp1, CastOp0);
        return new BitCastInst(Sel, II.getType());
      }
    }

    break;
  }

  case Intrinsic::x86_ssse3_pshuf_b_128:
  case Intrinsic::x86_avx2_pshuf_b:
  case Intrinsic::x86_avx512_pshuf_b_512:
    if (Value *V = simplifyX86pshufb(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_avx_vpermilvar_ps:
  case Intrinsic::x86_avx_vpermilvar_ps_256:
  case Intrinsic::x86_avx512_vpermilvar_ps_512:
  case Intrinsic::x86_avx_vpermilvar_pd:
  case Intrinsic::x86_avx_vpermilvar_pd_256:
  case Intrinsic::x86_avx512_vpermilvar_pd_512:
    if (Value *V = simplifyX86vpermilvar(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_avx2_permd:
  case Intrinsic::x86_avx2_permps:
  case Intrinsic::x86_avx512_permvar_df_256:
  case Intrinsic::x86_avx512_permvar_df_512:
  case Intrinsic::x86_avx512_permvar_di_256:
  case Intrinsic::x86_avx512_permvar_di_512:
  case Intrinsic::x86_avx512_permvar_hi_128:
  case Intrinsic::x86_avx512_permvar_hi_256:
  case Intrinsic::x86_avx512_permvar_hi_512:
  case Intrinsic::x86_avx512_permvar_qi_128:
  case Intrinsic::x86_avx512_permvar_qi_256:
  case Intrinsic::x86_avx512_permvar_qi_512:
  case Intrinsic::x86_avx512_permvar_sf_512:
  case Intrinsic::x86_avx512_permvar_si_512:
    if (Value *V = simplifyX86vpermv(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  case Intrinsic::x86_avx_maskload_ps:
  case Intrinsic::x86_avx_maskload_pd:
  case Intrinsic::x86_avx_maskload_ps_256:
  case Intrinsic::x86_avx_maskload_pd_256:
  case Intrinsic::x86_avx2_maskload_d:
  case Intrinsic::x86_avx2_maskload_q:
  case Intrinsic::x86_avx2_maskload_d_256:
  case Intrinsic::x86_avx2_maskload_q_256:
    if (Instruction *I = simplifyX86MaskedLoad(II, IC)) {
      return I;
    }
    break;

  case Intrinsic::x86_sse2_maskmov_dqu:
  case Intrinsic::x86_avx_maskstore_ps:
  case Intrinsic::x86_avx_maskstore_pd:
  case Intrinsic::x86_avx_maskstore_ps_256:
  case Intrinsic::x86_avx_maskstore_pd_256:
  case Intrinsic::x86_avx2_maskstore_d:
  case Intrinsic::x86_avx2_maskstore_q:
  case Intrinsic::x86_avx2_maskstore_d_256:
  case Intrinsic::x86_avx2_maskstore_q_256:
    if (simplifyX86MaskedStore(II, IC)) {
      return nullptr;
    }
    break;

  case Intrinsic::x86_addcarry_32:
  case Intrinsic::x86_addcarry_64:
    if (Value *V = simplifyX86addcarry(II, IC.Builder)) {
      return IC.replaceInstUsesWith(II, V);
    }
    break;

  default:
    break;
  }
  return None;
}

Optional<Value *> X86TTIImpl::simplifyDemandedUseBitsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedMask, KnownBits &Known,
    bool &KnownBitsComputed) const {
  switch (II.getIntrinsicID()) {
  default:
    break;
  case Intrinsic::x86_mmx_pmovmskb:
  case Intrinsic::x86_sse_movmsk_ps:
  case Intrinsic::x86_sse2_movmsk_pd:
  case Intrinsic::x86_sse2_pmovmskb_128:
  case Intrinsic::x86_avx_movmsk_ps_256:
  case Intrinsic::x86_avx_movmsk_pd_256:
  case Intrinsic::x86_avx2_pmovmskb: {
    // MOVMSK copies the vector elements' sign bits to the low bits
    // and zeros the high bits.
    unsigned ArgWidth;
    if (II.getIntrinsicID() == Intrinsic::x86_mmx_pmovmskb) {
      ArgWidth = 8; // Arg is x86_mmx, but treated as <8 x i8>.
    } else {
      auto Arg = II.getArgOperand(0);
      auto ArgType = cast<VectorType>(Arg->getType());
      ArgWidth = ArgType->getNumElements();
    }

    // If we don't need any of low bits then return zero,
    // we know that DemandedMask is non-zero already.
    APInt DemandedElts = DemandedMask.zextOrTrunc(ArgWidth);
    Type *VTy = II.getType();
    if (DemandedElts.isNullValue()) {
      return ConstantInt::getNullValue(VTy);
    }

    // We know that the upper bits are set to zero.
    Known.Zero.setBitsFrom(ArgWidth);
    KnownBitsComputed = true;
    break;
  }
  case Intrinsic::x86_sse42_crc32_64_64:
    Known.Zero.setBitsFrom(32);
    KnownBitsComputed = true;
    break;
  }
  return None;
}

Optional<Value *> X86TTIImpl::simplifyDemandedVectorEltsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
    APInt &UndefElts2, APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        simplifyAndSetOp) const {
  unsigned VWidth = cast<FixedVectorType>(II.getType())->getNumElements();
  switch (II.getIntrinsicID()) {
  default:
    break;
  case Intrinsic::x86_xop_vfrcz_ss:
  case Intrinsic::x86_xop_vfrcz_sd:
    // The instructions for these intrinsics are speced to zero upper bits not
    // pass them through like other scalar intrinsics. So we shouldn't just
    // use Arg0 if DemandedElts[0] is clear like we do for other intrinsics.
    // Instead we should return a zero vector.
    if (!DemandedElts[0]) {
      IC.addToWorklist(&II);
      return ConstantAggregateZero::get(II.getType());
    }

    // Only the lower element is used.
    DemandedElts = 1;
    simplifyAndSetOp(&II, 0, DemandedElts, UndefElts);

    // Only the lower element is undefined. The high elements are zero.
    UndefElts = UndefElts[0];
    break;

  // Unary scalar-as-vector operations that work column-wise.
  case Intrinsic::x86_sse_rcp_ss:
  case Intrinsic::x86_sse_rsqrt_ss:
    simplifyAndSetOp(&II, 0, DemandedElts, UndefElts);

    // If lowest element of a scalar op isn't used then use Arg0.
    if (!DemandedElts[0]) {
      IC.addToWorklist(&II);
      return II.getArgOperand(0);
    }
    // TODO: If only low elt lower SQRT to FSQRT (with rounding/exceptions
    // checks).
    break;

  // Binary scalar-as-vector operations that work column-wise. The high
  // elements come from operand 0. The low element is a function of both
  // operands.
  case Intrinsic::x86_sse_min_ss:
  case Intrinsic::x86_sse_max_ss:
  case Intrinsic::x86_sse_cmp_ss:
  case Intrinsic::x86_sse2_min_sd:
  case Intrinsic::x86_sse2_max_sd:
  case Intrinsic::x86_sse2_cmp_sd: {
    simplifyAndSetOp(&II, 0, DemandedElts, UndefElts);

    // If lowest element of a scalar op isn't used then use Arg0.
    if (!DemandedElts[0]) {
      IC.addToWorklist(&II);
      return II.getArgOperand(0);
    }

    // Only lower element is used for operand 1.
    DemandedElts = 1;
    simplifyAndSetOp(&II, 1, DemandedElts, UndefElts2);

    // Lower element is undefined if both lower elements are undefined.
    // Consider things like undef&0.  The result is known zero, not undef.
    if (!UndefElts2[0])
      UndefElts.clearBit(0);

    break;
  }

  // Binary scalar-as-vector operations that work column-wise. The high
  // elements come from operand 0 and the low element comes from operand 1.
  case Intrinsic::x86_sse41_round_ss:
  case Intrinsic::x86_sse41_round_sd: {
    // Don't use the low element of operand 0.
    APInt DemandedElts2 = DemandedElts;
    DemandedElts2.clearBit(0);
    simplifyAndSetOp(&II, 0, DemandedElts2, UndefElts);

    // If lowest element of a scalar op isn't used then use Arg0.
    if (!DemandedElts[0]) {
      IC.addToWorklist(&II);
      return II.getArgOperand(0);
    }

    // Only lower element is used for operand 1.
    DemandedElts = 1;
    simplifyAndSetOp(&II, 1, DemandedElts, UndefElts2);

    // Take the high undef elements from operand 0 and take the lower element
    // from operand 1.
    UndefElts.clearBit(0);
    UndefElts |= UndefElts2[0];
    break;
  }

  // Three input scalar-as-vector operations that work column-wise. The high
  // elements come from operand 0 and the low element is a function of all
  // three inputs.
  case Intrinsic::x86_avx512_mask_add_ss_round:
  case Intrinsic::x86_avx512_mask_div_ss_round:
  case Intrinsic::x86_avx512_mask_mul_ss_round:
  case Intrinsic::x86_avx512_mask_sub_ss_round:
  case Intrinsic::x86_avx512_mask_max_ss_round:
  case Intrinsic::x86_avx512_mask_min_ss_round:
  case Intrinsic::x86_avx512_mask_add_sd_round:
  case Intrinsic::x86_avx512_mask_div_sd_round:
  case Intrinsic::x86_avx512_mask_mul_sd_round:
  case Intrinsic::x86_avx512_mask_sub_sd_round:
  case Intrinsic::x86_avx512_mask_max_sd_round:
  case Intrinsic::x86_avx512_mask_min_sd_round:
    simplifyAndSetOp(&II, 0, DemandedElts, UndefElts);

    // If lowest element of a scalar op isn't used then use Arg0.
    if (!DemandedElts[0]) {
      IC.addToWorklist(&II);
      return II.getArgOperand(0);
    }

    // Only lower element is used for operand 1 and 2.
    DemandedElts = 1;
    simplifyAndSetOp(&II, 1, DemandedElts, UndefElts2);
    simplifyAndSetOp(&II, 2, DemandedElts, UndefElts3);

    // Lower element is undefined if all three lower elements are undefined.
    // Consider things like undef&0.  The result is known zero, not undef.
    if (!UndefElts2[0] || !UndefElts3[0])
      UndefElts.clearBit(0);

    break;

  case Intrinsic::x86_sse2_packssdw_128:
  case Intrinsic::x86_sse2_packsswb_128:
  case Intrinsic::x86_sse2_packuswb_128:
  case Intrinsic::x86_sse41_packusdw:
  case Intrinsic::x86_avx2_packssdw:
  case Intrinsic::x86_avx2_packsswb:
  case Intrinsic::x86_avx2_packusdw:
  case Intrinsic::x86_avx2_packuswb:
  case Intrinsic::x86_avx512_packssdw_512:
  case Intrinsic::x86_avx512_packsswb_512:
  case Intrinsic::x86_avx512_packusdw_512:
  case Intrinsic::x86_avx512_packuswb_512: {
    auto *Ty0 = II.getArgOperand(0)->getType();
    unsigned InnerVWidth = cast<VectorType>(Ty0)->getNumElements();
    assert(VWidth == (InnerVWidth * 2) && "Unexpected input size");

    unsigned NumLanes = Ty0->getPrimitiveSizeInBits() / 128;
    unsigned VWidthPerLane = VWidth / NumLanes;
    unsigned InnerVWidthPerLane = InnerVWidth / NumLanes;

    // Per lane, pack the elements of the first input and then the second.
    // e.g.
    // v8i16 PACK(v4i32 X, v4i32 Y) - (X[0..3],Y[0..3])
    // v32i8 PACK(v16i16 X, v16i16 Y) - (X[0..7],Y[0..7]),(X[8..15],Y[8..15])
    for (int OpNum = 0; OpNum != 2; ++OpNum) {
      APInt OpDemandedElts(InnerVWidth, 0);
      for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
        unsigned LaneIdx = Lane * VWidthPerLane;
        for (unsigned Elt = 0; Elt != InnerVWidthPerLane; ++Elt) {
          unsigned Idx = LaneIdx + Elt + InnerVWidthPerLane * OpNum;
          if (DemandedElts[Idx])
            OpDemandedElts.setBit((Lane * InnerVWidthPerLane) + Elt);
        }
      }

      // Demand elements from the operand.
      APInt OpUndefElts(InnerVWidth, 0);
      simplifyAndSetOp(&II, OpNum, OpDemandedElts, OpUndefElts);

      // Pack the operand's UNDEF elements, one lane at a time.
      OpUndefElts = OpUndefElts.zext(VWidth);
      for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
        APInt LaneElts = OpUndefElts.lshr(InnerVWidthPerLane * Lane);
        LaneElts = LaneElts.getLoBits(InnerVWidthPerLane);
        LaneElts <<= InnerVWidthPerLane * (2 * Lane + OpNum);
        UndefElts |= LaneElts;
      }
    }
    break;
  }

  // PSHUFB
  case Intrinsic::x86_ssse3_pshuf_b_128:
  case Intrinsic::x86_avx2_pshuf_b:
  case Intrinsic::x86_avx512_pshuf_b_512:
  // PERMILVAR
  case Intrinsic::x86_avx_vpermilvar_ps:
  case Intrinsic::x86_avx_vpermilvar_ps_256:
  case Intrinsic::x86_avx512_vpermilvar_ps_512:
  case Intrinsic::x86_avx_vpermilvar_pd:
  case Intrinsic::x86_avx_vpermilvar_pd_256:
  case Intrinsic::x86_avx512_vpermilvar_pd_512:
  // PERMV
  case Intrinsic::x86_avx2_permd:
  case Intrinsic::x86_avx2_permps: {
    simplifyAndSetOp(&II, 1, DemandedElts, UndefElts);
    break;
  }

  // SSE4A instructions leave the upper 64-bits of the 128-bit result
  // in an undefined state.
  case Intrinsic::x86_sse4a_extrq:
  case Intrinsic::x86_sse4a_extrqi:
  case Intrinsic::x86_sse4a_insertq:
  case Intrinsic::x86_sse4a_insertqi:
    UndefElts.setHighBits(VWidth / 2);
    break;
  }
  return None;
}
