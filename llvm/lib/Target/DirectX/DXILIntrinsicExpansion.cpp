//===- DXILIntrinsicExpansion.cpp - Prepare LLVM Module for DXIL encoding--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains DXIL intrinsic expansions for those that don't have
//  opcodes in DirectX Intermediate Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DXILIntrinsicExpansion.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "dxil-intrinsic-expansion"

using namespace llvm;

class DXILIntrinsicExpansionLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILIntrinsicExpansionLegacy() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static char ID; // Pass identification.
};

static bool isIntrinsicExpansion(Function &F) {
  switch (F.getIntrinsicID()) {
  case Intrinsic::abs:
  case Intrinsic::atan2:
  case Intrinsic::exp:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::pow:
  case Intrinsic::dx_all:
  case Intrinsic::dx_any:
  case Intrinsic::dx_cross:
  case Intrinsic::dx_uclamp:
  case Intrinsic::dx_sclamp:
  case Intrinsic::dx_nclamp:
  case Intrinsic::dx_degrees:
  case Intrinsic::dx_lerp:
  case Intrinsic::dx_length:
  case Intrinsic::dx_normalize:
  case Intrinsic::dx_fdot:
  case Intrinsic::dx_sdot:
  case Intrinsic::dx_udot:
  case Intrinsic::dx_sign:
  case Intrinsic::dx_step:
  case Intrinsic::dx_radians:
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_fadd:
    return true;
  }
  return false;
}
static Value *expandVecReduceAdd(CallInst *Orig, Intrinsic::ID IntrinsicId) {
  assert(IntrinsicId == Intrinsic::vector_reduce_add ||
         IntrinsicId == Intrinsic::vector_reduce_fadd);

  IRBuilder<> Builder(Orig);
  bool IsFAdd = (IntrinsicId == Intrinsic::vector_reduce_fadd);

  Value *X = Orig->getOperand(IsFAdd ? 1 : 0);
  Type *Ty = X->getType();
  auto *XVec = dyn_cast<FixedVectorType>(Ty);
  unsigned XVecSize = XVec->getNumElements();
  Value *Sum = Builder.CreateExtractElement(X, static_cast<uint64_t>(0));

  // Handle the initial start value for floating-point addition.
  if (IsFAdd) {
    Constant *StartValue = dyn_cast<Constant>(Orig->getOperand(0));
    if (StartValue && !StartValue->isZeroValue())
      Sum = Builder.CreateFAdd(Sum, StartValue);
  }

  // Accumulate the remaining vector elements.
  for (unsigned I = 1; I < XVecSize; I++) {
    Value *Elt = Builder.CreateExtractElement(X, I);
    if (IsFAdd)
      Sum = Builder.CreateFAdd(Sum, Elt);
    else
      Sum = Builder.CreateAdd(Sum, Elt);
  }

  return Sum;
}

static Value *expandAbs(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();
  Constant *Zero = Ty->isVectorTy()
                       ? ConstantVector::getSplat(
                             ElementCount::getFixed(
                                 cast<FixedVectorType>(Ty)->getNumElements()),
                             ConstantInt::get(EltTy, 0))
                       : ConstantInt::get(EltTy, 0);
  auto *V = Builder.CreateSub(Zero, X);
  return Builder.CreateIntrinsic(Ty, Intrinsic::smax, {X, V}, nullptr,
                                 "dx.max");
}

static Value *expandCrossIntrinsic(CallInst *Orig) {

  VectorType *VT = cast<VectorType>(Orig->getType());
  if (cast<FixedVectorType>(VT)->getNumElements() != 3)
    report_fatal_error(Twine("return vector must have exactly 3 elements"),
                       /* gen_crash_diag=*/false);

  Value *op0 = Orig->getOperand(0);
  Value *op1 = Orig->getOperand(1);
  IRBuilder<> Builder(Orig);

  Value *op0_x = Builder.CreateExtractElement(op0, (uint64_t)0, "x0");
  Value *op0_y = Builder.CreateExtractElement(op0, 1, "x1");
  Value *op0_z = Builder.CreateExtractElement(op0, 2, "x2");

  Value *op1_x = Builder.CreateExtractElement(op1, (uint64_t)0, "y0");
  Value *op1_y = Builder.CreateExtractElement(op1, 1, "y1");
  Value *op1_z = Builder.CreateExtractElement(op1, 2, "y2");

  auto MulSub = [&](Value *x0, Value *y0, Value *x1, Value *y1) -> Value * {
    Value *xy = Builder.CreateFMul(x0, y1);
    Value *yx = Builder.CreateFMul(y0, x1);
    return Builder.CreateFSub(xy, yx, Orig->getName());
  };

  Value *yz_zy = MulSub(op0_y, op0_z, op1_y, op1_z);
  Value *zx_xz = MulSub(op0_z, op0_x, op1_z, op1_x);
  Value *xy_yx = MulSub(op0_x, op0_y, op1_x, op1_y);

  Value *cross = UndefValue::get(VT);
  cross = Builder.CreateInsertElement(cross, yz_zy, (uint64_t)0);
  cross = Builder.CreateInsertElement(cross, zx_xz, 1);
  cross = Builder.CreateInsertElement(cross, xy_yx, 2);
  return cross;
}

// Create appropriate DXIL float dot intrinsic for the given A and B operands
// The appropriate opcode will be determined by the size of the operands
// The dot product is placed in the position indicated by Orig
static Value *expandFloatDotIntrinsic(CallInst *Orig, Value *A, Value *B) {
  Type *ATy = A->getType();
  [[maybe_unused]] Type *BTy = B->getType();
  assert(ATy->isVectorTy() && BTy->isVectorTy());

  IRBuilder<> Builder(Orig);

  auto *AVec = dyn_cast<FixedVectorType>(ATy);

  assert(ATy->getScalarType()->isFloatingPointTy());

  Intrinsic::ID DotIntrinsic = Intrinsic::dx_dot4;
  switch (AVec->getNumElements()) {
  case 2:
    DotIntrinsic = Intrinsic::dx_dot2;
    break;
  case 3:
    DotIntrinsic = Intrinsic::dx_dot3;
    break;
  case 4:
    DotIntrinsic = Intrinsic::dx_dot4;
    break;
  default:
    report_fatal_error(
        Twine("Invalid dot product input vector: length is outside 2-4"),
        /* gen_crash_diag=*/false);
    return nullptr;
  }
  return Builder.CreateIntrinsic(ATy->getScalarType(), DotIntrinsic,
                                 ArrayRef<Value *>{A, B}, nullptr, "dot");
}

// Create the appropriate DXIL float dot intrinsic for the operands of Orig
// The appropriate opcode will be determined by the size of the operands
// The dot product is placed in the position indicated by Orig
static Value *expandFloatDotIntrinsic(CallInst *Orig) {
  return expandFloatDotIntrinsic(Orig, Orig->getOperand(0),
                                 Orig->getOperand(1));
}

// Expand integer dot product to multiply and add ops
static Value *expandIntegerDotIntrinsic(CallInst *Orig,
                                        Intrinsic::ID DotIntrinsic) {
  assert(DotIntrinsic == Intrinsic::dx_sdot ||
         DotIntrinsic == Intrinsic::dx_udot);
  Value *A = Orig->getOperand(0);
  Value *B = Orig->getOperand(1);
  Type *ATy = A->getType();
  [[maybe_unused]] Type *BTy = B->getType();
  assert(ATy->isVectorTy() && BTy->isVectorTy());

  IRBuilder<> Builder(Orig);

  auto *AVec = dyn_cast<FixedVectorType>(ATy);

  assert(ATy->getScalarType()->isIntegerTy());

  Value *Result;
  Intrinsic::ID MadIntrinsic = DotIntrinsic == Intrinsic::dx_sdot
                                   ? Intrinsic::dx_imad
                                   : Intrinsic::dx_umad;
  Value *Elt0 = Builder.CreateExtractElement(A, (uint64_t)0);
  Value *Elt1 = Builder.CreateExtractElement(B, (uint64_t)0);
  Result = Builder.CreateMul(Elt0, Elt1);
  for (unsigned I = 1; I < AVec->getNumElements(); I++) {
    Elt0 = Builder.CreateExtractElement(A, I);
    Elt1 = Builder.CreateExtractElement(B, I);
    Result = Builder.CreateIntrinsic(Result->getType(), MadIntrinsic,
                                     ArrayRef<Value *>{Elt0, Elt1, Result},
                                     nullptr, "dx.mad");
  }
  return Result;
}

static Value *expandExpIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();
  Constant *Log2eConst =
      Ty->isVectorTy() ? ConstantVector::getSplat(
                             ElementCount::getFixed(
                                 cast<FixedVectorType>(Ty)->getNumElements()),
                             ConstantFP::get(EltTy, numbers::log2ef))
                       : ConstantFP::get(EltTy, numbers::log2ef);
  Value *NewX = Builder.CreateFMul(Log2eConst, X);
  auto *Exp2Call =
      Builder.CreateIntrinsic(Ty, Intrinsic::exp2, {NewX}, nullptr, "dx.exp2");
  Exp2Call->setTailCall(Orig->isTailCall());
  Exp2Call->setAttributes(Orig->getAttributes());
  return Exp2Call;
}

static Value *expandAnyOrAllIntrinsic(CallInst *Orig,
                                      Intrinsic::ID intrinsicId) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();

  auto ApplyOp = [&Builder](Intrinsic::ID IntrinsicId, Value *Result,
                            Value *Elt) {
    if (IntrinsicId == Intrinsic::dx_any)
      return Builder.CreateOr(Result, Elt);
    assert(IntrinsicId == Intrinsic::dx_all);
    return Builder.CreateAnd(Result, Elt);
  };

  Value *Result = nullptr;
  if (!Ty->isVectorTy()) {
    Result = EltTy->isFloatingPointTy()
                 ? Builder.CreateFCmpUNE(X, ConstantFP::get(EltTy, 0))
                 : Builder.CreateICmpNE(X, ConstantInt::get(EltTy, 0));
  } else {
    auto *XVec = dyn_cast<FixedVectorType>(Ty);
    Value *Cond =
        EltTy->isFloatingPointTy()
            ? Builder.CreateFCmpUNE(
                  X, ConstantVector::getSplat(
                         ElementCount::getFixed(XVec->getNumElements()),
                         ConstantFP::get(EltTy, 0)))
            : Builder.CreateICmpNE(
                  X, ConstantVector::getSplat(
                         ElementCount::getFixed(XVec->getNumElements()),
                         ConstantInt::get(EltTy, 0)));
    Result = Builder.CreateExtractElement(Cond, (uint64_t)0);
    for (unsigned I = 1; I < XVec->getNumElements(); I++) {
      Value *Elt = Builder.CreateExtractElement(Cond, I);
      Result = ApplyOp(intrinsicId, Result, Elt);
    }
  }
  return Result;
}

static Value *expandLengthIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();

  // Though dx.length does work on scalar type, we can optimize it to just emit
  // fabs, in CGBuiltin.cpp. We shouldn't see a scalar type here because
  // CGBuiltin.cpp should have emitted a fabs call.
  Value *Elt = Builder.CreateExtractElement(X, (uint64_t)0);
  auto *XVec = dyn_cast<FixedVectorType>(Ty);
  unsigned XVecSize = XVec->getNumElements();
  if (!(Ty->isVectorTy() && XVecSize > 1))
    report_fatal_error(Twine("Invalid input type for length intrinsic"),
                       /* gen_crash_diag=*/false);

  Value *Sum = Builder.CreateFMul(Elt, Elt);
  for (unsigned I = 1; I < XVecSize; I++) {
    Elt = Builder.CreateExtractElement(X, I);
    Value *Mul = Builder.CreateFMul(Elt, Elt);
    Sum = Builder.CreateFAdd(Sum, Mul);
  }
  return Builder.CreateIntrinsic(EltTy, Intrinsic::sqrt, ArrayRef<Value *>{Sum},
                                 nullptr, "elt.sqrt");
}

static Value *expandLerpIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Value *Y = Orig->getOperand(1);
  Value *S = Orig->getOperand(2);
  IRBuilder<> Builder(Orig);
  auto *V = Builder.CreateFSub(Y, X);
  V = Builder.CreateFMul(S, V);
  return Builder.CreateFAdd(X, V, "dx.lerp");
}

static Value *expandLogIntrinsic(CallInst *Orig,
                                 float LogConstVal = numbers::ln2f) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();
  Constant *Ln2Const =
      Ty->isVectorTy() ? ConstantVector::getSplat(
                             ElementCount::getFixed(
                                 cast<FixedVectorType>(Ty)->getNumElements()),
                             ConstantFP::get(EltTy, LogConstVal))
                       : ConstantFP::get(EltTy, LogConstVal);
  auto *Log2Call =
      Builder.CreateIntrinsic(Ty, Intrinsic::log2, {X}, nullptr, "elt.log2");
  Log2Call->setTailCall(Orig->isTailCall());
  Log2Call->setAttributes(Orig->getAttributes());
  return Builder.CreateFMul(Ln2Const, Log2Call);
}
static Value *expandLog10Intrinsic(CallInst *Orig) {
  return expandLogIntrinsic(Orig, numbers::ln2f / numbers::ln10f);
}

// Use dot product of vector operand with itself to calculate the length.
// Divide the vector by that length to normalize it.
static Value *expandNormalizeIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Type *Ty = Orig->getType();
  Type *EltTy = Ty->getScalarType();
  IRBuilder<> Builder(Orig);

  auto *XVec = dyn_cast<FixedVectorType>(Ty);
  if (!XVec) {
    if (auto *constantFP = dyn_cast<ConstantFP>(X)) {
      const APFloat &fpVal = constantFP->getValueAPF();
      if (fpVal.isZero())
        report_fatal_error(Twine("Invalid input scalar: length is zero"),
                           /* gen_crash_diag=*/false);
    }
    return Builder.CreateFDiv(X, X);
  }

  Value *DotProduct = expandFloatDotIntrinsic(Orig, X, X);

  // verify that the length is non-zero
  // (if the dot product is non-zero, then the length is non-zero)
  if (auto *constantFP = dyn_cast<ConstantFP>(DotProduct)) {
    const APFloat &fpVal = constantFP->getValueAPF();
    if (fpVal.isZero())
      report_fatal_error(Twine("Invalid input vector: length is zero"),
                         /* gen_crash_diag=*/false);
  }

  Value *Multiplicand = Builder.CreateIntrinsic(EltTy, Intrinsic::dx_rsqrt,
                                                ArrayRef<Value *>{DotProduct},
                                                nullptr, "dx.rsqrt");

  Value *MultiplicandVec =
      Builder.CreateVectorSplat(XVec->getNumElements(), Multiplicand);
  return Builder.CreateFMul(X, MultiplicandVec);
}

static Value *expandAtan2Intrinsic(CallInst *Orig) {
  Value *Y = Orig->getOperand(0);
  Value *X = Orig->getOperand(1);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);
  Builder.setFastMathFlags(Orig->getFastMathFlags());

  Value *Tan = Builder.CreateFDiv(Y, X);

  CallInst *Atan =
      Builder.CreateIntrinsic(Ty, Intrinsic::atan, {Tan}, nullptr, "Elt.Atan");
  Atan->setTailCall(Orig->isTailCall());
  Atan->setAttributes(Orig->getAttributes());

  // Modify atan result based on https://en.wikipedia.org/wiki/Atan2.
  Constant *Pi = ConstantFP::get(Ty, llvm::numbers::pi);
  Constant *HalfPi = ConstantFP::get(Ty, llvm::numbers::pi / 2);
  Constant *NegHalfPi = ConstantFP::get(Ty, -llvm::numbers::pi / 2);
  Constant *Zero = ConstantFP::get(Ty, 0);
  Value *AtanAddPi = Builder.CreateFAdd(Atan, Pi);
  Value *AtanSubPi = Builder.CreateFSub(Atan, Pi);

  // x > 0 -> atan.
  Value *Result = Atan;
  Value *XLt0 = Builder.CreateFCmpOLT(X, Zero);
  Value *XEq0 = Builder.CreateFCmpOEQ(X, Zero);
  Value *YGe0 = Builder.CreateFCmpOGE(Y, Zero);
  Value *YLt0 = Builder.CreateFCmpOLT(Y, Zero);

  // x < 0, y >= 0 -> atan + pi.
  Value *XLt0AndYGe0 = Builder.CreateAnd(XLt0, YGe0);
  Result = Builder.CreateSelect(XLt0AndYGe0, AtanAddPi, Result);

  // x < 0, y < 0 -> atan - pi.
  Value *XLt0AndYLt0 = Builder.CreateAnd(XLt0, YLt0);
  Result = Builder.CreateSelect(XLt0AndYLt0, AtanSubPi, Result);

  // x == 0, y < 0 -> -pi/2
  Value *XEq0AndYLt0 = Builder.CreateAnd(XEq0, YLt0);
  Result = Builder.CreateSelect(XEq0AndYLt0, NegHalfPi, Result);

  // x == 0, y > 0 -> pi/2
  Value *XEq0AndYGe0 = Builder.CreateAnd(XEq0, YGe0);
  Result = Builder.CreateSelect(XEq0AndYGe0, HalfPi, Result);

  return Result;
}

static Value *expandPowIntrinsic(CallInst *Orig) {

  Value *X = Orig->getOperand(0);
  Value *Y = Orig->getOperand(1);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);

  auto *Log2Call =
      Builder.CreateIntrinsic(Ty, Intrinsic::log2, {X}, nullptr, "elt.log2");
  auto *Mul = Builder.CreateFMul(Log2Call, Y);
  auto *Exp2Call =
      Builder.CreateIntrinsic(Ty, Intrinsic::exp2, {Mul}, nullptr, "elt.exp2");
  Exp2Call->setTailCall(Orig->isTailCall());
  Exp2Call->setAttributes(Orig->getAttributes());
  return Exp2Call;
}

static Value *expandStepIntrinsic(CallInst *Orig) {

  Value *X = Orig->getOperand(0);
  Value *Y = Orig->getOperand(1);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);

  Constant *One = ConstantFP::get(Ty->getScalarType(), 1.0);
  Constant *Zero = ConstantFP::get(Ty->getScalarType(), 0.0);
  Value *Cond = Builder.CreateFCmpOLT(Y, X);

  if (Ty != Ty->getScalarType()) {
    auto *XVec = dyn_cast<FixedVectorType>(Ty);
    One = ConstantVector::getSplat(
        ElementCount::getFixed(XVec->getNumElements()), One);
    Zero = ConstantVector::getSplat(
        ElementCount::getFixed(XVec->getNumElements()), Zero);
  }

  return Builder.CreateSelect(Cond, Zero, One);
}

static Value *expandRadiansIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);
  Value *PiOver180 = ConstantFP::get(Ty, llvm::numbers::pi / 180.0);
  return Builder.CreateFMul(X, PiOver180);
}

static Intrinsic::ID getMaxForClamp(Intrinsic::ID ClampIntrinsic) {
  if (ClampIntrinsic == Intrinsic::dx_uclamp)
    return Intrinsic::umax;
  if (ClampIntrinsic == Intrinsic::dx_sclamp)
    return Intrinsic::smax;
  assert(ClampIntrinsic == Intrinsic::dx_nclamp);
  return Intrinsic::maxnum;
}

static Intrinsic::ID getMinForClamp(Intrinsic::ID ClampIntrinsic) {
  if (ClampIntrinsic == Intrinsic::dx_uclamp)
    return Intrinsic::umin;
  if (ClampIntrinsic == Intrinsic::dx_sclamp)
    return Intrinsic::smin;
  assert(ClampIntrinsic == Intrinsic::dx_nclamp);
  return Intrinsic::minnum;
}

static Value *expandClampIntrinsic(CallInst *Orig,
                                   Intrinsic::ID ClampIntrinsic) {
  Value *X = Orig->getOperand(0);
  Value *Min = Orig->getOperand(1);
  Value *Max = Orig->getOperand(2);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);
  auto *MaxCall = Builder.CreateIntrinsic(Ty, getMaxForClamp(ClampIntrinsic),
                                          {X, Min}, nullptr, "dx.max");
  return Builder.CreateIntrinsic(Ty, getMinForClamp(ClampIntrinsic),
                                 {MaxCall, Max}, nullptr, "dx.min");
}

static Value *expandDegreesIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);
  Value *DegreesRatio = ConstantFP::get(Ty, 180.0 * llvm::numbers::inv_pi);
  return Builder.CreateFMul(X, DegreesRatio);
}

static Value *expandSignIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Type *Ty = X->getType();
  Type *ScalarTy = Ty->getScalarType();
  Type *RetTy = Orig->getType();
  Constant *Zero = Constant::getNullValue(Ty);

  IRBuilder<> Builder(Orig);

  Value *GT;
  Value *LT;
  if (ScalarTy->isFloatingPointTy()) {
    GT = Builder.CreateFCmpOLT(Zero, X);
    LT = Builder.CreateFCmpOLT(X, Zero);
  } else {
    assert(ScalarTy->isIntegerTy());
    GT = Builder.CreateICmpSLT(Zero, X);
    LT = Builder.CreateICmpSLT(X, Zero);
  }

  Value *ZextGT = Builder.CreateZExt(GT, RetTy);
  Value *ZextLT = Builder.CreateZExt(LT, RetTy);

  return Builder.CreateSub(ZextGT, ZextLT);
}

static bool expandIntrinsic(Function &F, CallInst *Orig) {
  Value *Result = nullptr;
  Intrinsic::ID IntrinsicId = F.getIntrinsicID();
  switch (IntrinsicId) {
  case Intrinsic::abs:
    Result = expandAbs(Orig);
    break;
  case Intrinsic::atan2:
    Result = expandAtan2Intrinsic(Orig);
    break;
  case Intrinsic::exp:
    Result = expandExpIntrinsic(Orig);
    break;
  case Intrinsic::log:
    Result = expandLogIntrinsic(Orig);
    break;
  case Intrinsic::log10:
    Result = expandLog10Intrinsic(Orig);
    break;
  case Intrinsic::pow:
    Result = expandPowIntrinsic(Orig);
    break;
  case Intrinsic::dx_all:
  case Intrinsic::dx_any:
    Result = expandAnyOrAllIntrinsic(Orig, IntrinsicId);
    break;
  case Intrinsic::dx_cross:
    Result = expandCrossIntrinsic(Orig);
    break;
  case Intrinsic::dx_uclamp:
  case Intrinsic::dx_sclamp:
  case Intrinsic::dx_nclamp:
    Result = expandClampIntrinsic(Orig, IntrinsicId);
    break;
  case Intrinsic::dx_degrees:
    Result = expandDegreesIntrinsic(Orig);
    break;
  case Intrinsic::dx_lerp:
    Result = expandLerpIntrinsic(Orig);
    break;
  case Intrinsic::dx_length:
    Result = expandLengthIntrinsic(Orig);
    break;
  case Intrinsic::dx_normalize:
    Result = expandNormalizeIntrinsic(Orig);
    break;
  case Intrinsic::dx_fdot:
    Result = expandFloatDotIntrinsic(Orig);
    break;
  case Intrinsic::dx_sdot:
  case Intrinsic::dx_udot:
    Result = expandIntegerDotIntrinsic(Orig, IntrinsicId);
    break;
  case Intrinsic::dx_sign:
    Result = expandSignIntrinsic(Orig);
    break;
  case Intrinsic::dx_step:
    Result = expandStepIntrinsic(Orig);
    break;
  case Intrinsic::dx_radians:
    Result = expandRadiansIntrinsic(Orig);
    break;
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_fadd:
    Result = expandVecReduceAdd(Orig, IntrinsicId);
    break;
  }
  if (Result) {
    Orig->replaceAllUsesWith(Result);
    Orig->eraseFromParent();
    return true;
  }
  return false;
}

static bool expansionIntrinsics(Module &M) {
  for (auto &F : make_early_inc_range(M.functions())) {
    if (!isIntrinsicExpansion(F))
      continue;
    bool IntrinsicExpanded = false;
    for (User *U : make_early_inc_range(F.users())) {
      auto *IntrinsicCall = dyn_cast<CallInst>(U);
      if (!IntrinsicCall)
        continue;
      IntrinsicExpanded = expandIntrinsic(F, IntrinsicCall);
    }
    if (F.user_empty() && IntrinsicExpanded)
      F.eraseFromParent();
  }
  return true;
}

PreservedAnalyses DXILIntrinsicExpansion::run(Module &M,
                                              ModuleAnalysisManager &) {
  if (expansionIntrinsics(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

bool DXILIntrinsicExpansionLegacy::runOnModule(Module &M) {
  return expansionIntrinsics(M);
}

void DXILIntrinsicExpansionLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<DXILResourceWrapperPass>();
}

char DXILIntrinsicExpansionLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                      "DXIL Intrinsic Expansion", false, false)
INITIALIZE_PASS_END(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                    "DXIL Intrinsic Expansion", false, false)

ModulePass *llvm::createDXILIntrinsicExpansionLegacyPass() {
  return new DXILIntrinsicExpansionLegacy();
}
