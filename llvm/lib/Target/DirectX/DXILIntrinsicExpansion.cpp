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
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "dxil-intrinsic-expansion"

using namespace llvm;

class DXILIntrinsicExpansionLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILIntrinsicExpansionLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};

static bool resourceAccessNeeds64BitExpansion(Module *M, Type *OverloadTy,
                                              bool IsRaw) {
  if (IsRaw && M->getTargetTriple().getDXILVersion() > VersionTuple(1, 2))
    return false;

  Type *ScalarTy = OverloadTy->getScalarType();
  return ScalarTy->isDoubleTy() || ScalarTy->isIntegerTy(64);
}

static Value *expand16BitIsInf(CallInst *Orig) {
  Module *M = Orig->getModule();
  if (M->getTargetTriple().getDXILVersion() >= VersionTuple(1, 9))
    return nullptr;

  Value *Val = Orig->getOperand(0);
  Type *ValTy = Val->getType();
  if (!ValTy->getScalarType()->isHalfTy())
    return nullptr;

  IRBuilder<> Builder(Orig);
  Type *IType = Type::getInt16Ty(M->getContext());
  Constant *PosInf =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0x7c00))
          : ConstantInt::get(IType, 0x7c00);

  Constant *NegInf =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0xfc00))
          : ConstantInt::get(IType, 0xfc00);

  Value *IVal = Builder.CreateBitCast(Val, PosInf->getType());
  Value *B1 = Builder.CreateICmpEQ(IVal, PosInf);
  Value *B2 = Builder.CreateICmpEQ(IVal, NegInf);
  Value *B3 = Builder.CreateOr(B1, B2);
  return B3;
}

static Value *expand16BitIsNaN(CallInst *Orig) {
  Module *M = Orig->getModule();
  if (M->getTargetTriple().getDXILVersion() >= VersionTuple(1, 9))
    return nullptr;

  Value *Val = Orig->getOperand(0);
  Type *ValTy = Val->getType();
  if (!ValTy->getScalarType()->isHalfTy())
    return nullptr;

  IRBuilder<> Builder(Orig);
  Type *IType = Type::getInt16Ty(M->getContext());

  Constant *ExpBitMask =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0x7c00))
          : ConstantInt::get(IType, 0x7c00);
  Constant *SigBitMask =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0x3ff))
          : ConstantInt::get(IType, 0x3ff);

  Constant *Zero =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0))
          : ConstantInt::get(IType, 0);

  Value *IVal = Builder.CreateBitCast(Val, ExpBitMask->getType());
  Value *Exp = Builder.CreateAnd(IVal, ExpBitMask);
  Value *B1 = Builder.CreateICmpEQ(Exp, ExpBitMask);

  Value *Sig = Builder.CreateAnd(IVal, SigBitMask);
  Value *B2 = Builder.CreateICmpNE(Sig, Zero);
  Value *B3 = Builder.CreateAnd(B1, B2);
  return B3;
}

static Value *expand16BitIsFinite(CallInst *Orig) {
  Module *M = Orig->getModule();
  if (M->getTargetTriple().getDXILVersion() >= VersionTuple(1, 9))
    return nullptr;

  Value *Val = Orig->getOperand(0);
  Type *ValTy = Val->getType();
  if (!ValTy->getScalarType()->isHalfTy())
    return nullptr;

  IRBuilder<> Builder(Orig);
  Type *IType = Type::getInt16Ty(M->getContext());

  Constant *ExpBitMask =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0x7c00))
          : ConstantInt::get(IType, 0x7c00);

  Value *IVal = Builder.CreateBitCast(Val, ExpBitMask->getType());
  Value *Exp = Builder.CreateAnd(IVal, ExpBitMask);
  Value *B1 = Builder.CreateICmpNE(Exp, ExpBitMask);
  return B1;
}

static Value *expand16BitIsNormal(CallInst *Orig) {
  Module *M = Orig->getModule();
  if (M->getTargetTriple().getDXILVersion() >= VersionTuple(1, 9))
    return nullptr;

  Value *Val = Orig->getOperand(0);
  Type *ValTy = Val->getType();
  if (!ValTy->getScalarType()->isHalfTy())
    return nullptr;

  IRBuilder<> Builder(Orig);
  Type *IType = Type::getInt16Ty(M->getContext());

  Constant *ExpBitMask =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0x7c00))
          : ConstantInt::get(IType, 0x7c00);
  Constant *Zero =
      ValTy->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    cast<FixedVectorType>(ValTy)->getNumElements()),
                ConstantInt::get(IType, 0))
          : ConstantInt::get(IType, 0);

  Value *IVal = Builder.CreateBitCast(Val, ExpBitMask->getType());
  Value *Exp = Builder.CreateAnd(IVal, ExpBitMask);
  Value *NotAllZeroes = Builder.CreateICmpNE(Exp, Zero);
  Value *NotAllOnes = Builder.CreateICmpNE(Exp, ExpBitMask);
  Value *B1 = Builder.CreateAnd(NotAllZeroes, NotAllOnes);
  return B1;
}

static bool isIntrinsicExpansion(Function &F) {
  switch (F.getIntrinsicID()) {
  case Intrinsic::abs:
  case Intrinsic::atan2:
  case Intrinsic::exp:
  case Intrinsic::is_fpclass:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::dx_all:
  case Intrinsic::dx_any:
  case Intrinsic::dx_cross:
  case Intrinsic::dx_uclamp:
  case Intrinsic::dx_sclamp:
  case Intrinsic::dx_nclamp:
  case Intrinsic::dx_degrees:
  case Intrinsic::dx_isinf:
  case Intrinsic::dx_lerp:
  case Intrinsic::dx_normalize:
  case Intrinsic::dx_fdot:
  case Intrinsic::dx_sdot:
  case Intrinsic::dx_udot:
  case Intrinsic::dx_sign:
  case Intrinsic::dx_step:
  case Intrinsic::dx_radians:
  case Intrinsic::usub_sat:
  case Intrinsic::vector_reduce_add:
  case Intrinsic::vector_reduce_fadd:
    return true;
  case Intrinsic::dx_resource_load_rawbuffer:
    return resourceAccessNeeds64BitExpansion(
        F.getParent(), F.getReturnType()->getStructElementType(0),
        /*IsRaw*/ true);
  case Intrinsic::dx_resource_load_typedbuffer:
    return resourceAccessNeeds64BitExpansion(
        F.getParent(), F.getReturnType()->getStructElementType(0),
        /*IsRaw*/ false);
  case Intrinsic::dx_resource_store_rawbuffer:
    return resourceAccessNeeds64BitExpansion(
        F.getParent(), F.getFunctionType()->getParamType(3), /*IsRaw*/ true);
  case Intrinsic::dx_resource_store_typedbuffer:
    return resourceAccessNeeds64BitExpansion(
        F.getParent(), F.getFunctionType()->getParamType(2), /*IsRaw*/ false);
  }
  return false;
}

static Value *expandUsubSat(CallInst *Orig) {
  Value *A = Orig->getArgOperand(0);
  Value *B = Orig->getArgOperand(1);
  Type *Ty = A->getType();

  IRBuilder<> Builder(Orig);

  Value *Cmp = Builder.CreateICmpULT(A, B, "usub.cmp");
  Value *Sub = Builder.CreateSub(A, B, "usub.sub");
  Value *Zero = ConstantInt::get(Ty, 0);
  return Builder.CreateSelect(Cmp, Zero, Sub, "usub.sat");
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
    reportFatalUsageError("return vector must have exactly 3 elements");

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

  Value *cross = PoisonValue::get(VT);
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
  int NumElts = AVec->getNumElements();
  switch (NumElts) {
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
    reportFatalUsageError(
        "Invalid dot product input vector: length is outside 2-4");
    return nullptr;
  }

  SmallVector<Value *> Args;
  for (int I = 0; I < NumElts; ++I)
    Args.push_back(Builder.CreateExtractElement(A, Builder.getInt32(I)));
  for (int I = 0; I < NumElts; ++I)
    Args.push_back(Builder.CreateExtractElement(B, Builder.getInt32(I)));
  return Builder.CreateIntrinsic(ATy->getScalarType(), DotIntrinsic, Args,
                                 nullptr, "dot");
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

static Value *expandIsFPClass(CallInst *Orig) {
  Value *T = Orig->getArgOperand(1);
  auto *TCI = dyn_cast<ConstantInt>(T);

  // These FPClassTest cases have DXIL opcodes, so they will be handled in
  // DXIL Op Lowering instead for all non f16 cases.
  switch (TCI->getZExtValue()) {
  case FPClassTest::fcInf:
    return expand16BitIsInf(Orig);
  case FPClassTest::fcNan:
    return expand16BitIsNaN(Orig);
  case FPClassTest::fcNormal:
    return expand16BitIsNormal(Orig);
  case FPClassTest::fcFinite:
    return expand16BitIsFinite(Orig);
  }

  IRBuilder<> Builder(Orig);

  Value *F = Orig->getArgOperand(0);
  Type *FTy = F->getType();
  unsigned FNumElem = 0; // 0 => F is not a vector

  unsigned BitWidth; // Bit width of F or the ElemTy of F
  Type *BitCastTy;   // An IntNTy of the same bitwidth as F or ElemTy of F

  if (auto *FVecTy = dyn_cast<FixedVectorType>(FTy)) {
    Type *ElemTy = FVecTy->getElementType();
    FNumElem = FVecTy->getNumElements();
    BitWidth = ElemTy->getPrimitiveSizeInBits();
    BitCastTy = FixedVectorType::get(Builder.getIntNTy(BitWidth), FNumElem);
  } else {
    BitWidth = FTy->getPrimitiveSizeInBits();
    BitCastTy = Builder.getIntNTy(BitWidth);
  }

  Value *FBitCast = Builder.CreateBitCast(F, BitCastTy);
  switch (TCI->getZExtValue()) {
  case FPClassTest::fcNegZero: {
    Value *NegZero =
        ConstantInt::get(Builder.getIntNTy(BitWidth), 1 << (BitWidth - 1));
    Value *RetVal;
    if (FNumElem) {
      Value *NegZeroSplat = Builder.CreateVectorSplat(FNumElem, NegZero);
      RetVal =
          Builder.CreateICmpEQ(FBitCast, NegZeroSplat, "is.fpclass.negzero");
    } else
      RetVal = Builder.CreateICmpEQ(FBitCast, NegZero, "is.fpclass.negzero");
    return RetVal;
  }
  default:
    reportFatalUsageError("Unsupported FPClassTest");
  }
}

static Value *expandAnyOrAllIntrinsic(CallInst *Orig,
                                      Intrinsic::ID IntrinsicId) {
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
      Result = ApplyOp(IntrinsicId, Result, Elt);
    }
  }
  return Result;
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
        reportFatalUsageError("Invalid input scalar: length is zero");
    }
    return Builder.CreateFDiv(X, X);
  }

  Value *DotProduct = expandFloatDotIntrinsic(Orig, X, X);

  // verify that the length is non-zero
  // (if the dot product is non-zero, then the length is non-zero)
  if (auto *constantFP = dyn_cast<ConstantFP>(DotProduct)) {
    const APFloat &fpVal = constantFP->getValueAPF();
    if (fpVal.isZero())
      reportFatalUsageError("Invalid input vector: length is zero");
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

static Value *expandPowIntrinsic(CallInst *Orig, Intrinsic::ID IntrinsicId) {

  Value *X = Orig->getOperand(0);
  Value *Y = Orig->getOperand(1);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig);

  if (IntrinsicId == Intrinsic::powi)
    Y = Builder.CreateSIToFP(Y, Ty);

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

static bool expandBufferLoadIntrinsic(CallInst *Orig, bool IsRaw) {
  IRBuilder<> Builder(Orig);

  Type *BufferTy = Orig->getType()->getStructElementType(0);
  Type *ScalarTy = BufferTy->getScalarType();
  bool IsDouble = ScalarTy->isDoubleTy();
  assert(IsDouble || ScalarTy->isIntegerTy(64) &&
                         "Only expand double or int64 scalars or vectors");
  bool IsVector = false;
  unsigned ExtractNum = 2;
  if (auto *VT = dyn_cast<FixedVectorType>(BufferTy)) {
    ExtractNum = 2 * VT->getNumElements();
    IsVector = true;
    assert(IsRaw || ExtractNum == 4 && "TypedBufferLoad vector must be size 2");
  }

  SmallVector<Value *, 2> Loads;
  Value *Result = PoisonValue::get(BufferTy);
  unsigned Base = 0;
  // If we need to extract more than 4 i32; we need to break it up into
  // more than one load. LoadNum tells us how many i32s we are loading in
  // each load
  while (ExtractNum > 0) {
    unsigned LoadNum = std::min(ExtractNum, 4u);
    Type *Ty = VectorType::get(Builder.getInt32Ty(), LoadNum, false);

    Type *LoadType = StructType::get(Ty, Builder.getInt1Ty());
    Intrinsic::ID LoadIntrinsic = Intrinsic::dx_resource_load_typedbuffer;
    SmallVector<Value *, 3> Args = {Orig->getOperand(0), Orig->getOperand(1)};
    if (IsRaw) {
      LoadIntrinsic = Intrinsic::dx_resource_load_rawbuffer;
      Value *Tmp = Builder.getInt32(4 * Base * 2);
      Args.push_back(Builder.CreateAdd(Orig->getOperand(2), Tmp));
    }

    CallInst *Load = Builder.CreateIntrinsic(LoadType, LoadIntrinsic, Args);
    Loads.push_back(Load);

    // extract the buffer load's result
    Value *Extract = Builder.CreateExtractValue(Load, {0});

    SmallVector<Value *> ExtractElements;
    for (unsigned I = 0; I < LoadNum; ++I)
      ExtractElements.push_back(
          Builder.CreateExtractElement(Extract, Builder.getInt32(I)));

    // combine into double(s) or int64(s)
    for (unsigned I = 0; I < LoadNum; I += 2) {
      Value *Combined = nullptr;
      if (IsDouble)
        // For doubles, use dx_asdouble intrinsic
        Combined = Builder.CreateIntrinsic(
            Builder.getDoubleTy(), Intrinsic::dx_asdouble,
            {ExtractElements[I], ExtractElements[I + 1]});
      else {
        // For int64, manually combine two int32s
        // First, zero-extend both values to i64
        Value *Lo =
            Builder.CreateZExt(ExtractElements[I], Builder.getInt64Ty());
        Value *Hi =
            Builder.CreateZExt(ExtractElements[I + 1], Builder.getInt64Ty());
        // Shift the high bits left by 32 bits
        Value *ShiftedHi = Builder.CreateShl(Hi, Builder.getInt64(32));
        // OR the high and low bits together
        Combined = Builder.CreateOr(Lo, ShiftedHi);
      }

      if (IsVector)
        Result = Builder.CreateInsertElement(Result, Combined,
                                             Builder.getInt32((I / 2) + Base));
      else
        Result = Combined;
    }

    ExtractNum -= LoadNum;
    Base += LoadNum / 2;
  }

  Value *CheckBit = nullptr;
  for (User *U : make_early_inc_range(Orig->users())) {
    // If it's not a ExtractValueInst, we don't know how to
    // handle it
    auto *EVI = dyn_cast<ExtractValueInst>(U);
    if (!EVI)
      llvm_unreachable("Unexpected user of typedbufferload");

    ArrayRef<unsigned> Indices = EVI->getIndices();
    assert(Indices.size() == 1);

    if (Indices[0] == 0) {
      // Use of the value(s)
      EVI->replaceAllUsesWith(Result);
    } else {
      // Use of the check bit
      assert(Indices[0] == 1 && "Unexpected type for typedbufferload");
      // Note: This does not always match the historical behaviour of DXC.
      // See https://github.com/microsoft/DirectXShaderCompiler/issues/7622
      if (!CheckBit) {
        SmallVector<Value *, 2> CheckBits;
        for (Value *L : Loads)
          CheckBits.push_back(Builder.CreateExtractValue(L, {1}));
        CheckBit = Builder.CreateAnd(CheckBits);
      }
      EVI->replaceAllUsesWith(CheckBit);
    }
    EVI->eraseFromParent();
  }
  Orig->eraseFromParent();
  return true;
}

static bool expandBufferStoreIntrinsic(CallInst *Orig, bool IsRaw) {
  IRBuilder<> Builder(Orig);

  unsigned ValIndex = IsRaw ? 3 : 2;
  Type *BufferTy = Orig->getFunctionType()->getParamType(ValIndex);
  Type *ScalarTy = BufferTy->getScalarType();
  bool IsDouble = ScalarTy->isDoubleTy();
  assert((IsDouble || ScalarTy->isIntegerTy(64)) &&
         "Only expand double or int64 scalars or vectors");

  // Determine if we're dealing with a vector or scalar
  bool IsVector = false;
  unsigned ExtractNum = 2;
  unsigned VecLen = 0;
  if (auto *VT = dyn_cast<FixedVectorType>(BufferTy)) {
    VecLen = VT->getNumElements();
    assert(IsRaw || VecLen == 2 && "TypedBufferStore vector must be size 2");
    ExtractNum = VecLen * 2;
    IsVector = true;
  }

  // Create the appropriate vector type for the result
  Type *Int32Ty = Builder.getInt32Ty();
  Type *ResultTy = VectorType::get(Int32Ty, ExtractNum, false);
  Value *Val = PoisonValue::get(ResultTy);

  Type *SplitElementTy = Int32Ty;
  if (IsVector)
    SplitElementTy = VectorType::get(SplitElementTy, VecLen, false);

  Value *LowBits = nullptr;
  Value *HighBits = nullptr;
  // Split the 64-bit values into 32-bit components
  if (IsDouble) {
    auto *SplitTy = llvm::StructType::get(SplitElementTy, SplitElementTy);
    Value *Split = Builder.CreateIntrinsic(SplitTy, Intrinsic::dx_splitdouble,
                                           {Orig->getOperand(ValIndex)});
    LowBits = Builder.CreateExtractValue(Split, 0);
    HighBits = Builder.CreateExtractValue(Split, 1);
  } else {
    // Handle int64 type(s)
    Value *InputVal = Orig->getOperand(ValIndex);
    Constant *ShiftAmt = Builder.getInt64(32);
    if (IsVector)
      ShiftAmt =
          ConstantVector::getSplat(ElementCount::getFixed(VecLen), ShiftAmt);

    // Split into low and high 32-bit parts
    LowBits = Builder.CreateTrunc(InputVal, SplitElementTy);
    Value *ShiftedVal = Builder.CreateLShr(InputVal, ShiftAmt);
    HighBits = Builder.CreateTrunc(ShiftedVal, SplitElementTy);
  }

  if (IsVector) {
    SmallVector<int, 8> Mask;
    for (unsigned I = 0; I < VecLen; ++I) {
      Mask.push_back(I);
      Mask.push_back(I + VecLen);
    }
    Val = Builder.CreateShuffleVector(LowBits, HighBits, Mask);
  } else {
    Val = Builder.CreateInsertElement(Val, LowBits, Builder.getInt32(0));
    Val = Builder.CreateInsertElement(Val, HighBits, Builder.getInt32(1));
  }

  // If we need to extract more than 4 i32; we need to break it up into
  // more than one store. StoreNum tells us how many i32s we are storing in
  // each store
  unsigned Base = 0;
  while (ExtractNum > 0) {
    unsigned StoreNum = std::min(ExtractNum, 4u);

    Intrinsic::ID StoreIntrinsic = Intrinsic::dx_resource_store_typedbuffer;
    SmallVector<Value *, 4> Args = {Orig->getOperand(0), Orig->getOperand(1)};
    if (IsRaw) {
      StoreIntrinsic = Intrinsic::dx_resource_store_rawbuffer;
      Value *Tmp = Builder.getInt32(4 * Base);
      Args.push_back(Builder.CreateAdd(Orig->getOperand(2), Tmp));
    }

    SmallVector<int, 4> Mask;
    for (unsigned I = 0; I < StoreNum; ++I) {
      Mask.push_back(Base + I);
    }

    Value *SubVal = Val;
    if (VecLen > 2)
      SubVal = Builder.CreateShuffleVector(Val, Mask);

    Args.push_back(SubVal);
    // Create the final intrinsic call
    Builder.CreateIntrinsic(Builder.getVoidTy(), StoreIntrinsic, Args);

    ExtractNum -= StoreNum;
    Base += StoreNum;
  }
  Orig->eraseFromParent();
  return true;
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
  case Intrinsic::is_fpclass:
    Result = expandIsFPClass(Orig);
    break;
  case Intrinsic::log:
    Result = expandLogIntrinsic(Orig);
    break;
  case Intrinsic::log10:
    Result = expandLog10Intrinsic(Orig);
    break;
  case Intrinsic::pow:
  case Intrinsic::powi:
    Result = expandPowIntrinsic(Orig, IntrinsicId);
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
  case Intrinsic::dx_isinf:
    Result = expand16BitIsInf(Orig);
    break;
  case Intrinsic::dx_lerp:
    Result = expandLerpIntrinsic(Orig);
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
  case Intrinsic::dx_resource_load_rawbuffer:
    if (expandBufferLoadIntrinsic(Orig, /*IsRaw*/ true))
      return true;
    break;
  case Intrinsic::dx_resource_store_rawbuffer:
    if (expandBufferStoreIntrinsic(Orig, /*IsRaw*/ true))
      return true;
    break;
  case Intrinsic::dx_resource_load_typedbuffer:
    if (expandBufferLoadIntrinsic(Orig, /*IsRaw*/ false))
      return true;
    break;
  case Intrinsic::dx_resource_store_typedbuffer:
    if (expandBufferStoreIntrinsic(Orig, /*IsRaw*/ false))
      return true;
    break;
  case Intrinsic::usub_sat:
    Result = expandUsubSat(Orig);
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

char DXILIntrinsicExpansionLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                      "DXIL Intrinsic Expansion", false, false)
INITIALIZE_PASS_END(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                    "DXIL Intrinsic Expansion", false, false)

ModulePass *llvm::createDXILIntrinsicExpansionLegacyPass() {
  return new DXILIntrinsicExpansionLegacy();
}
