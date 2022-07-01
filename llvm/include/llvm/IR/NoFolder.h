//===- NoFolder.h - Constant folding helper ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NoFolder class, a helper for IRBuilder.  It provides
// IRBuilder with a set of methods for creating unfolded constants.  This is
// useful for learners trying to understand how LLVM IR works, and who don't
// want details to be hidden by the constant folder.  For general constant
// creation and folding, use ConstantExpr and the routines in
// llvm/Analysis/ConstantFolding.h.
//
// Note: since it is not actually possible to create unfolded constants, this
// class returns instructions rather than constants.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_NOFOLDER_H
#define LLVM_IR_NOFOLDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilderFolder.h"

namespace llvm {

/// NoFolder - Create "constants" (actually, instructions) with no folding.
class NoFolder final : public IRBuilderFolder {
  virtual void anchor();

public:
  explicit NoFolder() = default;

  //===--------------------------------------------------------------------===//
  // Value-based folders.
  //
  // Return an existing value or a constant if the operation can be simplified.
  // Otherwise return nullptr.
  //===--------------------------------------------------------------------===//
  Value *FoldAdd(Value *LHS, Value *RHS, bool HasNUW = false,
                 bool HasNSW = false) const override {
    return nullptr;
  }

  Value *FoldAnd(Value *LHS, Value *RHS) const override { return nullptr; }

  Value *FoldOr(Value *LHS, Value *RHS) const override { return nullptr; }

  Value *FoldICmp(CmpInst::Predicate P, Value *LHS, Value *RHS) const override {
    return nullptr;
  }

  Value *FoldGEP(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                 bool IsInBounds = false) const override {
    return nullptr;
  }

  Value *FoldSelect(Value *C, Value *True, Value *False) const override {
    return nullptr;
  }

  Value *FoldExtractValue(Value *Agg,
                          ArrayRef<unsigned> IdxList) const override {
    return nullptr;
  }

  Value *FoldInsertValue(Value *Agg, Value *Val,
                         ArrayRef<unsigned> IdxList) const override {
    return nullptr;
  }

  Value *FoldExtractElement(Value *Vec, Value *Idx) const override {
    return nullptr;
  }

  Value *FoldInsertElement(Value *Vec, Value *NewElt,
                           Value *Idx) const override {
    return nullptr;
  }

  Value *FoldShuffleVector(Value *V1, Value *V2,
                           ArrayRef<int> Mask) const override {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateFAdd(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateFAdd(LHS, RHS);
  }

  Instruction *CreateSub(Constant *LHS, Constant *RHS,
                         bool HasNUW = false,
                         bool HasNSW = false) const override {
    BinaryOperator *BO = BinaryOperator::CreateSub(LHS, RHS);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *CreateFSub(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateFSub(LHS, RHS);
  }

  Instruction *CreateMul(Constant *LHS, Constant *RHS,
                         bool HasNUW = false,
                         bool HasNSW = false) const override {
    BinaryOperator *BO = BinaryOperator::CreateMul(LHS, RHS);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *CreateFMul(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateFMul(LHS, RHS);
  }

  Instruction *CreateUDiv(Constant *LHS, Constant *RHS,
                          bool isExact = false) const override {
    if (!isExact)
      return BinaryOperator::CreateUDiv(LHS, RHS);
    return BinaryOperator::CreateExactUDiv(LHS, RHS);
  }

  Instruction *CreateSDiv(Constant *LHS, Constant *RHS,
                          bool isExact = false) const override {
    if (!isExact)
      return BinaryOperator::CreateSDiv(LHS, RHS);
    return BinaryOperator::CreateExactSDiv(LHS, RHS);
  }

  Instruction *CreateFDiv(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateFDiv(LHS, RHS);
  }

  Instruction *CreateURem(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateURem(LHS, RHS);
  }

  Instruction *CreateSRem(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateSRem(LHS, RHS);
  }

  Instruction *CreateFRem(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateFRem(LHS, RHS);
  }

  Instruction *CreateShl(Constant *LHS, Constant *RHS, bool HasNUW = false,
                         bool HasNSW = false) const override {
    BinaryOperator *BO = BinaryOperator::CreateShl(LHS, RHS);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *CreateLShr(Constant *LHS, Constant *RHS,
                          bool isExact = false) const override {
    if (!isExact)
      return BinaryOperator::CreateLShr(LHS, RHS);
    return BinaryOperator::CreateExactLShr(LHS, RHS);
  }

  Instruction *CreateAShr(Constant *LHS, Constant *RHS,
                          bool isExact = false) const override {
    if (!isExact)
      return BinaryOperator::CreateAShr(LHS, RHS);
    return BinaryOperator::CreateExactAShr(LHS, RHS);
  }

  Instruction *CreateXor(Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::CreateXor(LHS, RHS);
  }

  Instruction *CreateBinOp(Instruction::BinaryOps Opc,
                           Constant *LHS, Constant *RHS) const override {
    return BinaryOperator::Create(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateNeg(Constant *C,
                         bool HasNUW = false,
                         bool HasNSW = false) const override {
    BinaryOperator *BO = BinaryOperator::CreateNeg(C);
    if (HasNUW) BO->setHasNoUnsignedWrap();
    if (HasNSW) BO->setHasNoSignedWrap();
    return BO;
  }

  Instruction *CreateFNeg(Constant *C) const override {
    return UnaryOperator::CreateFNeg(C);
  }

  Instruction *CreateNot(Constant *C) const override {
    return BinaryOperator::CreateNot(C);
  }

  Instruction *CreateUnOp(Instruction::UnaryOps Opc,
                          Constant *C) const override {
    return UnaryOperator::Create(Opc, C);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Instruction *CreateCast(Instruction::CastOps Op, Constant *C,
                          Type *DestTy) const override {
    return CastInst::Create(Op, C, DestTy);
  }

  Instruction *CreatePointerCast(Constant *C, Type *DestTy) const override {
    return CastInst::CreatePointerCast(C, DestTy);
  }

  Instruction *CreatePointerBitCastOrAddrSpaceCast(
      Constant *C, Type *DestTy) const override {
    return CastInst::CreatePointerBitCastOrAddrSpaceCast(C, DestTy);
  }

  Instruction *CreateIntCast(Constant *C, Type *DestTy,
                             bool isSigned) const override {
    return CastInst::CreateIntegerCast(C, DestTy, isSigned);
  }

  Instruction *CreateFPCast(Constant *C, Type *DestTy) const override {
    return CastInst::CreateFPCast(C, DestTy);
  }

  Instruction *CreateBitCast(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }

  Instruction *CreateIntToPtr(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }

  Instruction *CreatePtrToInt(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }

  Instruction *CreateZExtOrBitCast(Constant *C, Type *DestTy) const override {
    return CastInst::CreateZExtOrBitCast(C, DestTy);
  }

  Instruction *CreateSExtOrBitCast(Constant *C, Type *DestTy) const override {
    return CastInst::CreateSExtOrBitCast(C, DestTy);
  }

  Instruction *CreateTruncOrBitCast(Constant *C, Type *DestTy) const override {
    return CastInst::CreateTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Instruction *CreateFCmp(CmpInst::Predicate P,
                          Constant *LHS, Constant *RHS) const override {
    return new FCmpInst(P, LHS, RHS);
  }
};

} // end namespace llvm

#endif // LLVM_IR_NOFOLDER_H
