//===- ConstantFolder.h - Constant folding helper ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstantFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants
// with minimal folding.  For general constant creation and folding,
// use ConstantExpr and the routines in llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONSTANTFOLDER_H
#define LLVM_IR_CONSTANTFOLDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/IRBuilderFolder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"

namespace llvm {

/// ConstantFolder - Create constants with minimum, target independent, folding.
class ConstantFolder final : public IRBuilderFolder {
  virtual void anchor();

public:
  explicit ConstantFolder() = default;

  //===--------------------------------------------------------------------===//
  // Value-based folders.
  //
  // Return an existing value or a constant if the operation can be simplified.
  // Otherwise return nullptr.
  //===--------------------------------------------------------------------===//

  Value *FoldBinOp(Instruction::BinaryOps Opc, Value *LHS,
                   Value *RHS) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC) {
      if (ConstantExpr::isDesirableBinOp(Opc))
        return ConstantExpr::get(Opc, LC, RC);
      return ConstantFoldBinaryInstruction(Opc, LC, RC);
    }
    return nullptr;
  }

  Value *FoldExactBinOp(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                        bool IsExact) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC) {
      if (ConstantExpr::isDesirableBinOp(Opc))
        return ConstantExpr::get(Opc, LC, RC,
                                 IsExact ? PossiblyExactOperator::IsExact : 0);
      return ConstantFoldBinaryInstruction(Opc, LC, RC);
    }
    return nullptr;
  }

  Value *FoldNoWrapBinOp(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                         bool HasNUW, bool HasNSW) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC) {
      if (ConstantExpr::isDesirableBinOp(Opc)) {
        unsigned Flags = 0;
        if (HasNUW)
          Flags |= OverflowingBinaryOperator::NoUnsignedWrap;
        if (HasNSW)
          Flags |= OverflowingBinaryOperator::NoSignedWrap;
        return ConstantExpr::get(Opc, LC, RC, Flags);
      }
      return ConstantFoldBinaryInstruction(Opc, LC, RC);
    }
    return nullptr;
  }

  Value *FoldBinOpFMF(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                      FastMathFlags FMF) const override {
    return FoldBinOp(Opc, LHS, RHS);
  }

  Value *FoldUnOpFMF(Instruction::UnaryOps Opc, Value *V,
                      FastMathFlags FMF) const override {
    if (Constant *C = dyn_cast<Constant>(V))
      return ConstantFoldUnaryInstruction(Opc, C);
    return nullptr;
  }

  Value *FoldICmp(CmpInst::Predicate P, Value *LHS, Value *RHS) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC)
      return ConstantExpr::getCompare(P, LC, RC);
    return nullptr;
  }

  Value *FoldGEP(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                 bool IsInBounds = false) const override {
    if (!ConstantExpr::isSupportedGetElementPtr(Ty))
      return nullptr;

    if (auto *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      if (any_of(IdxList, [](Value *V) { return !isa<Constant>(V); }))
        return nullptr;

      if (IsInBounds)
        return ConstantExpr::getInBoundsGetElementPtr(Ty, PC, IdxList);
      else
        return ConstantExpr::getGetElementPtr(Ty, PC, IdxList);
    }
    return nullptr;
  }

  Value *FoldSelect(Value *C, Value *True, Value *False) const override {
    auto *CC = dyn_cast<Constant>(C);
    auto *TC = dyn_cast<Constant>(True);
    auto *FC = dyn_cast<Constant>(False);
    if (CC && TC && FC)
      return ConstantFoldSelectInstruction(CC, TC, FC);
    return nullptr;
  }

  Value *FoldExtractValue(Value *Agg,
                          ArrayRef<unsigned> IdxList) const override {
    if (auto *CAgg = dyn_cast<Constant>(Agg))
      return ConstantFoldExtractValueInstruction(CAgg, IdxList);
    return nullptr;
  };

  Value *FoldInsertValue(Value *Agg, Value *Val,
                         ArrayRef<unsigned> IdxList) const override {
    auto *CAgg = dyn_cast<Constant>(Agg);
    auto *CVal = dyn_cast<Constant>(Val);
    if (CAgg && CVal)
      return ConstantFoldInsertValueInstruction(CAgg, CVal, IdxList);
    return nullptr;
  }

  Value *FoldExtractElement(Value *Vec, Value *Idx) const override {
    auto *CVec = dyn_cast<Constant>(Vec);
    auto *CIdx = dyn_cast<Constant>(Idx);
    if (CVec && CIdx)
      return ConstantExpr::getExtractElement(CVec, CIdx);
    return nullptr;
  }

  Value *FoldInsertElement(Value *Vec, Value *NewElt,
                           Value *Idx) const override {
    auto *CVec = dyn_cast<Constant>(Vec);
    auto *CNewElt = dyn_cast<Constant>(NewElt);
    auto *CIdx = dyn_cast<Constant>(Idx);
    if (CVec && CNewElt && CIdx)
      return ConstantExpr::getInsertElement(CVec, CNewElt, CIdx);
    return nullptr;
  }

  Value *FoldShuffleVector(Value *V1, Value *V2,
                           ArrayRef<int> Mask) const override {
    auto *C1 = dyn_cast<Constant>(V1);
    auto *C2 = dyn_cast<Constant>(V2);
    if (C1 && C2)
      return ConstantExpr::getShuffleVector(C1, C2, Mask);
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       Type *DestTy) const override {
    return ConstantExpr::getCast(Op, C, DestTy);
  }

  Constant *CreatePointerCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getPointerCast(C, DestTy);
  }

  Constant *CreatePointerBitCastOrAddrSpaceCast(Constant *C,
                                                Type *DestTy) const override {
    return ConstantExpr::getPointerBitCastOrAddrSpaceCast(C, DestTy);
  }

  Constant *CreateIntCast(Constant *C, Type *DestTy,
                          bool isSigned) const override {
    return ConstantExpr::getIntegerCast(C, DestTy, isSigned);
  }

  Constant *CreateFPCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getFPCast(C, DestTy);
  }

  Constant *CreateBitCast(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }

  Constant *CreateIntToPtr(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }

  Constant *CreatePtrToInt(Constant *C, Type *DestTy) const override {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }

  Constant *CreateZExtOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getZExtOrBitCast(C, DestTy);
  }

  Constant *CreateSExtOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getSExtOrBitCast(C, DestTy);
  }

  Constant *CreateTruncOrBitCast(Constant *C, Type *DestTy) const override {
    return ConstantExpr::getTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const override {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }
};

} // end namespace llvm

#endif // LLVM_IR_CONSTANTFOLDER_H
