//====- TargetFolder.h - Constant folding helper ---------------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants with
// target dependent folding, in addition to the same target-independent
// folding that the ConstantFolder class provides.  For general constant
// creation and folding, use ConstantExpr and the routines in
// llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETFOLDER_H
#define LLVM_ANALYSIS_TARGETFOLDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilderFolder.h"
#include "llvm/IR/Operator.h"

namespace llvm {

class Constant;
class DataLayout;
class Type;

/// TargetFolder - Create constants with target dependent folding.
class TargetFolder final : public IRBuilderFolder {
  const DataLayout &DL;

  /// Fold - Fold the constant using target specific information.
  Constant *Fold(Constant *C) const {
    return ConstantFoldConstant(C, DL);
  }

  virtual void anchor();

public:
  explicit TargetFolder(const DataLayout &DL) : DL(DL) {}

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
    if (LC && RC)
      return Fold(ConstantExpr::get(Opc, LC, RC));
    return nullptr;
  }

  Value *FoldExactBinOp(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                        bool IsExact) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC)
      return Fold(ConstantExpr::get(
          Opc, LC, RC, IsExact ? PossiblyExactOperator::IsExact : 0));
    return nullptr;
  }

  Value *FoldNoWrapBinOp(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                         bool HasNUW, bool HasNSW) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC) {
      unsigned Flags = 0;
      if (HasNUW)
        Flags |= OverflowingBinaryOperator::NoUnsignedWrap;
      if (HasNSW)
        Flags |= OverflowingBinaryOperator::NoSignedWrap;
      return Fold(ConstantExpr::get(Opc, LC, RC, Flags));
    }
    return nullptr;
  }

  Value *FoldBinOpFMF(Instruction::BinaryOps Opc, Value *LHS, Value *RHS,
                      FastMathFlags FMF) const override {
    return FoldBinOp(Opc, LHS, RHS);
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
    if (auto *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      if (any_of(IdxList, [](Value *V) { return !isa<Constant>(V); }))
        return nullptr;
      if (IsInBounds)
        return Fold(ConstantExpr::getInBoundsGetElementPtr(Ty, PC, IdxList));
      else
        return Fold(ConstantExpr::getGetElementPtr(Ty, PC, IdxList));
    }
    return nullptr;
  }

  Value *FoldSelect(Value *C, Value *True, Value *False) const override {
    auto *CC = dyn_cast<Constant>(C);
    auto *TC = dyn_cast<Constant>(True);
    auto *FC = dyn_cast<Constant>(False);
    if (CC && TC && FC)
      return Fold(ConstantExpr::getSelect(CC, TC, FC));

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
      return Fold(ConstantExpr::getInsertValue(CAgg, CVal, IdxList));
    return nullptr;
  }

  Value *FoldExtractElement(Value *Vec, Value *Idx) const override {
    auto *CVec = dyn_cast<Constant>(Vec);
    auto *CIdx = dyn_cast<Constant>(Idx);
    if (CVec && CIdx)
      return Fold(ConstantExpr::getExtractElement(CVec, CIdx));
    return nullptr;
  }

  Value *FoldInsertElement(Value *Vec, Value *NewElt,
                           Value *Idx) const override {
    auto *CVec = dyn_cast<Constant>(Vec);
    auto *CNewElt = dyn_cast<Constant>(NewElt);
    auto *CIdx = dyn_cast<Constant>(Idx);
    if (CVec && CNewElt && CIdx)
      return Fold(ConstantExpr::getInsertElement(CVec, CNewElt, CIdx));
    return nullptr;
  }

  Value *FoldShuffleVector(Value *V1, Value *V2,
                           ArrayRef<int> Mask) const override {
    auto *C1 = dyn_cast<Constant>(V1);
    auto *C2 = dyn_cast<Constant>(V2);
    if (C1 && C2)
      return Fold(ConstantExpr::getShuffleVector(C1, C2, Mask));
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return Fold(ConstantExpr::getNeg(C, HasNUW, HasNSW));
  }
  Constant *CreateFNeg(Constant *C) const override {
    return Fold(ConstantExpr::getFNeg(C));
  }
  Constant *CreateNot(Constant *C) const override {
    return Fold(ConstantExpr::getNot(C));
  }

  Constant *CreateUnOp(Instruction::UnaryOps Opc, Constant *C) const override {
    return Fold(ConstantExpr::get(Opc, C));
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getCast(Op, C, DestTy));
  }
  Constant *CreateIntCast(Constant *C, Type *DestTy,
                          bool isSigned) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getIntegerCast(C, DestTy, isSigned));
  }
  Constant *CreatePointerCast(Constant *C, Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getPointerCast(C, DestTy));
  }
  Constant *CreateFPCast(Constant *C, Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getFPCast(C, DestTy));
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
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getZExtOrBitCast(C, DestTy));
  }
  Constant *CreateSExtOrBitCast(Constant *C, Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getSExtOrBitCast(C, DestTy));
  }
  Constant *CreateTruncOrBitCast(Constant *C, Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getTruncOrBitCast(C, DestTy));
  }

  Constant *CreatePointerBitCastOrAddrSpaceCast(Constant *C,
                                                Type *DestTy) const override {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getPointerBitCastOrAddrSpaceCast(C, DestTy));
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const override {
    return Fold(ConstantExpr::getCompare(P, LHS, RHS));
  }
};

}

#endif
