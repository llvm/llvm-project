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
  Value *FoldAdd(Value *LHS, Value *RHS, bool HasNUW = false,
                 bool HasNSW = false) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC)
      return Fold(ConstantExpr::getAdd(LC, RC, HasNUW, HasNSW));
    return nullptr;
  }

  Value *FoldAnd(Value *LHS, Value *RHS) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC)
      return Fold(ConstantExpr::getAnd(LC, RC));
    return nullptr;
  }

  Value *FoldOr(Value *LHS, Value *RHS) const override {
    auto *LC = dyn_cast<Constant>(LHS);
    auto *RC = dyn_cast<Constant>(RHS);
    if (LC && RC)
      return Fold(ConstantExpr::getOr(LC, RC));
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
      return Fold(ConstantExpr::getExtractValue(CAgg, IdxList));
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

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateFAdd(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getFAdd(LHS, RHS));
  }
  Constant *CreateSub(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return Fold(ConstantExpr::getSub(LHS, RHS, HasNUW, HasNSW));
  }
  Constant *CreateFSub(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getFSub(LHS, RHS));
  }
  Constant *CreateMul(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return Fold(ConstantExpr::getMul(LHS, RHS, HasNUW, HasNSW));
  }
  Constant *CreateFMul(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getFMul(LHS, RHS));
  }
  Constant *CreateUDiv(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return Fold(ConstantExpr::getUDiv(LHS, RHS, isExact));
  }
  Constant *CreateSDiv(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return Fold(ConstantExpr::getSDiv(LHS, RHS, isExact));
  }
  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getFDiv(LHS, RHS));
  }
  Constant *CreateURem(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getURem(LHS, RHS));
  }
  Constant *CreateSRem(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getSRem(LHS, RHS));
  }
  Constant *CreateFRem(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getFRem(LHS, RHS));
  }
  Constant *CreateShl(Constant *LHS, Constant *RHS,
                      bool HasNUW = false, bool HasNSW = false) const override {
    return Fold(ConstantExpr::getShl(LHS, RHS, HasNUW, HasNSW));
  }
  Constant *CreateLShr(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return Fold(ConstantExpr::getLShr(LHS, RHS, isExact));
  }
  Constant *CreateAShr(Constant *LHS, Constant *RHS,
                       bool isExact = false) const override {
    return Fold(ConstantExpr::getAShr(LHS, RHS, isExact));
  }
  Constant *CreateXor(Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::getXor(LHS, RHS));
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const override {
    return Fold(ConstantExpr::get(Opc, LHS, RHS));
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

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const override {
    return Fold(ConstantExpr::getExtractElement(Vec, Idx));
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const override {
    return Fold(ConstantExpr::getInsertElement(Vec, NewElt, Idx));
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                ArrayRef<int> Mask) const override {
    return Fold(ConstantExpr::getShuffleVector(V1, V2, Mask));
  }
};

}

#endif
