//====-- llvm/Support/TargetFolder.h - Constant folding helper -*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants with
// target dependent folding.  For general constant creation and folding,
// use ConstantExpr and the routines in llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETFOLDER_H
#define LLVM_SUPPORT_TARGETFOLDER_H

#include "llvm/Constants.h"
#include "llvm/Analysis/ConstantFolding.h"

namespace llvm {

class TargetData;

/// TargetFolder - Create constants with target dependent folding.
class TargetFolder {
  const TargetData &TD;

  /// Fold - Fold the constant using target specific information.
  Constant *Fold(Constant *C) const {
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
      if (Constant *CF = ConstantFoldConstantExpression(CE, &TD))
        return CF;
    return C;
  }

public:
  TargetFolder(const TargetData &TheTD) : TD(TheTD) {}

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateAdd(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getAdd(LHS, RHS));
  }
  Constant *CreateSub(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getSub(LHS, RHS));
  }
  Constant *CreateMul(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getMul(LHS, RHS));
  }
  Constant *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getUDiv(LHS, RHS));
  }
  Constant *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getSDiv(LHS, RHS));
  }
  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getFDiv(LHS, RHS));
  }
  Constant *CreateURem(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getURem(LHS, RHS));
  }
  Constant *CreateSRem(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getSRem(LHS, RHS));
  }
  Constant *CreateFRem(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getFRem(LHS, RHS));
  }
  Constant *CreateShl(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getShl(LHS, RHS));
  }
  Constant *CreateLShr(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getLShr(LHS, RHS));
  }
  Constant *CreateAShr(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getAShr(LHS, RHS));
  }
  Constant *CreateAnd(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getAnd(LHS, RHS));
  }
  Constant *CreateOr(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getOr(LHS, RHS));
  }
  Constant *CreateXor(Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::getXor(LHS, RHS));
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const {
    return Fold(ConstantExpr::get(Opc, LHS, RHS));
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C) const {
    return Fold(ConstantExpr::getNeg(C));
  }
  Constant *CreateNot(Constant *C) const {
    return Fold(ConstantExpr::getNot(C));
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return Fold(ConstantExpr::getGetElementPtr(C, IdxList, NumIdx));
  }
  Constant *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                                unsigned NumIdx) const {
    return Fold(ConstantExpr::getGetElementPtr(C, IdxList, NumIdx));
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       const Type *DestTy) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getCast(Op, C, DestTy));
  }
  Constant *CreateIntCast(Constant *C, const Type *DestTy,
                          bool isSigned) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getIntegerCast(C, DestTy, isSigned));
  }

  Constant *CreateBitCast(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::BitCast, C, DestTy);
  }
  Constant *CreateIntToPtr(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::IntToPtr, C, DestTy);
  }
  Constant *CreatePtrToInt(Constant *C, const Type *DestTy) const {
    return CreateCast(Instruction::PtrToInt, C, DestTy);
  }
  Constant *CreateTruncOrBitCast(Constant *C, const Type *DestTy) const {
    if (C->getType() == DestTy)
      return C; // avoid calling Fold
    return Fold(ConstantExpr::getTruncOrBitCast(C, DestTy));
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateICmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Fold(ConstantExpr::getCompare(P, LHS, RHS));
  }
  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return Fold(ConstantExpr::getCompare(P, LHS, RHS));
  }
  Constant *CreateVICmp(CmpInst::Predicate P, Constant *LHS,
                        Constant *RHS) const {
    return Fold(ConstantExpr::getCompare(P, LHS, RHS));
  }
  Constant *CreateVFCmp(CmpInst::Predicate P, Constant *LHS,
                        Constant *RHS) const {
    return Fold(ConstantExpr::getCompare(P, LHS, RHS));
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateSelect(Constant *C, Constant *True, Constant *False) const {
    return Fold(ConstantExpr::getSelect(C, True, False));
  }

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return Fold(ConstantExpr::getExtractElement(Vec, Idx));
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const {
    return Fold(ConstantExpr::getInsertElement(Vec, NewElt, Idx));
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                Constant *Mask) const {
    return Fold(ConstantExpr::getShuffleVector(V1, V2, Mask));
  }

  Constant *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                               unsigned NumIdx) const {
    return Fold(ConstantExpr::getExtractValue(Agg, IdxList, NumIdx));
  }

  Constant *CreateInsertValue(Constant *Agg, Constant *Val,
                              const unsigned *IdxList, unsigned NumIdx) const {
    return Fold(ConstantExpr::getInsertValue(Agg, Val, IdxList, NumIdx));
  }
};

}

#endif
