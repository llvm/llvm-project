//===-- llvm/Support/ConstantFolder.h - Constant folding helper -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstantFolder class, a helper for IRBuilder.
// It provides IRBuilder with a set of methods for creating constants
// with minimal folding.  For general constant creation and folding,
// use ConstantExpr and the routines in llvm/Analysis/ConstantFolding.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CONSTANTFOLDER_H
#define LLVM_SUPPORT_CONSTANTFOLDER_H

#include "llvm/Constants.h"

namespace llvm {

/// ConstantFolder - Create constants with minimum, target independent, folding.
class ConstantFolder {
public:

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateAdd(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getAdd(LHS, RHS);
  }
  Constant *CreateSub(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getSub(LHS, RHS);
  }
  Constant *CreateMul(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getMul(LHS, RHS);
  }
  Constant *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getUDiv(LHS, RHS);
  }
  Constant *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getSDiv(LHS, RHS);
  }
  Constant *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getFDiv(LHS, RHS);
  }
  Constant *CreateURem(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getURem(LHS, RHS);
  }
  Constant *CreateSRem(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getSRem(LHS, RHS);
  }
  Constant *CreateFRem(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getFRem(LHS, RHS);
  }
  Constant *CreateShl(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getShl(LHS, RHS);
  }
  Constant *CreateLShr(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getLShr(LHS, RHS);
  }
  Constant *CreateAShr(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getAShr(LHS, RHS);
  }
  Constant *CreateAnd(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getAnd(LHS, RHS);
  }
  Constant *CreateOr(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getOr(LHS, RHS);
  }
  Constant *CreateXor(Constant *LHS, Constant *RHS) const {
    return ConstantExpr::getXor(LHS, RHS);
  }

  Constant *CreateBinOp(Instruction::BinaryOps Opc,
                        Constant *LHS, Constant *RHS) const {
    return ConstantExpr::get(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateNeg(Constant *C) const {
    return ConstantExpr::getNeg(C);
  }
  Constant *CreateNot(Constant *C) const {
    return ConstantExpr::getNot(C);
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
  }
  Constant *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                                unsigned NumIdx) const {
    return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Constant *CreateCast(Instruction::CastOps Op, Constant *C,
                       const Type *DestTy) const {
    return ConstantExpr::getCast(Op, C, DestTy);
  }
  Constant *CreateIntCast(Constant *C, const Type *DestTy,
                          bool isSigned) const {
    return ConstantExpr::getIntegerCast(C, DestTy, isSigned);
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
    return ConstantExpr::getTruncOrBitCast(C, DestTy);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateICmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }
  Constant *CreateFCmp(CmpInst::Predicate P, Constant *LHS,
                       Constant *RHS) const {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }
  Constant *CreateVICmp(CmpInst::Predicate P, Constant *LHS,
                        Constant *RHS) const {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }
  Constant *CreateVFCmp(CmpInst::Predicate P, Constant *LHS,
                        Constant *RHS) const {
    return ConstantExpr::getCompare(P, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateSelect(Constant *C, Constant *True, Constant *False) const {
    return ConstantExpr::getSelect(C, True, False);
  }

  Constant *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return ConstantExpr::getExtractElement(Vec, Idx);
  }

  Constant *CreateInsertElement(Constant *Vec, Constant *NewElt,
                                Constant *Idx) const {
    return ConstantExpr::getInsertElement(Vec, NewElt, Idx);
  }

  Constant *CreateShuffleVector(Constant *V1, Constant *V2,
                                Constant *Mask) const {
    return ConstantExpr::getShuffleVector(V1, V2, Mask);
  }

  Constant *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                               unsigned NumIdx) const {
    return ConstantExpr::getExtractValue(Agg, IdxList, NumIdx);
  }

  Constant *CreateInsertValue(Constant *Agg, Constant *Val,
                              const unsigned *IdxList, unsigned NumIdx) const {
    return ConstantExpr::getInsertValue(Agg, Val, IdxList, NumIdx);
  }
};

}

#endif
