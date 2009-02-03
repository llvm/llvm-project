//======-- llvm/Support/NoFolder.h - Constant folding helper -*- C++ -*-======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// class returns values rather than constants.  The values do not have names,
// even if names were provided to IRBuilder, which may be confusing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NOFOLDER_H
#define LLVM_SUPPORT_NOFOLDER_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"

namespace llvm {

/// NoFolder - Create "constants" (actually, values) with no folding.
class NoFolder {
public:

  //===--------------------------------------------------------------------===//
  // Binary Operators
  //===--------------------------------------------------------------------===//

  Value *CreateAdd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAdd(LHS, RHS);
  }
  Value *CreateSub(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSub(LHS, RHS);
  }
  Value *CreateMul(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateMul(LHS, RHS);
  }
  Value *CreateUDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateUDiv(LHS, RHS);
  }
  Value *CreateSDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSDiv(LHS, RHS);
  }
  Value *CreateFDiv(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFDiv(LHS, RHS);
  }
  Value *CreateURem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateURem(LHS, RHS);
  }
  Value *CreateSRem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateSRem(LHS, RHS);
  }
  Value *CreateFRem(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateFRem(LHS, RHS);
  }
  Value *CreateShl(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateShl(LHS, RHS);
  }
  Value *CreateLShr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateLShr(LHS, RHS);
  }
  Value *CreateAShr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAShr(LHS, RHS);
  }
  Value *CreateAnd(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateAnd(LHS, RHS);
  }
  Value *CreateOr(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateOr(LHS, RHS);
  }
  Value *CreateXor(Constant *LHS, Constant *RHS) const {
    return BinaryOperator::CreateXor(LHS, RHS);
  }

  Value *CreateBinOp(Instruction::BinaryOps Opc,
                     Constant *LHS, Constant *RHS) const {
    return BinaryOperator::Create(Opc, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Unary Operators
  //===--------------------------------------------------------------------===//

  Value *CreateNeg(Constant *C) const {
    return BinaryOperator::CreateNeg(C);
  }
  Value *CreateNot(Constant *C) const {
    return BinaryOperator::CreateNot(C);
  }

  //===--------------------------------------------------------------------===//
  // Memory Instructions
  //===--------------------------------------------------------------------===//

  Constant *CreateGetElementPtr(Constant *C, Constant* const *IdxList,
                                unsigned NumIdx) const {
    return ConstantExpr::getGetElementPtr(C, IdxList, NumIdx);
  }
  Value *CreateGetElementPtr(Constant *C, Value* const *IdxList,
                             unsigned NumIdx) const {
    return GetElementPtrInst::Create(C, IdxList, IdxList+NumIdx);
  }

  //===--------------------------------------------------------------------===//
  // Cast/Conversion Operators
  //===--------------------------------------------------------------------===//

  Value *CreateCast(Instruction::CastOps Op, Constant *C,
                    const Type *DestTy) const {
    return CastInst::Create(Op, C, DestTy);
  }
  Value *CreateIntCast(Constant *C, const Type *DestTy,
                       bool isSigned) const {
    return CastInst::CreateIntegerCast(C, DestTy, isSigned);
  }

  //===--------------------------------------------------------------------===//
  // Compare Instructions
  //===--------------------------------------------------------------------===//

  Value *CreateICmp(CmpInst::Predicate P, Constant *LHS, Constant *RHS) const {
    return new ICmpInst(P, LHS, RHS);
  }
  Value *CreateFCmp(CmpInst::Predicate P, Constant *LHS, Constant *RHS) const {
    return new FCmpInst(P, LHS, RHS);
  }
  Value *CreateVICmp(CmpInst::Predicate P, Constant *LHS, Constant *RHS) const {
    return new VICmpInst(P, LHS, RHS);
  }
  Value *CreateVFCmp(CmpInst::Predicate P, Constant *LHS, Constant *RHS) const {
    return new VFCmpInst(P, LHS, RHS);
  }

  //===--------------------------------------------------------------------===//
  // Other Instructions
  //===--------------------------------------------------------------------===//

  Value *CreateSelect(Constant *C, Constant *True, Constant *False) const {
    return SelectInst::Create(C, True, False);
  }

  Value *CreateExtractElement(Constant *Vec, Constant *Idx) const {
    return new ExtractElementInst(Vec, Idx);
  }

  Value *CreateInsertElement(Constant *Vec, Constant *NewElt,
                             Constant *Idx) const {
    return InsertElementInst::Create(Vec, NewElt, Idx);
  }

  Value *CreateShuffleVector(Constant *V1, Constant *V2, Constant *Mask) const {
    return new ShuffleVectorInst(V1, V2, Mask);
  }

  Value *CreateExtractValue(Constant *Agg, const unsigned *IdxList,
                            unsigned NumIdx) const {
    return ExtractValueInst::Create(Agg, IdxList, IdxList+NumIdx);
  }

  Value *CreateInsertValue(Constant *Agg, Constant *Val,
                           const unsigned *IdxList, unsigned NumIdx) const {
    return InsertValueInst::Create(Agg, Val, IdxList, IdxList+NumIdx);
  }
};

}

#endif
