//===- CmpInstAnalysis.cpp - Utils to help fold compares ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file holds routines to help analyse compare instructions
// and fold them into constants or other compare instructions
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CmpInstAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

unsigned llvm::getICmpCode(const ICmpInst *ICI, bool InvertPred) {
  ICmpInst::Predicate Pred = InvertPred ? ICI->getInversePredicate()
                                        : ICI->getPredicate();
  switch (Pred) {
      // False -> 0
    case ICmpInst::ICMP_UGT: return 1;  // 001
    case ICmpInst::ICMP_SGT: return 1;  // 001
    case ICmpInst::ICMP_EQ:  return 2;  // 010
    case ICmpInst::ICMP_UGE: return 3;  // 011
    case ICmpInst::ICMP_SGE: return 3;  // 011
    case ICmpInst::ICMP_ULT: return 4;  // 100
    case ICmpInst::ICMP_SLT: return 4;  // 100
    case ICmpInst::ICMP_NE:  return 5;  // 101
    case ICmpInst::ICMP_ULE: return 6;  // 110
    case ICmpInst::ICMP_SLE: return 6;  // 110
      // True -> 7
    default:
      llvm_unreachable("Invalid ICmp predicate!");
  }
}

Value *llvm::getICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                          CmpInst::Predicate &NewICmpPred) {
  switch (Code) {
    default: llvm_unreachable("Illegal ICmp code!");
    case 0: // False.
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    case 1: NewICmpPred = Sign ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT; break;
    case 2: NewICmpPred = ICmpInst::ICMP_EQ; break;
    case 3: NewICmpPred = Sign ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE; break;
    case 4: NewICmpPred = Sign ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT; break;
    case 5: NewICmpPred = ICmpInst::ICMP_NE; break;
    case 6: NewICmpPred = Sign ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE; break;
    case 7: // True.
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
  }
  return nullptr;
}

bool llvm::PredicatesFoldable(ICmpInst::Predicate p1, ICmpInst::Predicate p2) {
  return (CmpInst::isSigned(p1) == CmpInst::isSigned(p2)) ||
         (CmpInst::isSigned(p1) && ICmpInst::isEquality(p2)) ||
         (CmpInst::isSigned(p2) && ICmpInst::isEquality(p1));
}

bool llvm::decomposeBitTestICmp(const ICmpInst *I, CmpInst::Predicate &Pred,
                                Value *&X, Value *&Y, Value *&Z) {
  ConstantInt *C = dyn_cast<ConstantInt>(I->getOperand(1));
  if (!C)
    return false;

  switch (I->getPredicate()) {
  default:
    return false;
  case ICmpInst::ICMP_SLT:
    // X < 0 is equivalent to (X & SignMask) != 0.
    if (!C->isZero())
      return false;
    Y = ConstantInt::get(I->getContext(), APInt::getSignMask(C->getBitWidth()));
    Pred = ICmpInst::ICMP_NE;
    break;
  case ICmpInst::ICMP_SGT:
    // X > -1 is equivalent to (X & SignMask) == 0.
    if (!C->isAllOnesValue())
      return false;
    Y = ConstantInt::get(I->getContext(), APInt::getSignMask(C->getBitWidth()));
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_ULT:
    // X <u 2^n is equivalent to (X & ~(2^n-1)) == 0.
    if (!C->getValue().isPowerOf2())
      return false;
    Y = ConstantInt::get(I->getContext(), -C->getValue());
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_UGT:
    // X >u 2^n-1 is equivalent to (X & ~(2^n-1)) != 0.
    if (!(C->getValue() + 1).isPowerOf2())
      return false;
    Y = ConstantInt::get(I->getContext(), ~C->getValue());
    Pred = ICmpInst::ICMP_NE;
    break;
  }

  X = I->getOperand(0);
  Z = ConstantInt::getNullValue(C->getType());
  return true;
}
