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

#include "llvm/Analysis/CmpInstAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"

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

bool llvm::decomposeBitTestICmp(Value *LHS, Value *RHS,
                                CmpInst::Predicate &Pred,
                                Value *&X, APInt &Mask, bool LookThruTrunc) {
  using namespace PatternMatch;

  const APInt *C;
  if (!match(RHS, m_APInt(C)))
    return false;

  switch (Pred) {
  default:
    return false;
  case ICmpInst::ICMP_SLT:
    // X < 0 is equivalent to (X & SignMask) != 0.
    if (!C->isNullValue())
      return false;
    Mask = APInt::getSignMask(C->getBitWidth());
    Pred = ICmpInst::ICMP_NE;
    break;
  case ICmpInst::ICMP_SLE:
    // X <= -1 is equivalent to (X & SignMask) != 0.
    if (!C->isAllOnesValue())
      return false;
    Mask = APInt::getSignMask(C->getBitWidth());
    Pred = ICmpInst::ICMP_NE;
    break;
  case ICmpInst::ICMP_SGT:
    // X > -1 is equivalent to (X & SignMask) == 0.
    if (!C->isAllOnesValue())
      return false;
    Mask = APInt::getSignMask(C->getBitWidth());
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_SGE:
    // X >= 0 is equivalent to (X & SignMask) == 0.
    if (!C->isNullValue())
      return false;
    Mask = APInt::getSignMask(C->getBitWidth());
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_ULT:
    // X <u 2^n is equivalent to (X & ~(2^n-1)) == 0.
    if (!C->isPowerOf2())
      return false;
    Mask = -*C;
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_ULE:
    // X <=u 2^n-1 is equivalent to (X & ~(2^n-1)) == 0.
    if (!(*C + 1).isPowerOf2())
      return false;
    Mask = ~*C;
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_UGT:
    // X >u 2^n-1 is equivalent to (X & ~(2^n-1)) != 0.
    if (!(*C + 1).isPowerOf2())
      return false;
    Mask = ~*C;
    Pred = ICmpInst::ICMP_NE;
    break;
  case ICmpInst::ICMP_UGE:
    // X >=u 2^n is equivalent to (X & ~(2^n-1)) != 0.
    if (!C->isPowerOf2())
      return false;
    Mask = -*C;
    Pred = ICmpInst::ICMP_NE;
    break;
  }

  if (LookThruTrunc && match(LHS, m_Trunc(m_Value(X)))) {
    Mask = Mask.zext(X->getType()->getScalarSizeInBits());
  } else {
    X = LHS;
  }

  return true;
}
