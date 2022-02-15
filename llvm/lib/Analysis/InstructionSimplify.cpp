//===- InstructionSimplify.cpp - Fold instruction operands ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  All operands are assumed to have already been
// simplified: This is usually true and assuming it simplifies the logic (if
// they have not been simplified then results are correct but maybe suboptimal).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionSimplify.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/CmpInstAnalysis.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OverflowInstAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "instsimplify"

enum { RecursionLimit = 3 };

STATISTIC(NumExpand,  "Number of expansions");
STATISTIC(NumReassoc, "Number of reassociations");

static Value *SimplifyAndInst(Value *, Value *, const SimplifyQuery &, unsigned);
static Value *simplifyUnOp(unsigned, Value *, const SimplifyQuery &, unsigned);
static Value *simplifyFPUnOp(unsigned, Value *, const FastMathFlags &,
                             const SimplifyQuery &, unsigned);
static Value *SimplifyBinOp(unsigned, Value *, Value *, const SimplifyQuery &,
                            unsigned);
static Value *SimplifyBinOp(unsigned, Value *, Value *, const FastMathFlags &,
                            const SimplifyQuery &, unsigned);
static Value *SimplifyCmpInst(unsigned, Value *, Value *, const SimplifyQuery &,
                              unsigned);
static Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               const SimplifyQuery &Q, unsigned MaxRecurse);
static Value *SimplifyOrInst(Value *, Value *, const SimplifyQuery &, unsigned);
static Value *SimplifyXorInst(Value *, Value *, const SimplifyQuery &, unsigned);
static Value *SimplifyCastInst(unsigned, Value *, Type *,
                               const SimplifyQuery &, unsigned);
static Value *SimplifyGEPInst(Type *, Value *, ArrayRef<Value *>, bool,
                              const SimplifyQuery &, unsigned);
static Value *SimplifySelectInst(Value *, Value *, Value *,
                                 const SimplifyQuery &, unsigned);

static Value *foldSelectWithBinaryOp(Value *Cond, Value *TrueVal,
                                     Value *FalseVal) {
  BinaryOperator::BinaryOps BinOpCode;
  if (auto *BO = dyn_cast<BinaryOperator>(Cond))
    BinOpCode = BO->getOpcode();
  else
    return nullptr;

  CmpInst::Predicate ExpectedPred, Pred1, Pred2;
  if (BinOpCode == BinaryOperator::Or) {
    ExpectedPred = ICmpInst::ICMP_NE;
  } else if (BinOpCode == BinaryOperator::And) {
    ExpectedPred = ICmpInst::ICMP_EQ;
  } else
    return nullptr;

  // %A = icmp eq %TV, %FV
  // %B = icmp eq %X, %Y (and one of these is a select operand)
  // %C = and %A, %B
  // %D = select %C, %TV, %FV
  // -->
  // %FV

  // %A = icmp ne %TV, %FV
  // %B = icmp ne %X, %Y (and one of these is a select operand)
  // %C = or %A, %B
  // %D = select %C, %TV, %FV
  // -->
  // %TV
  Value *X, *Y;
  if (!match(Cond, m_c_BinOp(m_c_ICmp(Pred1, m_Specific(TrueVal),
                                      m_Specific(FalseVal)),
                             m_ICmp(Pred2, m_Value(X), m_Value(Y)))) ||
      Pred1 != Pred2 || Pred1 != ExpectedPred)
    return nullptr;

  if (X == TrueVal || X == FalseVal || Y == TrueVal || Y == FalseVal)
    return BinOpCode == BinaryOperator::Or ? TrueVal : FalseVal;

  return nullptr;
}

/// For a boolean type or a vector of boolean type, return false or a vector
/// with every element false.
static Constant *getFalse(Type *Ty) {
  return ConstantInt::getFalse(Ty);
}

/// For a boolean type or a vector of boolean type, return true or a vector
/// with every element true.
static Constant *getTrue(Type *Ty) {
  return ConstantInt::getTrue(Ty);
}

/// isSameCompare - Is V equivalent to the comparison "LHS Pred RHS"?
static bool isSameCompare(Value *V, CmpInst::Predicate Pred, Value *LHS,
                          Value *RHS) {
  CmpInst *Cmp = dyn_cast<CmpInst>(V);
  if (!Cmp)
    return false;
  CmpInst::Predicate CPred = Cmp->getPredicate();
  Value *CLHS = Cmp->getOperand(0), *CRHS = Cmp->getOperand(1);
  if (CPred == Pred && CLHS == LHS && CRHS == RHS)
    return true;
  return CPred == CmpInst::getSwappedPredicate(Pred) && CLHS == RHS &&
    CRHS == LHS;
}

/// Simplify comparison with true or false branch of select:
///  %sel = select i1 %cond, i32 %tv, i32 %fv
///  %cmp = icmp sle i32 %sel, %rhs
/// Compose new comparison by substituting %sel with either %tv or %fv
/// and see if it simplifies.
static Value *simplifyCmpSelCase(CmpInst::Predicate Pred, Value *LHS,
                                 Value *RHS, Value *Cond,
                                 const SimplifyQuery &Q, unsigned MaxRecurse,
                                 Constant *TrueOrFalse) {
  Value *SimplifiedCmp = SimplifyCmpInst(Pred, LHS, RHS, Q, MaxRecurse);
  if (SimplifiedCmp == Cond) {
    // %cmp simplified to the select condition (%cond).
    return TrueOrFalse;
  } else if (!SimplifiedCmp && isSameCompare(Cond, Pred, LHS, RHS)) {
    // It didn't simplify. However, if composed comparison is equivalent
    // to the select condition (%cond) then we can replace it.
    return TrueOrFalse;
  }
  return SimplifiedCmp;
}

/// Simplify comparison with true branch of select
static Value *simplifyCmpSelTrueCase(CmpInst::Predicate Pred, Value *LHS,
                                     Value *RHS, Value *Cond,
                                     const SimplifyQuery &Q,
                                     unsigned MaxRecurse) {
  return simplifyCmpSelCase(Pred, LHS, RHS, Cond, Q, MaxRecurse,
                            getTrue(Cond->getType()));
}

/// Simplify comparison with false branch of select
static Value *simplifyCmpSelFalseCase(CmpInst::Predicate Pred, Value *LHS,
                                      Value *RHS, Value *Cond,
                                      const SimplifyQuery &Q,
                                      unsigned MaxRecurse) {
  return simplifyCmpSelCase(Pred, LHS, RHS, Cond, Q, MaxRecurse,
                            getFalse(Cond->getType()));
}

/// We know comparison with both branches of select can be simplified, but they
/// are not equal. This routine handles some logical simplifications.
static Value *handleOtherCmpSelSimplifications(Value *TCmp, Value *FCmp,
                                               Value *Cond,
                                               const SimplifyQuery &Q,
                                               unsigned MaxRecurse) {
  // If the false value simplified to false, then the result of the compare
  // is equal to "Cond && TCmp".  This also catches the case when the false
  // value simplified to false and the true value to true, returning "Cond".
  // Folding select to and/or isn't poison-safe in general; impliesPoison
  // checks whether folding it does not convert a well-defined value into
  // poison.
  if (match(FCmp, m_Zero()) && impliesPoison(TCmp, Cond))
    if (Value *V = SimplifyAndInst(Cond, TCmp, Q, MaxRecurse))
      return V;
  // If the true value simplified to true, then the result of the compare
  // is equal to "Cond || FCmp".
  if (match(TCmp, m_One()) && impliesPoison(FCmp, Cond))
    if (Value *V = SimplifyOrInst(Cond, FCmp, Q, MaxRecurse))
      return V;
  // Finally, if the false value simplified to true and the true value to
  // false, then the result of the compare is equal to "!Cond".
  if (match(FCmp, m_One()) && match(TCmp, m_Zero()))
    if (Value *V = SimplifyXorInst(
            Cond, Constant::getAllOnesValue(Cond->getType()), Q, MaxRecurse))
      return V;
  return nullptr;
}

/// Does the given value dominate the specified phi node?
static bool valueDominatesPHI(Value *V, PHINode *P, const DominatorTree *DT) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    // Arguments and constants dominate all instructions.
    return true;

  // If we are processing instructions (and/or basic blocks) that have not been
  // fully added to a function, the parent nodes may still be null. Simply
  // return the conservative answer in these cases.
  if (!I->getParent() || !P->getParent() || !I->getFunction())
    return false;

  // If we have a DominatorTree then do a precise test.
  if (DT)
    return DT->dominates(I, P);

  // Otherwise, if the instruction is in the entry block and is not an invoke,
  // then it obviously dominates all phi nodes.
  if (I->getParent()->isEntryBlock() && !isa<InvokeInst>(I) &&
      !isa<CallBrInst>(I))
    return true;

  return false;
}

/// Try to simplify a binary operator of form "V op OtherOp" where V is
/// "(B0 opex B1)" by distributing 'op' across 'opex' as
/// "(B0 op OtherOp) opex (B1 op OtherOp)".
static Value *expandBinOp(Instruction::BinaryOps Opcode, Value *V,
                          Value *OtherOp, Instruction::BinaryOps OpcodeToExpand,
                          const SimplifyQuery &Q, unsigned MaxRecurse) {
  auto *B = dyn_cast<BinaryOperator>(V);
  if (!B || B->getOpcode() != OpcodeToExpand)
    return nullptr;
  Value *B0 = B->getOperand(0), *B1 = B->getOperand(1);
  Value *L = SimplifyBinOp(Opcode, B0, OtherOp, Q.getWithoutUndef(),
                           MaxRecurse);
  if (!L)
    return nullptr;
  Value *R = SimplifyBinOp(Opcode, B1, OtherOp, Q.getWithoutUndef(),
                           MaxRecurse);
  if (!R)
    return nullptr;

  // Does the expanded pair of binops simplify to the existing binop?
  if ((L == B0 && R == B1) ||
      (Instruction::isCommutative(OpcodeToExpand) && L == B1 && R == B0)) {
    ++NumExpand;
    return B;
  }

  // Otherwise, return "L op' R" if it simplifies.
  Value *S = SimplifyBinOp(OpcodeToExpand, L, R, Q, MaxRecurse);
  if (!S)
    return nullptr;

  ++NumExpand;
  return S;
}

/// Try to simplify binops of form "A op (B op' C)" or the commuted variant by
/// distributing op over op'.
static Value *expandCommutativeBinOp(Instruction::BinaryOps Opcode,
                                     Value *L, Value *R,
                                     Instruction::BinaryOps OpcodeToExpand,
                                     const SimplifyQuery &Q,
                                     unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  if (Value *V = expandBinOp(Opcode, L, R, OpcodeToExpand, Q, MaxRecurse))
    return V;
  if (Value *V = expandBinOp(Opcode, R, L, OpcodeToExpand, Q, MaxRecurse))
    return V;
  return nullptr;
}

/// Generic simplifications for associative binary operations.
/// Returns the simpler value, or null if none was found.
static Value *SimplifyAssociativeBinOp(Instruction::BinaryOps Opcode,
                                       Value *LHS, Value *RHS,
                                       const SimplifyQuery &Q,
                                       unsigned MaxRecurse) {
  assert(Instruction::isAssociative(Opcode) && "Not an associative operation!");

  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);

  // Transform: "(A op B) op C" ==> "A op (B op C)" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "B op C" simplify?
    if (Value *V = SimplifyBinOp(Opcode, B, C, Q, MaxRecurse)) {
      // It does!  Return "A op V" if it simplifies or is already available.
      // If V equals B then "A op V" is just the LHS.
      if (V == B) return LHS;
      // Otherwise return "A op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, A, V, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "(A op B) op C" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "A op B" simplify?
    if (Value *V = SimplifyBinOp(Opcode, A, B, Q, MaxRecurse)) {
      // It does!  Return "V op C" if it simplifies or is already available.
      // If V equals B then "V op C" is just the RHS.
      if (V == B) return RHS;
      // Otherwise return "V op C" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, C, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // The remaining transforms require commutativity as well as associativity.
  if (!Instruction::isCommutative(Opcode))
    return nullptr;

  // Transform: "(A op B) op C" ==> "(C op A) op B" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, Q, MaxRecurse)) {
      // It does!  Return "V op B" if it simplifies or is already available.
      // If V equals A then "V op B" is just the LHS.
      if (V == A) return LHS;
      // Otherwise return "V op B" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, B, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "B op (C op A)" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, Q, MaxRecurse)) {
      // It does!  Return "B op V" if it simplifies or is already available.
      // If V equals C then "B op V" is just the RHS.
      if (V == C) return RHS;
      // Otherwise return "B op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, B, V, Q, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  return nullptr;
}

/// In the case of a binary operation with a select instruction as an operand,
/// try to simplify the binop by seeing whether evaluating it on both branches
/// of the select results in the same value. Returns the common value if so,
/// otherwise returns null.
static Value *ThreadBinOpOverSelect(Instruction::BinaryOps Opcode, Value *LHS,
                                    Value *RHS, const SimplifyQuery &Q,
                                    unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  SelectInst *SI;
  if (isa<SelectInst>(LHS)) {
    SI = cast<SelectInst>(LHS);
  } else {
    assert(isa<SelectInst>(RHS) && "No select instruction operand!");
    SI = cast<SelectInst>(RHS);
  }

  // Evaluate the BinOp on the true and false branches of the select.
  Value *TV;
  Value *FV;
  if (SI == LHS) {
    TV = SimplifyBinOp(Opcode, SI->getTrueValue(), RHS, Q, MaxRecurse);
    FV = SimplifyBinOp(Opcode, SI->getFalseValue(), RHS, Q, MaxRecurse);
  } else {
    TV = SimplifyBinOp(Opcode, LHS, SI->getTrueValue(), Q, MaxRecurse);
    FV = SimplifyBinOp(Opcode, LHS, SI->getFalseValue(), Q, MaxRecurse);
  }

  // If they simplified to the same value, then return the common value.
  // If they both failed to simplify then return null.
  if (TV == FV)
    return TV;

  // If one branch simplified to undef, return the other one.
  if (TV && Q.isUndefValue(TV))
    return FV;
  if (FV && Q.isUndefValue(FV))
    return TV;

  // If applying the operation did not change the true and false select values,
  // then the result of the binop is the select itself.
  if (TV == SI->getTrueValue() && FV == SI->getFalseValue())
    return SI;

  // If one branch simplified and the other did not, and the simplified
  // value is equal to the unsimplified one, return the simplified value.
  // For example, select (cond, X, X & Z) & Z -> X & Z.
  if ((FV && !TV) || (TV && !FV)) {
    // Check that the simplified value has the form "X op Y" where "op" is the
    // same as the original operation.
    Instruction *Simplified = dyn_cast<Instruction>(FV ? FV : TV);
    if (Simplified && Simplified->getOpcode() == unsigned(Opcode)) {
      // The value that didn't simplify is "UnsimplifiedLHS op UnsimplifiedRHS".
      // We already know that "op" is the same as for the simplified value.  See
      // if the operands match too.  If so, return the simplified value.
      Value *UnsimplifiedBranch = FV ? SI->getTrueValue() : SI->getFalseValue();
      Value *UnsimplifiedLHS = SI == LHS ? UnsimplifiedBranch : LHS;
      Value *UnsimplifiedRHS = SI == LHS ? RHS : UnsimplifiedBranch;
      if (Simplified->getOperand(0) == UnsimplifiedLHS &&
          Simplified->getOperand(1) == UnsimplifiedRHS)
        return Simplified;
      if (Simplified->isCommutative() &&
          Simplified->getOperand(1) == UnsimplifiedLHS &&
          Simplified->getOperand(0) == UnsimplifiedRHS)
        return Simplified;
    }
  }

  return nullptr;
}

/// In the case of a comparison with a select instruction, try to simplify the
/// comparison by seeing whether both branches of the select result in the same
/// value. Returns the common value if so, otherwise returns null.
/// For example, if we have:
///  %tmp = select i1 %cmp, i32 1, i32 2
///  %cmp1 = icmp sle i32 %tmp, 3
/// We can simplify %cmp1 to true, because both branches of select are
/// less than 3. We compose new comparison by substituting %tmp with both
/// branches of select and see if it can be simplified.
static Value *ThreadCmpOverSelect(CmpInst::Predicate Pred, Value *LHS,
                                  Value *RHS, const SimplifyQuery &Q,
                                  unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  // Make sure the select is on the LHS.
  if (!isa<SelectInst>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<SelectInst>(LHS) && "Not comparing with a select instruction!");
  SelectInst *SI = cast<SelectInst>(LHS);
  Value *Cond = SI->getCondition();
  Value *TV = SI->getTrueValue();
  Value *FV = SI->getFalseValue();

  // Now that we have "cmp select(Cond, TV, FV), RHS", analyse it.
  // Does "cmp TV, RHS" simplify?
  Value *TCmp = simplifyCmpSelTrueCase(Pred, TV, RHS, Cond, Q, MaxRecurse);
  if (!TCmp)
    return nullptr;

  // Does "cmp FV, RHS" simplify?
  Value *FCmp = simplifyCmpSelFalseCase(Pred, FV, RHS, Cond, Q, MaxRecurse);
  if (!FCmp)
    return nullptr;

  // If both sides simplified to the same value, then use it as the result of
  // the original comparison.
  if (TCmp == FCmp)
    return TCmp;

  // The remaining cases only make sense if the select condition has the same
  // type as the result of the comparison, so bail out if this is not so.
  if (Cond->getType()->isVectorTy() == RHS->getType()->isVectorTy())
    return handleOtherCmpSelSimplifications(TCmp, FCmp, Cond, Q, MaxRecurse);

  return nullptr;
}

/// In the case of a binary operation with an operand that is a PHI instruction,
/// try to simplify the binop by seeing whether evaluating it on the incoming
/// phi values yields the same result for every value. If so returns the common
/// value, otherwise returns null.
static Value *ThreadBinOpOverPHI(Instruction::BinaryOps Opcode, Value *LHS,
                                 Value *RHS, const SimplifyQuery &Q,
                                 unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  PHINode *PI;
  if (isa<PHINode>(LHS)) {
    PI = cast<PHINode>(LHS);
    // Bail out if RHS and the phi may be mutually interdependent due to a loop.
    if (!valueDominatesPHI(RHS, PI, Q.DT))
      return nullptr;
  } else {
    assert(isa<PHINode>(RHS) && "No PHI instruction operand!");
    PI = cast<PHINode>(RHS);
    // Bail out if LHS and the phi may be mutually interdependent due to a loop.
    if (!valueDominatesPHI(LHS, PI, Q.DT))
      return nullptr;
  }

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = nullptr;
  for (Value *Incoming : PI->incoming_values()) {
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    Value *V = PI == LHS ?
      SimplifyBinOp(Opcode, Incoming, RHS, Q, MaxRecurse) :
      SimplifyBinOp(Opcode, LHS, Incoming, Q, MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return nullptr;
    CommonValue = V;
  }

  return CommonValue;
}

/// In the case of a comparison with a PHI instruction, try to simplify the
/// comparison by seeing whether comparing with all of the incoming phi values
/// yields the same result every time. If so returns the common result,
/// otherwise returns null.
static Value *ThreadCmpOverPHI(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                               const SimplifyQuery &Q, unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return nullptr;

  // Make sure the phi is on the LHS.
  if (!isa<PHINode>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<PHINode>(LHS) && "Not comparing with a phi instruction!");
  PHINode *PI = cast<PHINode>(LHS);

  // Bail out if RHS and the phi may be mutually interdependent due to a loop.
  if (!valueDominatesPHI(RHS, PI, Q.DT))
    return nullptr;

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = nullptr;
  for (unsigned u = 0, e = PI->getNumIncomingValues(); u < e; ++u) {
    Value *Incoming = PI->getIncomingValue(u);
    Instruction *InTI = PI->getIncomingBlock(u)->getTerminator();
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    // Change the context instruction to the "edge" that flows into the phi.
    // This is important because that is where incoming is actually "evaluated"
    // even though it is used later somewhere else.
    Value *V = SimplifyCmpInst(Pred, Incoming, RHS, Q.getWithInstruction(InTI),
                               MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return nullptr;
    CommonValue = V;
  }

  return CommonValue;
}

static Constant *foldOrCommuteConstant(Instruction::BinaryOps Opcode,
                                       Value *&Op0, Value *&Op1,
                                       const SimplifyQuery &Q) {
  if (auto *CLHS = dyn_cast<Constant>(Op0)) {
    if (auto *CRHS = dyn_cast<Constant>(Op1))
      return ConstantFoldBinaryOpOperands(Opcode, CLHS, CRHS, Q.DL);

    // Canonicalize the constant to the RHS if this is a commutative operation.
    if (Instruction::isCommutative(Opcode))
      std::swap(Op0, Op1);
  }
  return nullptr;
}

/// Given operands for an Add, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyAddInst(Value *Op0, Value *Op1, bool IsNSW, bool IsNUW,
                              const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::Add, Op0, Op1, Q))
    return C;

  // X + poison -> poison
  if (isa<PoisonValue>(Op1))
    return Op1;

  // X + undef -> undef
  if (Q.isUndefValue(Op1))
    return Op1;

  // X + 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // If two operands are negative, return 0.
  if (isKnownNegation(Op0, Op1))
    return Constant::getNullValue(Op0->getType());

  // X + (Y - X) -> Y
  // (Y - X) + X -> Y
  // Eg: X + -X -> 0
  Value *Y = nullptr;
  if (match(Op1, m_Sub(m_Value(Y), m_Specific(Op0))) ||
      match(Op0, m_Sub(m_Value(Y), m_Specific(Op1))))
    return Y;

  // X + ~X -> -1   since   ~X = -X-1
  Type *Ty = Op0->getType();
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Ty);

  // add nsw/nuw (xor Y, signmask), signmask --> Y
  // The no-wrapping add guarantees that the top bit will be set by the add.
  // Therefore, the xor must be clearing the already set sign bit of Y.
  if ((IsNSW || IsNUW) && match(Op1, m_SignMask()) &&
      match(Op0, m_Xor(m_Value(Y), m_SignMask())))
    return Y;

  // add nuw %x, -1  ->  -1, because %x can only be 0.
  if (IsNUW && match(Op1, m_AllOnes()))
    return Op1; // Which is -1.

  /// i1 add -> xor.
  if (MaxRecurse && Op0->getType()->isIntOrIntVectorTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Add, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Threading Add over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A + select(cond, B, C)" means evaluating
  // "A+B" and "A+C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A+B" and "A+C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifyAddInst(Value *Op0, Value *Op1, bool IsNSW, bool IsNUW,
                             const SimplifyQuery &Query) {
  return ::SimplifyAddInst(Op0, Op1, IsNSW, IsNUW, Query, RecursionLimit);
}

/// Compute the base pointer and cumulative constant offsets for V.
///
/// This strips all constant offsets off of V, leaving it the base pointer, and
/// accumulates the total constant offset applied in the returned constant. It
/// returns 0 if V is not a pointer, and returns the constant '0' if there are
/// no constant offsets applied.
///
/// This is very similar to GetPointerBaseWithConstantOffset except it doesn't
/// follow non-inbounds geps. This allows it to remain usable for icmp ult/etc.
/// folding.
static Constant *stripAndComputeConstantOffsets(const DataLayout &DL, Value *&V,
                                                bool AllowNonInbounds = false) {
  assert(V->getType()->isPtrOrPtrVectorTy());

  APInt Offset = APInt::getZero(DL.getIndexTypeSizeInBits(V->getType()));

  V = V->stripAndAccumulateConstantOffsets(DL, Offset, AllowNonInbounds);
  // As that strip may trace through `addrspacecast`, need to sext or trunc
  // the offset calculated.
  Type *IntIdxTy = DL.getIndexType(V->getType())->getScalarType();
  Offset = Offset.sextOrTrunc(IntIdxTy->getIntegerBitWidth());

  Constant *OffsetIntPtr = ConstantInt::get(IntIdxTy, Offset);
  if (VectorType *VecTy = dyn_cast<VectorType>(V->getType()))
    return ConstantVector::getSplat(VecTy->getElementCount(), OffsetIntPtr);
  return OffsetIntPtr;
}

/// Compute the constant difference between two pointer values.
/// If the difference is not a constant, returns zero.
static Constant *computePointerDifference(const DataLayout &DL, Value *LHS,
                                          Value *RHS) {
  Constant *LHSOffset = stripAndComputeConstantOffsets(DL, LHS);
  Constant *RHSOffset = stripAndComputeConstantOffsets(DL, RHS);

  // If LHS and RHS are not related via constant offsets to the same base
  // value, there is nothing we can do here.
  if (LHS != RHS)
    return nullptr;

  // Otherwise, the difference of LHS - RHS can be computed as:
  //    LHS - RHS
  //  = (LHSOffset + Base) - (RHSOffset + Base)
  //  = LHSOffset - RHSOffset
  return ConstantExpr::getSub(LHSOffset, RHSOffset);
}

/// Given operands for a Sub, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::Sub, Op0, Op1, Q))
    return C;

  // X - poison -> poison
  // poison - X -> poison
  if (isa<PoisonValue>(Op0) || isa<PoisonValue>(Op1))
    return PoisonValue::get(Op0->getType());

  // X - undef -> undef
  // undef - X -> undef
  if (Q.isUndefValue(Op0) || Q.isUndefValue(Op1))
    return UndefValue::get(Op0->getType());

  // X - 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X - X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // Is this a negation?
  if (match(Op0, m_Zero())) {
    // 0 - X -> 0 if the sub is NUW.
    if (isNUW)
      return Constant::getNullValue(Op0->getType());

    KnownBits Known = computeKnownBits(Op1, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    if (Known.Zero.isMaxSignedValue()) {
      // Op1 is either 0 or the minimum signed value. If the sub is NSW, then
      // Op1 must be 0 because negating the minimum signed value is undefined.
      if (isNSW)
        return Constant::getNullValue(Op0->getType());

      // 0 - X -> X if X is 0 or the minimum signed value.
      return Op1;
    }
  }

  // (X + Y) - Z -> X + (Y - Z) or Y + (X - Z) if everything simplifies.
  // For example, (X + Y) - Y -> X; (Y + X) - Y -> X
  Value *X = nullptr, *Y = nullptr, *Z = Op1;
  if (MaxRecurse && match(Op0, m_Add(m_Value(X), m_Value(Y)))) { // (X + Y) - Z
    // See if "V === Y - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Y, Z, Q, MaxRecurse-1))
      // It does!  Now see if "X + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, X, V, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, Q, MaxRecurse-1))
      // It does!  Now see if "Y + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, Y, V, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // X - (Y + Z) -> (X - Y) - Z or (X - Z) - Y if everything simplifies.
  // For example, X - (X + 1) -> -1
  X = Op0;
  if (MaxRecurse && match(Op1, m_Add(m_Value(Y), m_Value(Z)))) { // X - (Y + Z)
    // See if "V === X - Y" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Y, Q, MaxRecurse-1))
      // It does!  Now see if "V - Z" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Z, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, Q, MaxRecurse-1))
      // It does!  Now see if "V - Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Y, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // Z - (X - Y) -> (Z - X) + Y if everything simplifies.
  // For example, X - (X - Y) -> Y.
  Z = Op0;
  if (MaxRecurse && match(Op1, m_Sub(m_Value(X), m_Value(Y)))) // Z - (X - Y)
    // See if "V === Z - X" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Z, X, Q, MaxRecurse-1))
      // It does!  Now see if "V + Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, V, Y, Q, MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }

  // trunc(X) - trunc(Y) -> trunc(X - Y) if everything simplifies.
  if (MaxRecurse && match(Op0, m_Trunc(m_Value(X))) &&
      match(Op1, m_Trunc(m_Value(Y))))
    if (X->getType() == Y->getType())
      // See if "V === X - Y" simplifies.
      if (Value *V = SimplifyBinOp(Instruction::Sub, X, Y, Q, MaxRecurse-1))
        // It does!  Now see if "trunc V" simplifies.
        if (Value *W = SimplifyCastInst(Instruction::Trunc, V, Op0->getType(),
                                        Q, MaxRecurse - 1))
          // It does, return the simplified "trunc V".
          return W;

  // Variations on GEP(base, I, ...) - GEP(base, i, ...) -> GEP(null, I-i, ...).
  if (match(Op0, m_PtrToInt(m_Value(X))) &&
      match(Op1, m_PtrToInt(m_Value(Y))))
    if (Constant *Result = computePointerDifference(Q.DL, X, Y))
      return ConstantExpr::getIntegerCast(Result, Op0->getType(), true);

  // i1 sub -> xor.
  if (MaxRecurse && Op0->getType()->isIntOrIntVectorTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Threading Sub over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A - select(cond, B, C)" means evaluating
  // "A-B" and "A-C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A-B" and "A-C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const SimplifyQuery &Q) {
  return ::SimplifySubInst(Op0, Op1, isNSW, isNUW, Q, RecursionLimit);
}

/// Given operands for a Mul, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyMulInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                              unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::Mul, Op0, Op1, Q))
    return C;

  // X * poison -> poison
  if (isa<PoisonValue>(Op1))
    return Op1;

  // X * undef -> 0
  // X * 0 -> 0
  if (Q.isUndefValue(Op1) || match(Op1, m_Zero()))
    return Constant::getNullValue(Op0->getType());

  // X * 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  // (X / Y) * Y -> X if the division is exact.
  Value *X = nullptr;
  if (Q.IIQ.UseInstrInfo &&
      (match(Op0,
             m_Exact(m_IDiv(m_Value(X), m_Specific(Op1)))) ||     // (X / Y) * Y
       match(Op1, m_Exact(m_IDiv(m_Value(X), m_Specific(Op0)))))) // Y * (X / Y)
    return X;

  // i1 mul -> and.
  if (MaxRecurse && Op0->getType()->isIntOrIntVectorTy(1))
    if (Value *V = SimplifyAndInst(Op0, Op1, Q, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Mul, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Mul distributes over Add. Try some generic simplifications based on this.
  if (Value *V = expandCommutativeBinOp(Instruction::Mul, Op0, Op1,
                                        Instruction::Add, Q, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::Mul, Op0, Op1, Q,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Mul, Op0, Op1, Q,
                                      MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyMulInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyMulInst(Op0, Op1, Q, RecursionLimit);
}

/// Check for common or similar folds of integer division or integer remainder.
/// This applies to all 4 opcodes (sdiv/udiv/srem/urem).
static Value *simplifyDivRem(Instruction::BinaryOps Opcode, Value *Op0,
                             Value *Op1, const SimplifyQuery &Q) {
  bool IsDiv = (Opcode == Instruction::SDiv || Opcode == Instruction::UDiv);
  bool IsSigned = (Opcode == Instruction::SDiv || Opcode == Instruction::SRem);

  Type *Ty = Op0->getType();

  // X / undef -> poison
  // X % undef -> poison
  if (Q.isUndefValue(Op1) || isa<PoisonValue>(Op1))
    return PoisonValue::get(Ty);

  // X / 0 -> poison
  // X % 0 -> poison
  // We don't need to preserve faults!
  if (match(Op1, m_Zero()))
    return PoisonValue::get(Ty);

  // If any element of a constant divisor fixed width vector is zero or undef
  // the behavior is undefined and we can fold the whole op to poison.
  auto *Op1C = dyn_cast<Constant>(Op1);
  auto *VTy = dyn_cast<FixedVectorType>(Ty);
  if (Op1C && VTy) {
    unsigned NumElts = VTy->getNumElements();
    for (unsigned i = 0; i != NumElts; ++i) {
      Constant *Elt = Op1C->getAggregateElement(i);
      if (Elt && (Elt->isNullValue() || Q.isUndefValue(Elt)))
        return PoisonValue::get(Ty);
    }
  }

  // poison / X -> poison
  // poison % X -> poison
  if (isa<PoisonValue>(Op0))
    return Op0;

  // undef / X -> 0
  // undef % X -> 0
  if (Q.isUndefValue(Op0))
    return Constant::getNullValue(Ty);

  // 0 / X -> 0
  // 0 % X -> 0
  if (match(Op0, m_Zero()))
    return Constant::getNullValue(Op0->getType());

  // X / X -> 1
  // X % X -> 0
  if (Op0 == Op1)
    return IsDiv ? ConstantInt::get(Ty, 1) : Constant::getNullValue(Ty);

  // X / 1 -> X
  // X % 1 -> 0
  // If this is a boolean op (single-bit element type), we can't have
  // division-by-zero or remainder-by-zero, so assume the divisor is 1.
  // Similarly, if we're zero-extending a boolean divisor, then assume it's a 1.
  Value *X;
  if (match(Op1, m_One()) || Ty->isIntOrIntVectorTy(1) ||
      (match(Op1, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)))
    return IsDiv ? Op0 : Constant::getNullValue(Ty);

  // If X * Y does not overflow, then:
  //   X * Y / Y -> X
  //   X * Y % Y -> 0
  if (match(Op0, m_c_Mul(m_Value(X), m_Specific(Op1)))) {
    auto *Mul = cast<OverflowingBinaryOperator>(Op0);
    // The multiplication can't overflow if it is defined not to, or if
    // X == A / Y for some A.
    if ((IsSigned && Q.IIQ.hasNoSignedWrap(Mul)) ||
        (!IsSigned && Q.IIQ.hasNoUnsignedWrap(Mul)) ||
        (IsSigned && match(X, m_SDiv(m_Value(), m_Specific(Op1)))) ||
        (!IsSigned && match(X, m_UDiv(m_Value(), m_Specific(Op1))))) {
      return IsDiv ? X : Constant::getNullValue(Op0->getType());
    }
  }

  return nullptr;
}

/// Given a predicate and two operands, return true if the comparison is true.
/// This is a helper for div/rem simplification where we return some other value
/// when we can prove a relationship between the operands.
static bool isICmpTrue(ICmpInst::Predicate Pred, Value *LHS, Value *RHS,
                       const SimplifyQuery &Q, unsigned MaxRecurse) {
  Value *V = SimplifyICmpInst(Pred, LHS, RHS, Q, MaxRecurse);
  Constant *C = dyn_cast_or_null<Constant>(V);
  return (C && C->isAllOnesValue());
}

/// Return true if we can simplify X / Y to 0. Remainder can adapt that answer
/// to simplify X % Y to X.
static bool isDivZero(Value *X, Value *Y, const SimplifyQuery &Q,
                      unsigned MaxRecurse, bool IsSigned) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return false;

  if (IsSigned) {
    // |X| / |Y| --> 0
    //
    // We require that 1 operand is a simple constant. That could be extended to
    // 2 variables if we computed the sign bit for each.
    //
    // Make sure that a constant is not the minimum signed value because taking
    // the abs() of that is undefined.
    Type *Ty = X->getType();
    const APInt *C;
    if (match(X, m_APInt(C)) && !C->isMinSignedValue()) {
      // Is the variable divisor magnitude always greater than the constant
      // dividend magnitude?
      // |Y| > |C| --> Y < -abs(C) or Y > abs(C)
      Constant *PosDividendC = ConstantInt::get(Ty, C->abs());
      Constant *NegDividendC = ConstantInt::get(Ty, -C->abs());
      if (isICmpTrue(CmpInst::ICMP_SLT, Y, NegDividendC, Q, MaxRecurse) ||
          isICmpTrue(CmpInst::ICMP_SGT, Y, PosDividendC, Q, MaxRecurse))
        return true;
    }
    if (match(Y, m_APInt(C))) {
      // Special-case: we can't take the abs() of a minimum signed value. If
      // that's the divisor, then all we have to do is prove that the dividend
      // is also not the minimum signed value.
      if (C->isMinSignedValue())
        return isICmpTrue(CmpInst::ICMP_NE, X, Y, Q, MaxRecurse);

      // Is the variable dividend magnitude always less than the constant
      // divisor magnitude?
      // |X| < |C| --> X > -abs(C) and X < abs(C)
      Constant *PosDivisorC = ConstantInt::get(Ty, C->abs());
      Constant *NegDivisorC = ConstantInt::get(Ty, -C->abs());
      if (isICmpTrue(CmpInst::ICMP_SGT, X, NegDivisorC, Q, MaxRecurse) &&
          isICmpTrue(CmpInst::ICMP_SLT, X, PosDivisorC, Q, MaxRecurse))
        return true;
    }
    return false;
  }

  // IsSigned == false.

  // Is the unsigned dividend known to be less than a constant divisor?
  // TODO: Convert this (and above) to range analysis
  //      ("computeConstantRangeIncludingKnownBits")?
  const APInt *C;
  if (match(Y, m_APInt(C)) &&
      computeKnownBits(X, Q.DL, 0, Q.AC, Q.CxtI, Q.DT).getMaxValue().ult(*C))
    return true;

  // Try again for any divisor:
  // Is the dividend unsigned less than the divisor?
  return isICmpTrue(ICmpInst::ICMP_ULT, X, Y, Q, MaxRecurse);
}

/// These are simplifications common to SDiv and UDiv.
static Value *simplifyDiv(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Opcode, Op0, Op1, Q))
    return C;

  if (Value *V = simplifyDivRem(Opcode, Op0, Op1, Q))
    return V;

  bool IsSigned = Opcode == Instruction::SDiv;

  // (X rem Y) / Y -> 0
  if ((IsSigned && match(Op0, m_SRem(m_Value(), m_Specific(Op1)))) ||
      (!IsSigned && match(Op0, m_URem(m_Value(), m_Specific(Op1)))))
    return Constant::getNullValue(Op0->getType());

  // (X /u C1) /u C2 -> 0 if C1 * C2 overflow
  ConstantInt *C1, *C2;
  if (!IsSigned && match(Op0, m_UDiv(m_Value(), m_ConstantInt(C1))) &&
      match(Op1, m_ConstantInt(C2))) {
    bool Overflow;
    (void)C1->getValue().umul_ov(C2->getValue(), Overflow);
    if (Overflow)
      return Constant::getNullValue(Op0->getType());
  }

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  if (isDivZero(Op0, Op1, Q, MaxRecurse, IsSigned))
    return Constant::getNullValue(Op0->getType());

  return nullptr;
}

/// These are simplifications common to SRem and URem.
static Value *simplifyRem(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Opcode, Op0, Op1, Q))
    return C;

  if (Value *V = simplifyDivRem(Opcode, Op0, Op1, Q))
    return V;

  // (X % Y) % Y -> X % Y
  if ((Opcode == Instruction::SRem &&
       match(Op0, m_SRem(m_Value(), m_Specific(Op1)))) ||
      (Opcode == Instruction::URem &&
       match(Op0, m_URem(m_Value(), m_Specific(Op1)))))
    return Op0;

  // (X << Y) % X -> 0
  if (Q.IIQ.UseInstrInfo &&
      ((Opcode == Instruction::SRem &&
        match(Op0, m_NSWShl(m_Specific(Op1), m_Value()))) ||
       (Opcode == Instruction::URem &&
        match(Op0, m_NUWShl(m_Specific(Op1), m_Value())))))
    return Constant::getNullValue(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If X / Y == 0, then X % Y == X.
  if (isDivZero(Op0, Op1, Q, MaxRecurse, Opcode == Instruction::SRem))
    return Op0;

  return nullptr;
}

/// Given operands for an SDiv, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifySDivInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                               unsigned MaxRecurse) {
  // If two operands are negated and no signed overflow, return -1.
  if (isKnownNegation(Op0, Op1, /*NeedNSW=*/true))
    return Constant::getAllOnesValue(Op0->getType());

  return simplifyDiv(Instruction::SDiv, Op0, Op1, Q, MaxRecurse);
}

Value *llvm::SimplifySDivInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifySDivInst(Op0, Op1, Q, RecursionLimit);
}

/// Given operands for a UDiv, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyUDivInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                               unsigned MaxRecurse) {
  return simplifyDiv(Instruction::UDiv, Op0, Op1, Q, MaxRecurse);
}

Value *llvm::SimplifyUDivInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyUDivInst(Op0, Op1, Q, RecursionLimit);
}

/// Given operands for an SRem, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifySRemInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                               unsigned MaxRecurse) {
  // If the divisor is 0, the result is undefined, so assume the divisor is -1.
  // srem Op0, (sext i1 X) --> srem Op0, -1 --> 0
  Value *X;
  if (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
    return ConstantInt::getNullValue(Op0->getType());

  // If the two operands are negated, return 0.
  if (isKnownNegation(Op0, Op1))
    return ConstantInt::getNullValue(Op0->getType());

  return simplifyRem(Instruction::SRem, Op0, Op1, Q, MaxRecurse);
}

Value *llvm::SimplifySRemInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifySRemInst(Op0, Op1, Q, RecursionLimit);
}

/// Given operands for a URem, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyURemInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                               unsigned MaxRecurse) {
  return simplifyRem(Instruction::URem, Op0, Op1, Q, MaxRecurse);
}

Value *llvm::SimplifyURemInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyURemInst(Op0, Op1, Q, RecursionLimit);
}

/// Returns true if a shift by \c Amount always yields poison.
static bool isPoisonShift(Value *Amount, const SimplifyQuery &Q) {
  Constant *C = dyn_cast<Constant>(Amount);
  if (!C)
    return false;

  // X shift by undef -> poison because it may shift by the bitwidth.
  if (Q.isUndefValue(C))
    return true;

  // Shifting by the bitwidth or more is undefined.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
    if (CI->getValue().uge(CI->getType()->getScalarSizeInBits()))
      return true;

  // If all lanes of a vector shift are undefined the whole shift is.
  if (isa<ConstantVector>(C) || isa<ConstantDataVector>(C)) {
    for (unsigned I = 0,
                  E = cast<FixedVectorType>(C->getType())->getNumElements();
         I != E; ++I)
      if (!isPoisonShift(C->getAggregateElement(I), Q))
        return false;
    return true;
  }

  return false;
}

/// Given operands for an Shl, LShr or AShr, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyShift(Instruction::BinaryOps Opcode, Value *Op0,
                            Value *Op1, bool IsNSW, const SimplifyQuery &Q,
                            unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Opcode, Op0, Op1, Q))
    return C;

  // poison shift by X -> poison
  if (isa<PoisonValue>(Op0))
    return Op0;

  // 0 shift by X -> 0
  if (match(Op0, m_Zero()))
    return Constant::getNullValue(Op0->getType());

  // X shift by 0 -> X
  // Shift-by-sign-extended bool must be shift-by-0 because shift-by-all-ones
  // would be poison.
  Value *X;
  if (match(Op1, m_Zero()) ||
      (match(Op1, m_SExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1)))
    return Op0;

  // Fold undefined shifts.
  if (isPoisonShift(Op1, Q))
    return PoisonValue::get(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, Q, MaxRecurse))
      return V;

  // If any bits in the shift amount make that value greater than or equal to
  // the number of bits in the type, the shift is undefined.
  KnownBits KnownAmt = computeKnownBits(Op1, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
  if (KnownAmt.getMinValue().uge(KnownAmt.getBitWidth()))
    return PoisonValue::get(Op0->getType());

  // If all valid bits in the shift amount are known zero, the first operand is
  // unchanged.
  unsigned NumValidShiftBits = Log2_32_Ceil(KnownAmt.getBitWidth());
  if (KnownAmt.countMinTrailingZeros() >= NumValidShiftBits)
    return Op0;

  // Check for nsw shl leading to a poison value.
  if (IsNSW) {
    assert(Opcode == Instruction::Shl && "Expected shl for nsw instruction");
    KnownBits KnownVal = computeKnownBits(Op0, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    KnownBits KnownShl = KnownBits::shl(KnownVal, KnownAmt);

    if (KnownVal.Zero.isSignBitSet())
      KnownShl.Zero.setSignBit();
    if (KnownVal.One.isSignBitSet())
      KnownShl.One.setSignBit();

    if (KnownShl.hasConflict())
      return PoisonValue::get(Op0->getType());
  }

  return nullptr;
}

/// Given operands for an Shl, LShr or AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyRightShift(Instruction::BinaryOps Opcode, Value *Op0,
                                 Value *Op1, bool isExact, const SimplifyQuery &Q,
                                 unsigned MaxRecurse) {
  if (Value *V =
          SimplifyShift(Opcode, Op0, Op1, /*IsNSW*/ false, Q, MaxRecurse))
    return V;

  // X >> X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // undef >> X -> 0
  // undef >> X -> undef (if it's exact)
  if (Q.isUndefValue(Op0))
    return isExact ? Op0 : Constant::getNullValue(Op0->getType());

  // The low bit cannot be shifted out of an exact shift if it is set.
  if (isExact) {
    KnownBits Op0Known = computeKnownBits(Op0, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT);
    if (Op0Known.One[0])
      return Op0;
  }

  return nullptr;
}

/// Given operands for an Shl, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Value *V =
          SimplifyShift(Instruction::Shl, Op0, Op1, isNSW, Q, MaxRecurse))
    return V;

  // undef << X -> 0
  // undef << X -> undef if (if it's NSW/NUW)
  if (Q.isUndefValue(Op0))
    return isNSW || isNUW ? Op0 : Constant::getNullValue(Op0->getType());

  // (X >> A) << A -> X
  Value *X;
  if (Q.IIQ.UseInstrInfo &&
      match(Op0, m_Exact(m_Shr(m_Value(X), m_Specific(Op1)))))
    return X;

  // shl nuw i8 C, %x  ->  C  iff C has sign bit set.
  if (isNUW && match(Op0, m_Negative()))
    return Op0;
  // NOTE: could use computeKnownBits() / LazyValueInfo,
  // but the cost-benefit analysis suggests it isn't worth it.

  return nullptr;
}

Value *llvm::SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const SimplifyQuery &Q) {
  return ::SimplifyShlInst(Op0, Op1, isNSW, isNUW, Q, RecursionLimit);
}

/// Given operands for an LShr, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                               const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Value *V = SimplifyRightShift(Instruction::LShr, Op0, Op1, isExact, Q,
                                    MaxRecurse))
      return V;

  // (X << A) >> A -> X
  Value *X;
  if (match(Op0, m_NUWShl(m_Value(X), m_Specific(Op1))))
    return X;

  // ((X << A) | Y) >> A -> X  if effective width of Y is not larger than A.
  // We can return X as we do in the above case since OR alters no bits in X.
  // SimplifyDemandedBits in InstCombine can do more general optimization for
  // bit manipulation. This pattern aims to provide opportunities for other
  // optimizers by supporting a simple but common case in InstSimplify.
  Value *Y;
  const APInt *ShRAmt, *ShLAmt;
  if (match(Op1, m_APInt(ShRAmt)) &&
      match(Op0, m_c_Or(m_NUWShl(m_Value(X), m_APInt(ShLAmt)), m_Value(Y))) &&
      *ShRAmt == *ShLAmt) {
    const KnownBits YKnown = computeKnownBits(Y, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    const unsigned EffWidthY = YKnown.countMaxActiveBits();
    if (ShRAmt->uge(EffWidthY))
      return X;
  }

  return nullptr;
}

Value *llvm::SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                              const SimplifyQuery &Q) {
  return ::SimplifyLShrInst(Op0, Op1, isExact, Q, RecursionLimit);
}

/// Given operands for an AShr, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                               const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Value *V = SimplifyRightShift(Instruction::AShr, Op0, Op1, isExact, Q,
                                    MaxRecurse))
    return V;

  // -1 >>a X --> -1
  // (-1 << X) a>> X --> -1
  // Do not return Op0 because it may contain undef elements if it's a vector.
  if (match(Op0, m_AllOnes()) ||
      match(Op0, m_Shl(m_AllOnes(), m_Specific(Op1))))
    return Constant::getAllOnesValue(Op0->getType());

  // (X << A) >> A -> X
  Value *X;
  if (Q.IIQ.UseInstrInfo && match(Op0, m_NSWShl(m_Value(X), m_Specific(Op1))))
    return X;

  // Arithmetic shifting an all-sign-bit value is a no-op.
  unsigned NumSignBits = ComputeNumSignBits(Op0, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
  if (NumSignBits == Op0->getType()->getScalarSizeInBits())
    return Op0;

  return nullptr;
}

Value *llvm::SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                              const SimplifyQuery &Q) {
  return ::SimplifyAShrInst(Op0, Op1, isExact, Q, RecursionLimit);
}

/// Commuted variants are assumed to be handled by calling this function again
/// with the parameters swapped.
static Value *simplifyUnsignedRangeCheck(ICmpInst *ZeroICmp,
                                         ICmpInst *UnsignedICmp, bool IsAnd,
                                         const SimplifyQuery &Q) {
  Value *X, *Y;

  ICmpInst::Predicate EqPred;
  if (!match(ZeroICmp, m_ICmp(EqPred, m_Value(Y), m_Zero())) ||
      !ICmpInst::isEquality(EqPred))
    return nullptr;

  ICmpInst::Predicate UnsignedPred;

  Value *A, *B;
  // Y = (A - B);
  if (match(Y, m_Sub(m_Value(A), m_Value(B)))) {
    if (match(UnsignedICmp,
              m_c_ICmp(UnsignedPred, m_Specific(A), m_Specific(B))) &&
        ICmpInst::isUnsigned(UnsignedPred)) {
      // A >=/<= B || (A - B) != 0  <-->  true
      if ((UnsignedPred == ICmpInst::ICMP_UGE ||
           UnsignedPred == ICmpInst::ICMP_ULE) &&
          EqPred == ICmpInst::ICMP_NE && !IsAnd)
        return ConstantInt::getTrue(UnsignedICmp->getType());
      // A </> B && (A - B) == 0  <-->  false
      if ((UnsignedPred == ICmpInst::ICMP_ULT ||
           UnsignedPred == ICmpInst::ICMP_UGT) &&
          EqPred == ICmpInst::ICMP_EQ && IsAnd)
        return ConstantInt::getFalse(UnsignedICmp->getType());

      // A </> B && (A - B) != 0  <-->  A </> B
      // A </> B || (A - B) != 0  <-->  (A - B) != 0
      if (EqPred == ICmpInst::ICMP_NE && (UnsignedPred == ICmpInst::ICMP_ULT ||
                                          UnsignedPred == ICmpInst::ICMP_UGT))
        return IsAnd ? UnsignedICmp : ZeroICmp;

      // A <=/>= B && (A - B) == 0  <-->  (A - B) == 0
      // A <=/>= B || (A - B) == 0  <-->  A <=/>= B
      if (EqPred == ICmpInst::ICMP_EQ && (UnsignedPred == ICmpInst::ICMP_ULE ||
                                          UnsignedPred == ICmpInst::ICMP_UGE))
        return IsAnd ? ZeroICmp : UnsignedICmp;
    }

    // Given  Y = (A - B)
    //   Y >= A && Y != 0  --> Y >= A  iff B != 0
    //   Y <  A || Y == 0  --> Y <  A  iff B != 0
    if (match(UnsignedICmp,
              m_c_ICmp(UnsignedPred, m_Specific(Y), m_Specific(A)))) {
      if (UnsignedPred == ICmpInst::ICMP_UGE && IsAnd &&
          EqPred == ICmpInst::ICMP_NE &&
          isKnownNonZero(B, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT))
        return UnsignedICmp;
      if (UnsignedPred == ICmpInst::ICMP_ULT && !IsAnd &&
          EqPred == ICmpInst::ICMP_EQ &&
          isKnownNonZero(B, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT))
        return UnsignedICmp;
    }
  }

  if (match(UnsignedICmp, m_ICmp(UnsignedPred, m_Value(X), m_Specific(Y))) &&
      ICmpInst::isUnsigned(UnsignedPred))
    ;
  else if (match(UnsignedICmp,
                 m_ICmp(UnsignedPred, m_Specific(Y), m_Value(X))) &&
           ICmpInst::isUnsigned(UnsignedPred))
    UnsignedPred = ICmpInst::getSwappedPredicate(UnsignedPred);
  else
    return nullptr;

  // X > Y && Y == 0  -->  Y == 0  iff X != 0
  // X > Y || Y == 0  -->  X > Y   iff X != 0
  if (UnsignedPred == ICmpInst::ICMP_UGT && EqPred == ICmpInst::ICMP_EQ &&
      isKnownNonZero(X, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT))
    return IsAnd ? ZeroICmp : UnsignedICmp;

  // X <= Y && Y != 0  -->  X <= Y  iff X != 0
  // X <= Y || Y != 0  -->  Y != 0  iff X != 0
  if (UnsignedPred == ICmpInst::ICMP_ULE && EqPred == ICmpInst::ICMP_NE &&
      isKnownNonZero(X, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT))
    return IsAnd ? UnsignedICmp : ZeroICmp;

  // The transforms below here are expected to be handled more generally with
  // simplifyAndOrOfICmpsWithLimitConst() or in InstCombine's
  // foldAndOrOfICmpsWithConstEq(). If we are looking to trim optimizer overlap,
  // these are candidates for removal.

  // X < Y && Y != 0  -->  X < Y
  // X < Y || Y != 0  -->  Y != 0
  if (UnsignedPred == ICmpInst::ICMP_ULT && EqPred == ICmpInst::ICMP_NE)
    return IsAnd ? UnsignedICmp : ZeroICmp;

  // X >= Y && Y == 0  -->  Y == 0
  // X >= Y || Y == 0  -->  X >= Y
  if (UnsignedPred == ICmpInst::ICMP_UGE && EqPred == ICmpInst::ICMP_EQ)
    return IsAnd ? ZeroICmp : UnsignedICmp;

  // X < Y && Y == 0  -->  false
  if (UnsignedPred == ICmpInst::ICMP_ULT && EqPred == ICmpInst::ICMP_EQ &&
      IsAnd)
    return getFalse(UnsignedICmp->getType());

  // X >= Y || Y != 0  -->  true
  if (UnsignedPred == ICmpInst::ICMP_UGE && EqPred == ICmpInst::ICMP_NE &&
      !IsAnd)
    return getTrue(UnsignedICmp->getType());

  return nullptr;
}

/// Commuted variants are assumed to be handled by calling this function again
/// with the parameters swapped.
static Value *simplifyAndOfICmpsWithSameOperands(ICmpInst *Op0, ICmpInst *Op1) {
  ICmpInst::Predicate Pred0, Pred1;
  Value *A ,*B;
  if (!match(Op0, m_ICmp(Pred0, m_Value(A), m_Value(B))) ||
      !match(Op1, m_ICmp(Pred1, m_Specific(A), m_Specific(B))))
    return nullptr;

  // We have (icmp Pred0, A, B) & (icmp Pred1, A, B).
  // If Op1 is always implied true by Op0, then Op0 is a subset of Op1, and we
  // can eliminate Op1 from this 'and'.
  if (ICmpInst::isImpliedTrueByMatchingCmp(Pred0, Pred1))
    return Op0;

  // Check for any combination of predicates that are guaranteed to be disjoint.
  if ((Pred0 == ICmpInst::getInversePredicate(Pred1)) ||
      (Pred0 == ICmpInst::ICMP_EQ && ICmpInst::isFalseWhenEqual(Pred1)) ||
      (Pred0 == ICmpInst::ICMP_SLT && Pred1 == ICmpInst::ICMP_SGT) ||
      (Pred0 == ICmpInst::ICMP_ULT && Pred1 == ICmpInst::ICMP_UGT))
    return getFalse(Op0->getType());

  return nullptr;
}

/// Commuted variants are assumed to be handled by calling this function again
/// with the parameters swapped.
static Value *simplifyOrOfICmpsWithSameOperands(ICmpInst *Op0, ICmpInst *Op1) {
  ICmpInst::Predicate Pred0, Pred1;
  Value *A ,*B;
  if (!match(Op0, m_ICmp(Pred0, m_Value(A), m_Value(B))) ||
      !match(Op1, m_ICmp(Pred1, m_Specific(A), m_Specific(B))))
    return nullptr;

  // We have (icmp Pred0, A, B) | (icmp Pred1, A, B).
  // If Op1 is always implied true by Op0, then Op0 is a subset of Op1, and we
  // can eliminate Op0 from this 'or'.
  if (ICmpInst::isImpliedTrueByMatchingCmp(Pred0, Pred1))
    return Op1;

  // Check for any combination of predicates that cover the entire range of
  // possibilities.
  if ((Pred0 == ICmpInst::getInversePredicate(Pred1)) ||
      (Pred0 == ICmpInst::ICMP_NE && ICmpInst::isTrueWhenEqual(Pred1)) ||
      (Pred0 == ICmpInst::ICMP_SLE && Pred1 == ICmpInst::ICMP_SGE) ||
      (Pred0 == ICmpInst::ICMP_ULE && Pred1 == ICmpInst::ICMP_UGE))
    return getTrue(Op0->getType());

  return nullptr;
}

/// Test if a pair of compares with a shared operand and 2 constants has an
/// empty set intersection, full set union, or if one compare is a superset of
/// the other.
static Value *simplifyAndOrOfICmpsWithConstants(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                                bool IsAnd) {
  // Look for this pattern: {and/or} (icmp X, C0), (icmp X, C1)).
  if (Cmp0->getOperand(0) != Cmp1->getOperand(0))
    return nullptr;

  const APInt *C0, *C1;
  if (!match(Cmp0->getOperand(1), m_APInt(C0)) ||
      !match(Cmp1->getOperand(1), m_APInt(C1)))
    return nullptr;

  auto Range0 = ConstantRange::makeExactICmpRegion(Cmp0->getPredicate(), *C0);
  auto Range1 = ConstantRange::makeExactICmpRegion(Cmp1->getPredicate(), *C1);

  // For and-of-compares, check if the intersection is empty:
  // (icmp X, C0) && (icmp X, C1) --> empty set --> false
  if (IsAnd && Range0.intersectWith(Range1).isEmptySet())
    return getFalse(Cmp0->getType());

  // For or-of-compares, check if the union is full:
  // (icmp X, C0) || (icmp X, C1) --> full set --> true
  if (!IsAnd && Range0.unionWith(Range1).isFullSet())
    return getTrue(Cmp0->getType());

  // Is one range a superset of the other?
  // If this is and-of-compares, take the smaller set:
  // (icmp sgt X, 4) && (icmp sgt X, 42) --> icmp sgt X, 42
  // If this is or-of-compares, take the larger set:
  // (icmp sgt X, 4) || (icmp sgt X, 42) --> icmp sgt X, 4
  if (Range0.contains(Range1))
    return IsAnd ? Cmp1 : Cmp0;
  if (Range1.contains(Range0))
    return IsAnd ? Cmp0 : Cmp1;

  return nullptr;
}

static Value *simplifyAndOrOfICmpsWithZero(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                           bool IsAnd) {
  ICmpInst::Predicate P0 = Cmp0->getPredicate(), P1 = Cmp1->getPredicate();
  if (!match(Cmp0->getOperand(1), m_Zero()) ||
      !match(Cmp1->getOperand(1), m_Zero()) || P0 != P1)
    return nullptr;

  if ((IsAnd && P0 != ICmpInst::ICMP_NE) || (!IsAnd && P1 != ICmpInst::ICMP_EQ))
    return nullptr;

  // We have either "(X == 0 || Y == 0)" or "(X != 0 && Y != 0)".
  Value *X = Cmp0->getOperand(0);
  Value *Y = Cmp1->getOperand(0);

  // If one of the compares is a masked version of a (not) null check, then
  // that compare implies the other, so we eliminate the other. Optionally, look
  // through a pointer-to-int cast to match a null check of a pointer type.

  // (X == 0) || (([ptrtoint] X & ?) == 0) --> ([ptrtoint] X & ?) == 0
  // (X == 0) || ((? & [ptrtoint] X) == 0) --> (? & [ptrtoint] X) == 0
  // (X != 0) && (([ptrtoint] X & ?) != 0) --> ([ptrtoint] X & ?) != 0
  // (X != 0) && ((? & [ptrtoint] X) != 0) --> (? & [ptrtoint] X) != 0
  if (match(Y, m_c_And(m_Specific(X), m_Value())) ||
      match(Y, m_c_And(m_PtrToInt(m_Specific(X)), m_Value())))
    return Cmp1;

  // (([ptrtoint] Y & ?) == 0) || (Y == 0) --> ([ptrtoint] Y & ?) == 0
  // ((? & [ptrtoint] Y) == 0) || (Y == 0) --> (? & [ptrtoint] Y) == 0
  // (([ptrtoint] Y & ?) != 0) && (Y != 0) --> ([ptrtoint] Y & ?) != 0
  // ((? & [ptrtoint] Y) != 0) && (Y != 0) --> (? & [ptrtoint] Y) != 0
  if (match(X, m_c_And(m_Specific(Y), m_Value())) ||
      match(X, m_c_And(m_PtrToInt(m_Specific(Y)), m_Value())))
    return Cmp0;

  return nullptr;
}

static Value *simplifyAndOfICmpsWithAdd(ICmpInst *Op0, ICmpInst *Op1,
                                        const InstrInfoQuery &IIQ) {
  // (icmp (add V, C0), C1) & (icmp V, C0)
  ICmpInst::Predicate Pred0, Pred1;
  const APInt *C0, *C1;
  Value *V;
  if (!match(Op0, m_ICmp(Pred0, m_Add(m_Value(V), m_APInt(C0)), m_APInt(C1))))
    return nullptr;

  if (!match(Op1, m_ICmp(Pred1, m_Specific(V), m_Value())))
    return nullptr;

  auto *AddInst = cast<OverflowingBinaryOperator>(Op0->getOperand(0));
  if (AddInst->getOperand(1) != Op1->getOperand(1))
    return nullptr;

  Type *ITy = Op0->getType();
  bool isNSW = IIQ.hasNoSignedWrap(AddInst);
  bool isNUW = IIQ.hasNoUnsignedWrap(AddInst);

  const APInt Delta = *C1 - *C0;
  if (C0->isStrictlyPositive()) {
    if (Delta == 2) {
      if (Pred0 == ICmpInst::ICMP_ULT && Pred1 == ICmpInst::ICMP_SGT)
        return getFalse(ITy);
      if (Pred0 == ICmpInst::ICMP_SLT && Pred1 == ICmpInst::ICMP_SGT && isNSW)
        return getFalse(ITy);
    }
    if (Delta == 1) {
      if (Pred0 == ICmpInst::ICMP_ULE && Pred1 == ICmpInst::ICMP_SGT)
        return getFalse(ITy);
      if (Pred0 == ICmpInst::ICMP_SLE && Pred1 == ICmpInst::ICMP_SGT && isNSW)
        return getFalse(ITy);
    }
  }
  if (C0->getBoolValue() && isNUW) {
    if (Delta == 2)
      if (Pred0 == ICmpInst::ICMP_ULT && Pred1 == ICmpInst::ICMP_UGT)
        return getFalse(ITy);
    if (Delta == 1)
      if (Pred0 == ICmpInst::ICMP_ULE && Pred1 == ICmpInst::ICMP_UGT)
        return getFalse(ITy);
  }

  return nullptr;
}

/// Try to eliminate compares with signed or unsigned min/max constants.
static Value *simplifyAndOrOfICmpsWithLimitConst(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                                 bool IsAnd) {
  // Canonicalize an equality compare as Cmp0.
  if (Cmp1->isEquality())
    std::swap(Cmp0, Cmp1);
  if (!Cmp0->isEquality())
    return nullptr;

  // The non-equality compare must include a common operand (X). Canonicalize
  // the common operand as operand 0 (the predicate is swapped if the common
  // operand was operand 1).
  ICmpInst::Predicate Pred0 = Cmp0->getPredicate();
  Value *X = Cmp0->getOperand(0);
  ICmpInst::Predicate Pred1;
  bool HasNotOp = match(Cmp1, m_c_ICmp(Pred1, m_Not(m_Specific(X)), m_Value()));
  if (!HasNotOp && !match(Cmp1, m_c_ICmp(Pred1, m_Specific(X), m_Value())))
    return nullptr;
  if (ICmpInst::isEquality(Pred1))
    return nullptr;

  // The equality compare must be against a constant. Flip bits if we matched
  // a bitwise not. Convert a null pointer constant to an integer zero value.
  APInt MinMaxC;
  const APInt *C;
  if (match(Cmp0->getOperand(1), m_APInt(C)))
    MinMaxC = HasNotOp ? ~*C : *C;
  else if (isa<ConstantPointerNull>(Cmp0->getOperand(1)))
    MinMaxC = APInt::getZero(8);
  else
    return nullptr;

  // DeMorganize if this is 'or': P0 || P1 --> !P0 && !P1.
  if (!IsAnd) {
    Pred0 = ICmpInst::getInversePredicate(Pred0);
    Pred1 = ICmpInst::getInversePredicate(Pred1);
  }

  // Normalize to unsigned compare and unsigned min/max value.
  // Example for 8-bit: -128 + 128 -> 0; 127 + 128 -> 255
  if (ICmpInst::isSigned(Pred1)) {
    Pred1 = ICmpInst::getUnsignedPredicate(Pred1);
    MinMaxC += APInt::getSignedMinValue(MinMaxC.getBitWidth());
  }

  // (X != MAX) && (X < Y) --> X < Y
  // (X == MAX) || (X >= Y) --> X >= Y
  if (MinMaxC.isMaxValue())
    if (Pred0 == ICmpInst::ICMP_NE && Pred1 == ICmpInst::ICMP_ULT)
      return Cmp1;

  // (X != MIN) && (X > Y) -->  X > Y
  // (X == MIN) || (X <= Y) --> X <= Y
  if (MinMaxC.isMinValue())
    if (Pred0 == ICmpInst::ICMP_NE && Pred1 == ICmpInst::ICMP_UGT)
      return Cmp1;

  return nullptr;
}

static Value *simplifyAndOfICmps(ICmpInst *Op0, ICmpInst *Op1,
                                 const SimplifyQuery &Q) {
  if (Value *X = simplifyUnsignedRangeCheck(Op0, Op1, /*IsAnd=*/true, Q))
    return X;
  if (Value *X = simplifyUnsignedRangeCheck(Op1, Op0, /*IsAnd=*/true, Q))
    return X;

  if (Value *X = simplifyAndOfICmpsWithSameOperands(Op0, Op1))
    return X;
  if (Value *X = simplifyAndOfICmpsWithSameOperands(Op1, Op0))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithConstants(Op0, Op1, true))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithLimitConst(Op0, Op1, true))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithZero(Op0, Op1, true))
    return X;

  if (Value *X = simplifyAndOfICmpsWithAdd(Op0, Op1, Q.IIQ))
    return X;
  if (Value *X = simplifyAndOfICmpsWithAdd(Op1, Op0, Q.IIQ))
    return X;

  return nullptr;
}

static Value *simplifyOrOfICmpsWithAdd(ICmpInst *Op0, ICmpInst *Op1,
                                       const InstrInfoQuery &IIQ) {
  // (icmp (add V, C0), C1) | (icmp V, C0)
  ICmpInst::Predicate Pred0, Pred1;
  const APInt *C0, *C1;
  Value *V;
  if (!match(Op0, m_ICmp(Pred0, m_Add(m_Value(V), m_APInt(C0)), m_APInt(C1))))
    return nullptr;

  if (!match(Op1, m_ICmp(Pred1, m_Specific(V), m_Value())))
    return nullptr;

  auto *AddInst = cast<BinaryOperator>(Op0->getOperand(0));
  if (AddInst->getOperand(1) != Op1->getOperand(1))
    return nullptr;

  Type *ITy = Op0->getType();
  bool isNSW = IIQ.hasNoSignedWrap(AddInst);
  bool isNUW = IIQ.hasNoUnsignedWrap(AddInst);

  const APInt Delta = *C1 - *C0;
  if (C0->isStrictlyPositive()) {
    if (Delta == 2) {
      if (Pred0 == ICmpInst::ICMP_UGE && Pred1 == ICmpInst::ICMP_SLE)
        return getTrue(ITy);
      if (Pred0 == ICmpInst::ICMP_SGE && Pred1 == ICmpInst::ICMP_SLE && isNSW)
        return getTrue(ITy);
    }
    if (Delta == 1) {
      if (Pred0 == ICmpInst::ICMP_UGT && Pred1 == ICmpInst::ICMP_SLE)
        return getTrue(ITy);
      if (Pred0 == ICmpInst::ICMP_SGT && Pred1 == ICmpInst::ICMP_SLE && isNSW)
        return getTrue(ITy);
    }
  }
  if (C0->getBoolValue() && isNUW) {
    if (Delta == 2)
      if (Pred0 == ICmpInst::ICMP_UGE && Pred1 == ICmpInst::ICMP_ULE)
        return getTrue(ITy);
    if (Delta == 1)
      if (Pred0 == ICmpInst::ICMP_UGT && Pred1 == ICmpInst::ICMP_ULE)
        return getTrue(ITy);
  }

  return nullptr;
}

static Value *simplifyOrOfICmps(ICmpInst *Op0, ICmpInst *Op1,
                                const SimplifyQuery &Q) {
  if (Value *X = simplifyUnsignedRangeCheck(Op0, Op1, /*IsAnd=*/false, Q))
    return X;
  if (Value *X = simplifyUnsignedRangeCheck(Op1, Op0, /*IsAnd=*/false, Q))
    return X;

  if (Value *X = simplifyOrOfICmpsWithSameOperands(Op0, Op1))
    return X;
  if (Value *X = simplifyOrOfICmpsWithSameOperands(Op1, Op0))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithConstants(Op0, Op1, false))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithLimitConst(Op0, Op1, false))
    return X;

  if (Value *X = simplifyAndOrOfICmpsWithZero(Op0, Op1, false))
    return X;

  if (Value *X = simplifyOrOfICmpsWithAdd(Op0, Op1, Q.IIQ))
    return X;
  if (Value *X = simplifyOrOfICmpsWithAdd(Op1, Op0, Q.IIQ))
    return X;

  return nullptr;
}

static Value *simplifyAndOrOfFCmps(const TargetLibraryInfo *TLI,
                                   FCmpInst *LHS, FCmpInst *RHS, bool IsAnd) {
  Value *LHS0 = LHS->getOperand(0), *LHS1 = LHS->getOperand(1);
  Value *RHS0 = RHS->getOperand(0), *RHS1 = RHS->getOperand(1);
  if (LHS0->getType() != RHS0->getType())
    return nullptr;

  FCmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();
  if ((PredL == FCmpInst::FCMP_ORD && PredR == FCmpInst::FCMP_ORD && IsAnd) ||
      (PredL == FCmpInst::FCMP_UNO && PredR == FCmpInst::FCMP_UNO && !IsAnd)) {
    // (fcmp ord NNAN, X) & (fcmp ord X, Y) --> fcmp ord X, Y
    // (fcmp ord NNAN, X) & (fcmp ord Y, X) --> fcmp ord Y, X
    // (fcmp ord X, NNAN) & (fcmp ord X, Y) --> fcmp ord X, Y
    // (fcmp ord X, NNAN) & (fcmp ord Y, X) --> fcmp ord Y, X
    // (fcmp uno NNAN, X) | (fcmp uno X, Y) --> fcmp uno X, Y
    // (fcmp uno NNAN, X) | (fcmp uno Y, X) --> fcmp uno Y, X
    // (fcmp uno X, NNAN) | (fcmp uno X, Y) --> fcmp uno X, Y
    // (fcmp uno X, NNAN) | (fcmp uno Y, X) --> fcmp uno Y, X
    if ((isKnownNeverNaN(LHS0, TLI) && (LHS1 == RHS0 || LHS1 == RHS1)) ||
        (isKnownNeverNaN(LHS1, TLI) && (LHS0 == RHS0 || LHS0 == RHS1)))
      return RHS;

    // (fcmp ord X, Y) & (fcmp ord NNAN, X) --> fcmp ord X, Y
    // (fcmp ord Y, X) & (fcmp ord NNAN, X) --> fcmp ord Y, X
    // (fcmp ord X, Y) & (fcmp ord X, NNAN) --> fcmp ord X, Y
    // (fcmp ord Y, X) & (fcmp ord X, NNAN) --> fcmp ord Y, X
    // (fcmp uno X, Y) | (fcmp uno NNAN, X) --> fcmp uno X, Y
    // (fcmp uno Y, X) | (fcmp uno NNAN, X) --> fcmp uno Y, X
    // (fcmp uno X, Y) | (fcmp uno X, NNAN) --> fcmp uno X, Y
    // (fcmp uno Y, X) | (fcmp uno X, NNAN) --> fcmp uno Y, X
    if ((isKnownNeverNaN(RHS0, TLI) && (RHS1 == LHS0 || RHS1 == LHS1)) ||
        (isKnownNeverNaN(RHS1, TLI) && (RHS0 == LHS0 || RHS0 == LHS1)))
      return LHS;
  }

  return nullptr;
}

static Value *simplifyAndOrOfCmps(const SimplifyQuery &Q,
                                  Value *Op0, Value *Op1, bool IsAnd) {
  // Look through casts of the 'and' operands to find compares.
  auto *Cast0 = dyn_cast<CastInst>(Op0);
  auto *Cast1 = dyn_cast<CastInst>(Op1);
  if (Cast0 && Cast1 && Cast0->getOpcode() == Cast1->getOpcode() &&
      Cast0->getSrcTy() == Cast1->getSrcTy()) {
    Op0 = Cast0->getOperand(0);
    Op1 = Cast1->getOperand(0);
  }

  Value *V = nullptr;
  auto *ICmp0 = dyn_cast<ICmpInst>(Op0);
  auto *ICmp1 = dyn_cast<ICmpInst>(Op1);
  if (ICmp0 && ICmp1)
    V = IsAnd ? simplifyAndOfICmps(ICmp0, ICmp1, Q)
              : simplifyOrOfICmps(ICmp0, ICmp1, Q);

  auto *FCmp0 = dyn_cast<FCmpInst>(Op0);
  auto *FCmp1 = dyn_cast<FCmpInst>(Op1);
  if (FCmp0 && FCmp1)
    V = simplifyAndOrOfFCmps(Q.TLI, FCmp0, FCmp1, IsAnd);

  if (!V)
    return nullptr;
  if (!Cast0)
    return V;

  // If we looked through casts, we can only handle a constant simplification
  // because we are not allowed to create a cast instruction here.
  if (auto *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(Cast0->getOpcode(), C, Cast0->getType());

  return nullptr;
}

/// Given a bitwise logic op, check if the operands are add/sub with a common
/// source value and inverted constant (identity: C - X -> ~(X + ~C)).
static Value *simplifyLogicOfAddSub(Value *Op0, Value *Op1,
                                    Instruction::BinaryOps Opcode) {
  assert(Op0->getType() == Op1->getType() && "Mismatched binop types");
  assert(BinaryOperator::isBitwiseLogicOp(Opcode) && "Expected logic op");
  Value *X;
  Constant *C1, *C2;
  if ((match(Op0, m_Add(m_Value(X), m_Constant(C1))) &&
       match(Op1, m_Sub(m_Constant(C2), m_Specific(X)))) ||
      (match(Op1, m_Add(m_Value(X), m_Constant(C1))) &&
       match(Op0, m_Sub(m_Constant(C2), m_Specific(X))))) {
    if (ConstantExpr::getNot(C1) == C2) {
      // (X + C) & (~C - X) --> (X + C) & ~(X + C) --> 0
      // (X + C) | (~C - X) --> (X + C) | ~(X + C) --> -1
      // (X + C) ^ (~C - X) --> (X + C) ^ ~(X + C) --> -1
      Type *Ty = Op0->getType();
      return Opcode == Instruction::And ? ConstantInt::getNullValue(Ty)
                                        : ConstantInt::getAllOnesValue(Ty);
    }
  }
  return nullptr;
}

/// Given operands for an And, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyAndInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                              unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::And, Op0, Op1, Q))
    return C;

  // X & poison -> poison
  if (isa<PoisonValue>(Op1))
    return Op1;

  // X & undef -> 0
  if (Q.isUndefValue(Op1))
    return Constant::getNullValue(Op0->getType());

  // X & X = X
  if (Op0 == Op1)
    return Op0;

  // X & 0 = 0
  if (match(Op1, m_Zero()))
    return Constant::getNullValue(Op0->getType());

  // X & -1 = X
  if (match(Op1, m_AllOnes()))
    return Op0;

  // A & ~A  =  ~A & A  =  0
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getNullValue(Op0->getType());

  // (A | ?) & A = A
  if (match(Op0, m_c_Or(m_Specific(Op1), m_Value())))
    return Op1;

  // A & (A | ?) = A
  if (match(Op1, m_c_Or(m_Specific(Op0), m_Value())))
    return Op0;

  // (X | Y) & (X | ~Y) --> X (commuted 8 ways)
  Value *X, *Y;
  if (match(Op0, m_c_Or(m_Value(X), m_Not(m_Value(Y)))) &&
      match(Op1, m_c_Or(m_Deferred(X), m_Deferred(Y))))
    return X;
  if (match(Op1, m_c_Or(m_Value(X), m_Not(m_Value(Y)))) &&
      match(Op0, m_c_Or(m_Deferred(X), m_Deferred(Y))))
    return X;

  if (Value *V = simplifyLogicOfAddSub(Op0, Op1, Instruction::And))
    return V;

  // A mask that only clears known zeros of a shifted value is a no-op.
  const APInt *Mask;
  const APInt *ShAmt;
  if (match(Op1, m_APInt(Mask))) {
    // If all bits in the inverted and shifted mask are clear:
    // and (shl X, ShAmt), Mask --> shl X, ShAmt
    if (match(Op0, m_Shl(m_Value(X), m_APInt(ShAmt))) &&
        (~(*Mask)).lshr(*ShAmt).isZero())
      return Op0;

    // If all bits in the inverted and shifted mask are clear:
    // and (lshr X, ShAmt), Mask --> lshr X, ShAmt
    if (match(Op0, m_LShr(m_Value(X), m_APInt(ShAmt))) &&
        (~(*Mask)).shl(*ShAmt).isZero())
      return Op0;
  }

  // If we have a multiplication overflow check that is being 'and'ed with a
  // check that one of the multipliers is not zero, we can omit the 'and', and
  // only keep the overflow check.
  if (isCheckForZeroAndMulWithOverflow(Op0, Op1, true))
    return Op1;
  if (isCheckForZeroAndMulWithOverflow(Op1, Op0, true))
    return Op0;

  // A & (-A) = A if A is a power of two or zero.
  if (match(Op0, m_Neg(m_Specific(Op1))) ||
      match(Op1, m_Neg(m_Specific(Op0)))) {
    if (isKnownToBeAPowerOfTwo(Op0, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI,
                               Q.DT))
      return Op0;
    if (isKnownToBeAPowerOfTwo(Op1, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI,
                               Q.DT))
      return Op1;
  }

  // This is a similar pattern used for checking if a value is a power-of-2:
  // (A - 1) & A --> 0 (if A is a power-of-2 or 0)
  // A & (A - 1) --> 0 (if A is a power-of-2 or 0)
  if (match(Op0, m_Add(m_Specific(Op1), m_AllOnes())) &&
      isKnownToBeAPowerOfTwo(Op1, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI, Q.DT))
    return Constant::getNullValue(Op1->getType());
  if (match(Op1, m_Add(m_Specific(Op0), m_AllOnes())) &&
      isKnownToBeAPowerOfTwo(Op0, Q.DL, /*OrZero*/ true, 0, Q.AC, Q.CxtI, Q.DT))
    return Constant::getNullValue(Op0->getType());

  if (Value *V = simplifyAndOrOfCmps(Q, Op0, Op1, true))
    return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::And, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // And distributes over Or.  Try some generic simplifications based on this.
  if (Value *V = expandCommutativeBinOp(Instruction::And, Op0, Op1,
                                        Instruction::Or, Q, MaxRecurse))
    return V;

  // And distributes over Xor.  Try some generic simplifications based on this.
  if (Value *V = expandCommutativeBinOp(Instruction::And, Op0, Op1,
                                        Instruction::Xor, Q, MaxRecurse))
    return V;

  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1)) {
    if (Op0->getType()->isIntOrIntVectorTy(1)) {
      // A & (A && B) -> A && B
      if (match(Op1, m_Select(m_Specific(Op0), m_Value(), m_Zero())))
        return Op1;
      else if (match(Op0, m_Select(m_Specific(Op1), m_Value(), m_Zero())))
        return Op0;
    }
    // If the operation is with the result of a select instruction, check
    // whether operating on either branch of the select always yields the same
    // value.
    if (Value *V = ThreadBinOpOverSelect(Instruction::And, Op0, Op1, Q,
                                         MaxRecurse))
      return V;
  }

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::And, Op0, Op1, Q,
                                      MaxRecurse))
      return V;

  // Assuming the effective width of Y is not larger than A, i.e. all bits
  // from X and Y are disjoint in (X << A) | Y,
  // if the mask of this AND op covers all bits of X or Y, while it covers
  // no bits from the other, we can bypass this AND op. E.g.,
  // ((X << A) | Y) & Mask -> Y,
  //     if Mask = ((1 << effective_width_of(Y)) - 1)
  // ((X << A) | Y) & Mask -> X << A,
  //     if Mask = ((1 << effective_width_of(X)) - 1) << A
  // SimplifyDemandedBits in InstCombine can optimize the general case.
  // This pattern aims to help other passes for a common case.
  Value *XShifted;
  if (match(Op1, m_APInt(Mask)) &&
      match(Op0, m_c_Or(m_CombineAnd(m_NUWShl(m_Value(X), m_APInt(ShAmt)),
                                     m_Value(XShifted)),
                        m_Value(Y)))) {
    const unsigned Width = Op0->getType()->getScalarSizeInBits();
    const unsigned ShftCnt = ShAmt->getLimitedValue(Width);
    const KnownBits YKnown = computeKnownBits(Y, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    const unsigned EffWidthY = YKnown.countMaxActiveBits();
    if (EffWidthY <= ShftCnt) {
      const KnownBits XKnown = computeKnownBits(X, Q.DL, 0, Q.AC, Q.CxtI,
                                                Q.DT);
      const unsigned EffWidthX = XKnown.countMaxActiveBits();
      const APInt EffBitsY = APInt::getLowBitsSet(Width, EffWidthY);
      const APInt EffBitsX = APInt::getLowBitsSet(Width, EffWidthX) << ShftCnt;
      // If the mask is extracting all bits from X or Y as is, we can skip
      // this AND op.
      if (EffBitsY.isSubsetOf(*Mask) && !EffBitsX.intersects(*Mask))
        return Y;
      if (EffBitsX.isSubsetOf(*Mask) && !EffBitsY.intersects(*Mask))
        return XShifted;
    }
  }

  // ((X | Y) ^ X ) & ((X | Y) ^ Y) --> 0
  // ((X | Y) ^ Y ) & ((X | Y) ^ X) --> 0
  BinaryOperator *Or;
  if (match(Op0, m_c_Xor(m_Value(X),
                         m_CombineAnd(m_BinOp(Or),
                                      m_c_Or(m_Deferred(X), m_Value(Y))))) &&
      match(Op1, m_c_Xor(m_Specific(Or), m_Specific(Y))))
    return Constant::getNullValue(Op0->getType());

  return nullptr;
}

Value *llvm::SimplifyAndInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyAndInst(Op0, Op1, Q, RecursionLimit);
}

static Value *simplifyOrLogic(Value *X, Value *Y) {
  assert(X->getType() == Y->getType() && "Expected same type for 'or' ops");
  Type *Ty = X->getType();

  // X | ~X --> -1
  if (match(Y, m_Not(m_Specific(X))))
    return ConstantInt::getAllOnesValue(Ty);

  // X | ~(X & ?) = -1
  if (match(Y, m_Not(m_c_And(m_Specific(X), m_Value()))))
    return ConstantInt::getAllOnesValue(Ty);

  // X | (X & ?) --> X
  if (match(Y, m_c_And(m_Specific(X), m_Value())))
    return X;

  Value *A, *B;

  // (A ^ B) | (A | B) --> A | B
  // (A ^ B) | (B | A) --> B | A
  if (match(X, m_Xor(m_Value(A), m_Value(B))) &&
      match(Y, m_c_Or(m_Specific(A), m_Specific(B))))
    return Y;

  // ~(A ^ B) | (A | B) --> -1
  // ~(A ^ B) | (B | A) --> -1
  if (match(X, m_Not(m_Xor(m_Value(A), m_Value(B)))) &&
      match(Y, m_c_Or(m_Specific(A), m_Specific(B))))
    return ConstantInt::getAllOnesValue(Ty);

  // (A & ~B) | (A ^ B) --> A ^ B
  // (~B & A) | (A ^ B) --> A ^ B
  // (A & ~B) | (B ^ A) --> B ^ A
  // (~B & A) | (B ^ A) --> B ^ A
  if (match(X, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Y, m_c_Xor(m_Specific(A), m_Specific(B))))
    return Y;

  // (~A ^ B) | (A & B) --> ~A ^ B
  // (B ^ ~A) | (A & B) --> B ^ ~A
  // (~A ^ B) | (B & A) --> ~A ^ B
  // (B ^ ~A) | (B & A) --> B ^ ~A
  if (match(X, m_c_Xor(m_Not(m_Value(A)), m_Value(B))) &&
      match(Y, m_c_And(m_Specific(A), m_Specific(B))))
    return X;

  // (~A | B) | (A ^ B) --> -1
  // (~A | B) | (B ^ A) --> -1
  // (B | ~A) | (A ^ B) --> -1
  // (B | ~A) | (B ^ A) --> -1
  if (match(X, m_c_Or(m_Not(m_Value(A)), m_Value(B))) &&
      match(Y, m_c_Xor(m_Specific(A), m_Specific(B))))
    return ConstantInt::getAllOnesValue(Ty);

  // (~A & B) | ~(A | B) --> ~A
  // (~A & B) | ~(B | A) --> ~A
  // (B & ~A) | ~(A | B) --> ~A
  // (B & ~A) | ~(B | A) --> ~A
  Value *NotA;
  if (match(X,
            m_c_And(m_CombineAnd(m_Value(NotA), m_NotForbidUndef(m_Value(A))),
                    m_Value(B))) &&
      match(Y, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))))
    return NotA;

  // ~(A ^ B) | (A & B) --> ~(A ^ B)
  // ~(A ^ B) | (B & A) --> ~(A ^ B)
  Value *NotAB;
  if (match(X, m_CombineAnd(m_NotForbidUndef(m_Xor(m_Value(A), m_Value(B))),
                            m_Value(NotAB))) &&
      match(Y, m_c_And(m_Specific(A), m_Specific(B))))
    return NotAB;

  // ~(A & B) | (A ^ B) --> ~(A & B)
  // ~(A & B) | (B ^ A) --> ~(A & B)
  if (match(X, m_CombineAnd(m_NotForbidUndef(m_And(m_Value(A), m_Value(B))),
                            m_Value(NotAB))) &&
      match(Y, m_c_Xor(m_Specific(A), m_Specific(B))))
    return NotAB;

  return nullptr;
}

/// Given operands for an Or, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyOrInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                             unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::Or, Op0, Op1, Q))
    return C;

  // X | poison -> poison
  if (isa<PoisonValue>(Op1))
    return Op1;

  // X | undef -> -1
  // X | -1 = -1
  // Do not return Op1 because it may contain undef elements if it's a vector.
  if (Q.isUndefValue(Op1) || match(Op1, m_AllOnes()))
    return Constant::getAllOnesValue(Op0->getType());

  // X | X = X
  // X | 0 = X
  if (Op0 == Op1 || match(Op1, m_Zero()))
    return Op0;

  if (Value *R = simplifyOrLogic(Op0, Op1))
    return R;
  if (Value *R = simplifyOrLogic(Op1, Op0))
    return R;

  if (Value *V = simplifyLogicOfAddSub(Op0, Op1, Instruction::Or))
    return V;

  // Rotated -1 is still -1:
  // (-1 << X) | (-1 >> (C - X)) --> -1
  // (-1 >> X) | (-1 << (C - X)) --> -1
  // ...with C <= bitwidth (and commuted variants).
  Value *X, *Y;
  if ((match(Op0, m_Shl(m_AllOnes(), m_Value(X))) &&
       match(Op1, m_LShr(m_AllOnes(), m_Value(Y)))) ||
      (match(Op1, m_Shl(m_AllOnes(), m_Value(X))) &&
       match(Op0, m_LShr(m_AllOnes(), m_Value(Y))))) {
    const APInt *C;
    if ((match(X, m_Sub(m_APInt(C), m_Specific(Y))) ||
         match(Y, m_Sub(m_APInt(C), m_Specific(X)))) &&
        C->ule(X->getType()->getScalarSizeInBits())) {
      return ConstantInt::getAllOnesValue(X->getType());
    }
  }

  if (Value *V = simplifyAndOrOfCmps(Q, Op0, Op1, false))
    return V;

  // If we have a multiplication overflow check that is being 'and'ed with a
  // check that one of the multipliers is not zero, we can omit the 'and', and
  // only keep the overflow check.
  if (isCheckForZeroAndMulWithOverflow(Op0, Op1, false))
    return Op1;
  if (isCheckForZeroAndMulWithOverflow(Op1, Op0, false))
    return Op0;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Or, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Or distributes over And.  Try some generic simplifications based on this.
  if (Value *V = expandCommutativeBinOp(Instruction::Or, Op0, Op1,
                                        Instruction::And, Q, MaxRecurse))
    return V;

  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1)) {
    if (Op0->getType()->isIntOrIntVectorTy(1)) {
      // A | (A || B) -> A || B
      if (match(Op1, m_Select(m_Specific(Op0), m_One(), m_Value())))
        return Op1;
      else if (match(Op0, m_Select(m_Specific(Op1), m_One(), m_Value())))
        return Op0;
    }
    // If the operation is with the result of a select instruction, check
    // whether operating on either branch of the select always yields the same
    // value.
    if (Value *V = ThreadBinOpOverSelect(Instruction::Or, Op0, Op1, Q,
                                         MaxRecurse))
      return V;
  }

  // (A & C1)|(B & C2)
  Value *A, *B;
  const APInt *C1, *C2;
  if (match(Op0, m_And(m_Value(A), m_APInt(C1))) &&
      match(Op1, m_And(m_Value(B), m_APInt(C2)))) {
    if (*C1 == ~*C2) {
      // (A & C1)|(B & C2)
      // If we have: ((V + N) & C1) | (V & C2)
      // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
      // replace with V+N.
      Value *N;
      if (C2->isMask() && // C2 == 0+1+
          match(A, m_c_Add(m_Specific(B), m_Value(N)))) {
        // Add commutes, try both ways.
        if (MaskedValueIsZero(N, *C2, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return A;
      }
      // Or commutes, try both ways.
      if (C1->isMask() &&
          match(B, m_c_Add(m_Specific(A), m_Value(N)))) {
        // Add commutes, try both ways.
        if (MaskedValueIsZero(N, *C1, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
          return B;
      }
    }
  }

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Or, Op0, Op1, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyOrInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyOrInst(Op0, Op1, Q, RecursionLimit);
}

/// Given operands for a Xor, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyXorInst(Value *Op0, Value *Op1, const SimplifyQuery &Q,
                              unsigned MaxRecurse) {
  if (Constant *C = foldOrCommuteConstant(Instruction::Xor, Op0, Op1, Q))
    return C;

  // X ^ poison -> poison
  if (isa<PoisonValue>(Op1))
    return Op1;

  // A ^ undef -> undef
  if (Q.isUndefValue(Op1))
    return Op1;

  // A ^ 0 = A
  if (match(Op1, m_Zero()))
    return Op0;

  // A ^ A = 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // A ^ ~A  =  ~A ^ A  =  -1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  auto foldAndOrNot = [](Value *X, Value *Y) -> Value * {
    Value *A, *B;
    // (~A & B) ^ (A | B) --> A -- There are 8 commuted variants.
    if (match(X, m_c_And(m_Not(m_Value(A)), m_Value(B))) &&
        match(Y, m_c_Or(m_Specific(A), m_Specific(B))))
      return A;

    // (~A | B) ^ (A & B) --> ~A -- There are 8 commuted variants.
    // The 'not' op must contain a complete -1 operand (no undef elements for
    // vector) for the transform to be safe.
    Value *NotA;
    if (match(X,
              m_c_Or(m_CombineAnd(m_NotForbidUndef(m_Value(A)), m_Value(NotA)),
                     m_Value(B))) &&
        match(Y, m_c_And(m_Specific(A), m_Specific(B))))
      return NotA;

    return nullptr;
  };
  if (Value *R = foldAndOrNot(Op0, Op1))
    return R;
  if (Value *R = foldAndOrNot(Op1, Op0))
    return R;

  if (Value *V = simplifyLogicOfAddSub(Op0, Op1, Instruction::Xor))
    return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Xor, Op0, Op1, Q,
                                          MaxRecurse))
    return V;

  // Threading Xor over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A ^ select(cond, B, C)" means evaluating
  // "A^B" and "A^C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A^B" and "A^C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return nullptr;
}

Value *llvm::SimplifyXorInst(Value *Op0, Value *Op1, const SimplifyQuery &Q) {
  return ::SimplifyXorInst(Op0, Op1, Q, RecursionLimit);
}


static Type *GetCompareTy(Value *Op) {
  return CmpInst::makeCmpResultType(Op->getType());
}

/// Rummage around inside V looking for something equivalent to the comparison
/// "LHS Pred RHS". Return such a value if found, otherwise return null.
/// Helper function for analyzing max/min idioms.
static Value *ExtractEquivalentCondition(Value *V, CmpInst::Predicate Pred,
                                         Value *LHS, Value *RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI)
    return nullptr;
  CmpInst *Cmp = dyn_cast<CmpInst>(SI->getCondition());
  if (!Cmp)
    return nullptr;
  Value *CmpLHS = Cmp->getOperand(0), *CmpRHS = Cmp->getOperand(1);
  if (Pred == Cmp->getPredicate() && LHS == CmpLHS && RHS == CmpRHS)
    return Cmp;
  if (Pred == CmpInst::getSwappedPredicate(Cmp->getPredicate()) &&
      LHS == CmpRHS && RHS == CmpLHS)
    return Cmp;
  return nullptr;
}

// A significant optimization not implemented here is assuming that alloca
// addresses are not equal to incoming argument values. They don't *alias*,
// as we say, but that doesn't mean they aren't equal, so we take a
// conservative approach.
//
// This is inspired in part by C++11 5.10p1:
//   "Two pointers of the same type compare equal if and only if they are both
//    null, both point to the same function, or both represent the same
//    address."
//
// This is pretty permissive.
//
// It's also partly due to C11 6.5.9p6:
//   "Two pointers compare equal if and only if both are null pointers, both are
//    pointers to the same object (including a pointer to an object and a
//    subobject at its beginning) or function, both are pointers to one past the
//    last element of the same array object, or one is a pointer to one past the
//    end of one array object and the other is a pointer to the start of a
//    different array object that happens to immediately follow the first array
//    object in the address space.)
//
// C11's version is more restrictive, however there's no reason why an argument
// couldn't be a one-past-the-end value for a stack object in the caller and be
// equal to the beginning of a stack object in the callee.
//
// If the C and C++ standards are ever made sufficiently restrictive in this
// area, it may be possible to update LLVM's semantics accordingly and reinstate
// this optimization.
static Constant *
computePointerICmp(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                   const SimplifyQuery &Q) {
  const DataLayout &DL = Q.DL;
  const TargetLibraryInfo *TLI = Q.TLI;
  const DominatorTree *DT = Q.DT;
  const Instruction *CxtI = Q.CxtI;
  const InstrInfoQuery &IIQ = Q.IIQ;

  // First, skip past any trivial no-ops.
  LHS = LHS->stripPointerCasts();
  RHS = RHS->stripPointerCasts();

  // A non-null pointer is not equal to a null pointer.
  if (isa<ConstantPointerNull>(RHS) && ICmpInst::isEquality(Pred) &&
      llvm::isKnownNonZero(LHS, DL, 0, nullptr, nullptr, nullptr,
                           IIQ.UseInstrInfo))
    return ConstantInt::get(GetCompareTy(LHS),
                            !CmpInst::isTrueWhenEqual(Pred));

  // We can only fold certain predicates on pointer comparisons.
  switch (Pred) {
  default:
    return nullptr;

    // Equality comaprisons are easy to fold.
  case CmpInst::ICMP_EQ:
  case CmpInst::ICMP_NE:
    break;

    // We can only handle unsigned relational comparisons because 'inbounds' on
    // a GEP only protects against unsigned wrapping.
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
    // However, we have to switch them to their signed variants to handle
    // negative indices from the base pointer.
    Pred = ICmpInst::getSignedPredicate(Pred);
    break;
  }

  // Strip off any constant offsets so that we can reason about them.
  // It's tempting to use getUnderlyingObject or even just stripInBoundsOffsets
  // here and compare base addresses like AliasAnalysis does, however there are
  // numerous hazards. AliasAnalysis and its utilities rely on special rules
  // governing loads and stores which don't apply to icmps. Also, AliasAnalysis
  // doesn't need to guarantee pointer inequality when it says NoAlias.
  Constant *LHSOffset = stripAndComputeConstantOffsets(DL, LHS);
  Constant *RHSOffset = stripAndComputeConstantOffsets(DL, RHS);

  // If LHS and RHS are related via constant offsets to the same base
  // value, we can replace it with an icmp which just compares the offsets.
  if (LHS == RHS)
    return ConstantExpr::getICmp(Pred, LHSOffset, RHSOffset);

  // Various optimizations for (in)equality comparisons.
  if (Pred == CmpInst::ICMP_EQ || Pred == CmpInst::ICMP_NE) {
    // Different non-empty allocations that exist at the same time have
    // different addresses (if the program can tell). Global variables always
    // exist, so they always exist during the lifetime of each other and all
    // allocas. Two different allocas usually have different addresses...
    //
    // However, if there's an @llvm.stackrestore dynamically in between two
    // allocas, they may have the same address. It's tempting to reduce the
    // scope of the problem by only looking at *static* allocas here. That would
    // cover the majority of allocas while significantly reducing the likelihood
    // of having an @llvm.stackrestore pop up in the middle. However, it's not
    // actually impossible for an @llvm.stackrestore to pop up in the middle of
    // an entry block. Also, if we have a block that's not attached to a
    // function, we can't tell if it's "static" under the current definition.
    // Theoretically, this problem could be fixed by creating a new kind of
    // instruction kind specifically for static allocas. Such a new instruction
    // could be required to be at the top of the entry block, thus preventing it
    // from being subject to a @llvm.stackrestore. Instcombine could even
    // convert regular allocas into these special allocas. It'd be nifty.
    // However, until then, this problem remains open.
    //
    // So, we'll assume that two non-empty allocas have different addresses
    // for now.
    //
    // With all that, if the offsets are within the bounds of their allocations
    // (and not one-past-the-end! so we can't use inbounds!), and their
    // allocations aren't the same, the pointers are not equal.
    //
    // Note that it's not necessary to check for LHS being a global variable
    // address, due to canonicalization and constant folding.
    if (isa<AllocaInst>(LHS) &&
        (isa<AllocaInst>(RHS) || isa<GlobalVariable>(RHS))) {
      ConstantInt *LHSOffsetCI = dyn_cast<ConstantInt>(LHSOffset);
      ConstantInt *RHSOffsetCI = dyn_cast<ConstantInt>(RHSOffset);
      uint64_t LHSSize, RHSSize;
      ObjectSizeOpts Opts;
      Opts.NullIsUnknownSize =
          NullPointerIsDefined(cast<AllocaInst>(LHS)->getFunction());
      if (LHSOffsetCI && RHSOffsetCI &&
          getObjectSize(LHS, LHSSize, DL, TLI, Opts) &&
          getObjectSize(RHS, RHSSize, DL, TLI, Opts)) {
        const APInt &LHSOffsetValue = LHSOffsetCI->getValue();
        const APInt &RHSOffsetValue = RHSOffsetCI->getValue();
        if (!LHSOffsetValue.isNegative() &&
            !RHSOffsetValue.isNegative() &&
            LHSOffsetValue.ult(LHSSize) &&
            RHSOffsetValue.ult(RHSSize)) {
          return ConstantInt::get(GetCompareTy(LHS),
                                  !CmpInst::isTrueWhenEqual(Pred));
        }
      }

      // Repeat the above check but this time without depending on DataLayout
      // or being able to compute a precise size.
      if (!cast<PointerType>(LHS->getType())->isEmptyTy() &&
          !cast<PointerType>(RHS->getType())->isEmptyTy() &&
          LHSOffset->isNullValue() &&
          RHSOffset->isNullValue())
        return ConstantInt::get(GetCompareTy(LHS),
                                !CmpInst::isTrueWhenEqual(Pred));
    }

    // Even if an non-inbounds GEP occurs along the path we can still optimize
    // equality comparisons concerning the result. We avoid walking the whole
    // chain again by starting where the last calls to
    // stripAndComputeConstantOffsets left off and accumulate the offsets.
    Constant *LHSNoBound = stripAndComputeConstantOffsets(DL, LHS, true);
    Constant *RHSNoBound = stripAndComputeConstantOffsets(DL, RHS, true);
    if (LHS == RHS)
      return ConstantExpr::getICmp(Pred,
                                   ConstantExpr::getAdd(LHSOffset, LHSNoBound),
                                   ConstantExpr::getAdd(RHSOffset, RHSNoBound));

    // If one side of the equality comparison must come from a noalias call
    // (meaning a system memory allocation function), and the other side must
    // come from a pointer that cannot overlap with dynamically-allocated
    // memory within the lifetime of the current function (allocas, byval
    // arguments, globals), then determine the comparison result here.
    SmallVector<const Value *, 8> LHSUObjs, RHSUObjs;
    getUnderlyingObjects(LHS, LHSUObjs);
    getUnderlyingObjects(RHS, RHSUObjs);

    // Is the set of underlying objects all noalias calls?
    auto IsNAC = [](ArrayRef<const Value *> Objects) {
      return all_of(Objects, isNoAliasCall);
    };

    // Is the set of underlying objects all things which must be disjoint from
    // noalias calls. For allocas, we consider only static ones (dynamic
    // allocas might be transformed into calls to malloc not simultaneously
    // live with the compared-to allocation). For globals, we exclude symbols
    // that might be resolve lazily to symbols in another dynamically-loaded
    // library (and, thus, could be malloc'ed by the implementation).
    auto IsAllocDisjoint = [](ArrayRef<const Value *> Objects) {
      return all_of(Objects, [](const Value *V) {
        if (const AllocaInst *AI = dyn_cast<AllocaInst>(V))
          return AI->getParent() && AI->getFunction() && AI->isStaticAlloca();
        if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
          return (GV->hasLocalLinkage() || GV->hasHiddenVisibility() ||
                  GV->hasProtectedVisibility() || GV->hasGlobalUnnamedAddr()) &&
                 !GV->isThreadLocal();
        if (const Argument *A = dyn_cast<Argument>(V))
          return A->hasByValAttr();
        return false;
      });
    };

    if ((IsNAC(LHSUObjs) && IsAllocDisjoint(RHSUObjs)) ||
        (IsNAC(RHSUObjs) && IsAllocDisjoint(LHSUObjs)))
        return ConstantInt::get(GetCompareTy(LHS),
                                !CmpInst::isTrueWhenEqual(Pred));

    // Fold comparisons for non-escaping pointer even if the allocation call
    // cannot be elided. We cannot fold malloc comparison to null. Also, the
    // dynamic allocation call could be either of the operands.  Note that
    // the other operand can not be based on the alloc - if it were, then
    // the cmp itself would be a capture.
    Value *MI = nullptr;
    if (isAllocLikeFn(LHS, TLI) &&
        llvm::isKnownNonZero(RHS, DL, 0, nullptr, CxtI, DT))
      MI = LHS;
    else if (isAllocLikeFn(RHS, TLI) &&
             llvm::isKnownNonZero(LHS, DL, 0, nullptr, CxtI, DT))
      MI = RHS;
    // FIXME: We should also fold the compare when the pointer escapes, but the
    // compare dominates the pointer escape
    if (MI && !PointerMayBeCaptured(MI, true, true))
      return ConstantInt::get(GetCompareTy(LHS),
                              CmpInst::isFalseWhenEqual(Pred));
  }

  // Otherwise, fail.
  return nullptr;
}

/// Fold an icmp when its operands have i1 scalar type.
static Value *simplifyICmpOfBools(CmpInst::Predicate Pred, Value *LHS,
                                  Value *RHS, const SimplifyQuery &Q) {
  Type *ITy = GetCompareTy(LHS); // The return type.
  Type *OpTy = LHS->getType();   // The operand type.
  if (!OpTy->isIntOrIntVectorTy(1))
    return nullptr;

  // A boolean compared to true/false can be reduced in 14 out of the 20
  // (10 predicates * 2 constants) possible combinations. The other
  // 6 cases require a 'not' of the LHS.

  auto ExtractNotLHS = [](Value *V) -> Value * {
    Value *X;
    if (match(V, m_Not(m_Value(X))))
      return X;
    return nullptr;
  };

  if (match(RHS, m_Zero())) {
    switch (Pred) {
    case CmpInst::ICMP_NE:  // X !=  0 -> X
    case CmpInst::ICMP_UGT: // X >u  0 -> X
    case CmpInst::ICMP_SLT: // X <s  0 -> X
      return LHS;

    case CmpInst::ICMP_EQ:  // not(X) ==  0 -> X != 0 -> X
    case CmpInst::ICMP_ULE: // not(X) <=u 0 -> X >u 0 -> X
    case CmpInst::ICMP_SGE: // not(X) >=s 0 -> X <s 0 -> X
      if (Value *X = ExtractNotLHS(LHS))
        return X;
      break;

    case CmpInst::ICMP_ULT: // X <u  0 -> false
    case CmpInst::ICMP_SGT: // X >s  0 -> false
      return getFalse(ITy);

    case CmpInst::ICMP_UGE: // X >=u 0 -> true
    case CmpInst::ICMP_SLE: // X <=s 0 -> true
      return getTrue(ITy);

    default: break;
    }
  } else if (match(RHS, m_One())) {
    switch (Pred) {
    case CmpInst::ICMP_EQ:  // X ==   1 -> X
    case CmpInst::ICMP_UGE: // X >=u  1 -> X
    case CmpInst::ICMP_SLE: // X <=s -1 -> X
      return LHS;

    case CmpInst::ICMP_NE:  // not(X) !=  1 -> X ==   1 -> X
    case CmpInst::ICMP_ULT: // not(X) <=u 1 -> X >=u  1 -> X
    case CmpInst::ICMP_SGT: // not(X) >s  1 -> X <=s -1 -> X
      if (Value *X = ExtractNotLHS(LHS))
        return X;
      break;

    case CmpInst::ICMP_UGT: // X >u   1 -> false
    case CmpInst::ICMP_SLT: // X <s  -1 -> false
      return getFalse(ITy);

    case CmpInst::ICMP_ULE: // X <=u  1 -> true
    case CmpInst::ICMP_SGE: // X >=s -1 -> true
      return getTrue(ITy);

    default: break;
    }
  }

  switch (Pred) {
  default:
    break;
  case ICmpInst::ICMP_UGE:
    if (isImpliedCondition(RHS, LHS, Q.DL).getValueOr(false))
      return getTrue(ITy);
    break;
  case ICmpInst::ICMP_SGE:
    /// For signed comparison, the values for an i1 are 0 and -1
    /// respectively. This maps into a truth table of:
    /// LHS | RHS | LHS >=s RHS   | LHS implies RHS
    ///  0  |  0  |  1 (0 >= 0)   |  1
    ///  0  |  1  |  1 (0 >= -1)  |  1
    ///  1  |  0  |  0 (-1 >= 0)  |  0
    ///  1  |  1  |  1 (-1 >= -1) |  1
    if (isImpliedCondition(LHS, RHS, Q.DL).getValueOr(false))
      return getTrue(ITy);
    break;
  case ICmpInst::ICMP_ULE:
    if (isImpliedCondition(LHS, RHS, Q.DL).getValueOr(false))
      return getTrue(ITy);
    break;
  }

  return nullptr;
}

/// Try hard to fold icmp with zero RHS because this is a common case.
static Value *simplifyICmpWithZero(CmpInst::Predicate Pred, Value *LHS,
                                   Value *RHS, const SimplifyQuery &Q) {
  if (!match(RHS, m_Zero()))
    return nullptr;

  Type *ITy = GetCompareTy(LHS); // The return type.
  switch (Pred) {
  default:
    llvm_unreachable("Unknown ICmp predicate!");
  case ICmpInst::ICMP_ULT:
    return getFalse(ITy);
  case ICmpInst::ICMP_UGE:
    return getTrue(ITy);
  case ICmpInst::ICMP_EQ:
  case ICmpInst::ICMP_ULE:
    if (isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT, Q.IIQ.UseInstrInfo))
      return getFalse(ITy);
    break;
  case ICmpInst::ICMP_NE:
  case ICmpInst::ICMP_UGT:
    if (isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT, Q.IIQ.UseInstrInfo))
      return getTrue(ITy);
    break;
  case ICmpInst::ICMP_SLT: {
    KnownBits LHSKnown = computeKnownBits(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    if (LHSKnown.isNegative())
      return getTrue(ITy);
    if (LHSKnown.isNonNegative())
      return getFalse(ITy);
    break;
  }
  case ICmpInst::ICMP_SLE: {
    KnownBits LHSKnown = computeKnownBits(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    if (LHSKnown.isNegative())
      return getTrue(ITy);
    if (LHSKnown.isNonNegative() &&
        isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
      return getFalse(ITy);
    break;
  }
  case ICmpInst::ICMP_SGE: {
    KnownBits LHSKnown = computeKnownBits(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    if (LHSKnown.isNegative())
      return getFalse(ITy);
    if (LHSKnown.isNonNegative())
      return getTrue(ITy);
    break;
  }
  case ICmpInst::ICMP_SGT: {
    KnownBits LHSKnown = computeKnownBits(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
    if (LHSKnown.isNegative())
      return getFalse(ITy);
    if (LHSKnown.isNonNegative() &&
        isKnownNonZero(LHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
      return getTrue(ITy);
    break;
  }
  }

  return nullptr;
}

static Value *simplifyICmpWithConstant(CmpInst::Predicate Pred, Value *LHS,
                                       Value *RHS, const InstrInfoQuery &IIQ) {
  Type *ITy = GetCompareTy(RHS); // The return type.

  Value *X;
  // Sign-bit checks can be optimized to true/false after unsigned
  // floating-point casts:
  // icmp slt (bitcast (uitofp X)),  0 --> false
  // icmp sgt (bitcast (uitofp X)), -1 --> true
  if (match(LHS, m_BitCast(m_UIToFP(m_Value(X))))) {
    if (Pred == ICmpInst::ICMP_SLT && match(RHS, m_Zero()))
      return ConstantInt::getFalse(ITy);
    if (Pred == ICmpInst::ICMP_SGT && match(RHS, m_AllOnes()))
      return ConstantInt::getTrue(ITy);
  }

  const APInt *C;
  if (!match(RHS, m_APIntAllowUndef(C)))
    return nullptr;

  // Rule out tautological comparisons (eg., ult 0 or uge 0).
  ConstantRange RHS_CR = ConstantRange::makeExactICmpRegion(Pred, *C);
  if (RHS_CR.isEmptySet())
    return ConstantInt::getFalse(ITy);
  if (RHS_CR.isFullSet())
    return ConstantInt::getTrue(ITy);

  ConstantRange LHS_CR =
      computeConstantRange(LHS, CmpInst::isSigned(Pred), IIQ.UseInstrInfo);
  if (!LHS_CR.isFullSet()) {
    if (RHS_CR.contains(LHS_CR))
      return ConstantInt::getTrue(ITy);
    if (RHS_CR.inverse().contains(LHS_CR))
      return ConstantInt::getFalse(ITy);
  }

  // (mul nuw/nsw X, MulC) != C --> true  (if C is not a multiple of MulC)
  // (mul nuw/nsw X, MulC) == C --> false (if C is not a multiple of MulC)
  const APInt *MulC;
  if (ICmpInst::isEquality(Pred) &&
      ((match(LHS, m_NUWMul(m_Value(), m_APIntAllowUndef(MulC))) &&
        *MulC != 0 && C->urem(*MulC) != 0) ||
       (match(LHS, m_NSWMul(m_Value(), m_APIntAllowUndef(MulC))) &&
        *MulC != 0 && C->srem(*MulC) != 0)))
    return ConstantInt::get(ITy, Pred == ICmpInst::ICMP_NE);

  return nullptr;
}

static Value *simplifyICmpWithBinOpOnLHS(
    CmpInst::Predicate Pred, BinaryOperator *LBO, Value *RHS,
    const SimplifyQuery &Q, unsigned MaxRecurse) {
  Type *ITy = GetCompareTy(RHS); // The return type.

  Value *Y = nullptr;
  // icmp pred (or X, Y), X
  if (match(LBO, m_c_Or(m_Value(Y), m_Specific(RHS)))) {
    if (Pred == ICmpInst::ICMP_ULT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_UGE)
      return getTrue(ITy);

    if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SGE) {
      KnownBits RHSKnown = computeKnownBits(RHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
      KnownBits YKnown = computeKnownBits(Y, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
      if (RHSKnown.isNonNegative() && YKnown.isNegative())
        return Pred == ICmpInst::ICMP_SLT ? getTrue(ITy) : getFalse(ITy);
      if (RHSKnown.isNegative() || YKnown.isNonNegative())
        return Pred == ICmpInst::ICMP_SLT ? getFalse(ITy) : getTrue(ITy);
    }
  }

  // icmp pred (and X, Y), X
  if (match(LBO, m_c_And(m_Value(), m_Specific(RHS)))) {
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
  }

  // icmp pred (urem X, Y), Y
  if (match(LBO, m_URem(m_Value(), m_Specific(RHS)))) {
    switch (Pred) {
    default:
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: {
      KnownBits Known = computeKnownBits(RHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
      if (!Known.isNonNegative())
        break;
      LLVM_FALLTHROUGH;
    }
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
      return getFalse(ITy);
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: {
      KnownBits Known = computeKnownBits(RHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT);
      if (!Known.isNonNegative())
        break;
      LLVM_FALLTHROUGH;
    }
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
      return getTrue(ITy);
    }
  }

  // icmp pred (urem X, Y), X
  if (match(LBO, m_URem(m_Specific(RHS), m_Value()))) {
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
  }

  // x >>u y <=u x --> true.
  // x >>u y >u  x --> false.
  // x udiv y <=u x --> true.
  // x udiv y >u  x --> false.
  if (match(LBO, m_LShr(m_Specific(RHS), m_Value())) ||
      match(LBO, m_UDiv(m_Specific(RHS), m_Value()))) {
    // icmp pred (X op Y), X
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
  }

  // If x is nonzero:
  // x >>u C <u  x --> true  for C != 0.
  // x >>u C !=  x --> true  for C != 0.
  // x >>u C >=u x --> false for C != 0.
  // x >>u C ==  x --> false for C != 0.
  // x udiv C <u  x --> true  for C != 1.
  // x udiv C !=  x --> true  for C != 1.
  // x udiv C >=u x --> false for C != 1.
  // x udiv C ==  x --> false for C != 1.
  // TODO: allow non-constant shift amount/divisor
  const APInt *C;
  if ((match(LBO, m_LShr(m_Specific(RHS), m_APInt(C))) && *C != 0) ||
      (match(LBO, m_UDiv(m_Specific(RHS), m_APInt(C))) && *C != 1)) {
    if (isKnownNonZero(RHS, Q.DL, 0, Q.AC, Q.CxtI, Q.DT)) {
      switch (Pred) {
      default:
        break;
      case ICmpInst::ICMP_EQ:
      case ICmpInst::ICMP_UGE:
        return getFalse(ITy);
      case ICmpInst::ICMP_NE:
      case ICmpInst::ICMP_ULT:
        return getTrue(ITy);
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_ULE:
        // UGT/ULE are handled by the more general case just above
        llvm_unreachable("Unexpected UGT/ULE, should have been handled");
      }
    }
  }

  // (x*C1)/C2 <= x for C1 <= C2.
  // This holds even if the multiplication overflows: Assume that x != 0 and
  // arithmetic is modulo M. For overflow to occur we must have C1 >= M/x and
  // thus C2 >= M/x. It follows that (x*C1)/C2 <= (M-1)/C2 <= ((M-1)*x)/M < x.
  //
  // Additionally, either the multiplication and division might be represented
  // as shifts:
  // (x*C1)>>C2 <= x for C1 < 2**C2.
  // (x<<C1)/C2 <= x for 2**C1 < C2.
  const APInt *C1, *C2;
  if ((match(LBO, m_UDiv(m_Mul(m_Specific(RHS), m_APInt(C1)), m_APInt(C2))) &&
       C1->ule(*C2)) ||
      (match(LBO, m_LShr(m_Mul(m_Specific(RHS), m_APInt(C1)), m_APInt(C2))) &&
       C1->ule(APInt(C2->getBitWidth(), 1) << *C2)) ||
      (match(LBO, m_UDiv(m_Shl(m_Specific(RHS), m_APInt(C1)), m_APInt(C2))) &&
       (APInt(C1->getBitWidth(), 1) << *C1).ule(*C2))) {
    if (Pred == ICmpInst::ICMP_UGT)
      return getFalse(ITy);
    if (Pred == ICmpInst::ICMP_ULE)
      return getTrue(ITy);
  }

  return nullptr;
}


// If only one of the icmp's operands has NSW flags, try to prove that:
//
//   icmp slt (x + C1), (x +nsw C2)
//
// is equivalent to:
//
//   icmp slt C1, C2
//
// which is true if x + C2 has the NSW flags set and:
// *) C1 < C2 && C1 >= 0, or
// *) C2 < C1 && C1 <= 0.
//
static bool trySimplifyICmpWithAdds(CmpInst::Predicate Pred, Value *LHS,
                                    Value *RHS) {
  // TODO: only support icmp slt for now.
  if (Pred != CmpInst::ICMP_SLT)
    return false;

  // Canonicalize nsw add as RHS.
  if (!match(RHS, m_NSWAdd(m_Value(), m_Value())))
    std::swap(LHS, RHS);
  if (!match(RHS, m_NSWAdd(m_Value(), m_Value())))
    return false;

  Value *X;
  const APInt *C1, *C2;
  if (!match(LHS, m_c_Add(m_Value(X), m_APInt(C1))) ||
      !match(RHS, m_c_Add(m_Specific(X), m_APInt(C2))))
    return false;

  return (C1->slt(*C2) && C1->isNonNegative()) ||
         (C2->slt(*C1) && C1->isNonPositive());
}


/// TODO: A large part of this logic is duplicated in InstCombine's
/// foldICmpBinOp(). We should be able to share that and avoid the code
/// duplication.
static Value *simplifyICmpWithBinOp(CmpInst::Predicate Pred, Value *LHS,
                                    Value *RHS, const SimplifyQuery &Q,
                                    unsigned MaxRecurse) {
  BinaryOperator *LBO = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *RBO = dyn_cast<BinaryOperator>(RHS);
  if (MaxRecurse && (LBO || RBO)) {
    // Analyze the case when either LHS or RHS is an add instruction.
    Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
    // LHS = A + B (or A and B are null); RHS = C + D (or C and D are null).
    bool NoLHSWrapProblem = false, NoRHSWrapProblem = false;
    if (LBO && LBO->getOpcode() == Instruction::Add) {
      A = LBO->getOperand(0);
      B = LBO->getOperand(1);
      NoLHSWrapProblem =
          ICmpInst::isEquality(Pred) ||
          (CmpInst::isUnsigned(Pred) &&
           Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(LBO))) ||
          (CmpInst::isSigned(Pred) &&
           Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(LBO)));
    }
    if (RBO && RBO->getOpcode() == Instruction::Add) {
      C = RBO->getOperand(0);
      D = RBO->getOperand(1);
      NoRHSWrapProblem =
          ICmpInst::isEquality(Pred) ||
          (CmpInst::isUnsigned(Pred) &&
           Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(RBO))) ||
          (CmpInst::isSigned(Pred) &&
           Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(RBO)));
    }

    // icmp (X+Y), X -> icmp Y, 0 for equalities or if there is no overflow.
    if ((A == RHS || B == RHS) && NoLHSWrapProblem)
      if (Value *V = SimplifyICmpInst(Pred, A == RHS ? B : A,
                                      Constant::getNullValue(RHS->getType()), Q,
                                      MaxRecurse - 1))
        return V;

    // icmp X, (X+Y) -> icmp 0, Y for equalities or if there is no overflow.
    if ((C == LHS || D == LHS) && NoRHSWrapProblem)
      if (Value *V =
              SimplifyICmpInst(Pred, Constant::getNullValue(LHS->getType()),
                               C == LHS ? D : C, Q, MaxRecurse - 1))
        return V;

    // icmp (X+Y), (X+Z) -> icmp Y,Z for equalities or if there is no overflow.
    bool CanSimplify = (NoLHSWrapProblem && NoRHSWrapProblem) ||
                       trySimplifyICmpWithAdds(Pred, LHS, RHS);
    if (A && C && (A == C || A == D || B == C || B == D) && CanSimplify) {
      // Determine Y and Z in the form icmp (X+Y), (X+Z).
      Value *Y, *Z;
      if (A == C) {
        // C + B == C + D  ->  B == D
        Y = B;
        Z = D;
      } else if (A == D) {
        // D + B == C + D  ->  B == C
        Y = B;
        Z = C;
      } else if (B == C) {
        // A + C == C + D  ->  A == D
        Y = A;
        Z = D;
      } else {
        assert(B == D);
        // A + D == C + D  ->  A == C
        Y = A;
        Z = C;
      }
      if (Value *V = SimplifyICmpInst(Pred, Y, Z, Q, MaxRecurse - 1))
        return V;
    }
  }

  if (LBO)
    if (Value *V = simplifyICmpWithBinOpOnLHS(Pred, LBO, RHS, Q, MaxRecurse))
      return V;

  if (RBO)
    if (Value *V = simplifyICmpWithBinOpOnLHS(
            ICmpInst::getSwappedPredicate(Pred), RBO, LHS, Q, MaxRecurse))
      return V;

  // 0 - (zext X) pred C
  if (!CmpInst::isUnsigned(Pred) && match(LHS, m_Neg(m_ZExt(m_Value())))) {
    const APInt *C;
    if (match(RHS, m_APInt(C))) {
      if (C->isStrictlyPositive()) {
        if (Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_NE)
          return ConstantInt::getTrue(GetCompareTy(RHS));
        if (Pred == ICmpInst::ICMP_SGE || Pred == ICmpInst::ICMP_EQ)
          return ConstantInt::getFalse(GetCompareTy(RHS));
      }
      if (C->isNonNegative()) {
        if (Pred == ICmpInst::ICMP_SLE)
          return ConstantInt::getTrue(GetCompareTy(RHS));
        if (Pred == ICmpInst::ICMP_SGT)
          return ConstantInt::getFalse(GetCompareTy(RHS));
      }
    }
  }

  //   If C2 is a power-of-2 and C is not:
  //   (C2 << X) == C --> false
  //   (C2 << X) != C --> true
  const APInt *C;
  if (match(LHS, m_Shl(m_Power2(), m_Value())) &&
      match(RHS, m_APIntAllowUndef(C)) && !C->isPowerOf2()) {
    // C2 << X can equal zero in some circumstances.
    // This simplification might be unsafe if C is zero.
    //
    // We know it is safe if:
    // - The shift is nsw. We can't shift out the one bit.
    // - The shift is nuw. We can't shift out the one bit.
    // - C2 is one.
    // - C isn't zero.
    if (Q.IIQ.hasNoSignedWrap(cast<OverflowingBinaryOperator>(LBO)) ||
        Q.IIQ.hasNoUnsignedWrap(cast<OverflowingBinaryOperator>(LBO)) ||
        match(LHS, m_Shl(m_One(), m_Value())) || !C->isZero()) {
      if (Pred == ICmpInst::ICMP_EQ)
        return ConstantInt::getFalse(GetCompareTy(RHS));
      if (Pred == ICmpInst::ICMP_NE)
        return ConstantInt::getTrue(GetCompareTy(RHS));
    }
  }

  // TODO: This is overly constrained. LHS can be any power-of-2.
  // (1 << X)  >u 0x8000 --> false
  // (1 << X) <=u 0x8000 --> true
  if (match(LHS, m_Shl(m_One(), m_Value())) && match(RHS, m_SignMask())) {
    if (Pred == ICmpInst::ICMP_UGT)
      return ConstantInt::getFalse(GetCompareTy(RHS));
    if (Pred == ICmpInst::ICMP_ULE)
      return ConstantInt::getTrue(GetCompareTy(RHS));
  }

  if (MaxRecurse && LBO && RBO && LBO->getOpcode() == RBO->getOpcode() &&
      LBO->getOperand(1) == RBO->getOperand(1)) {
    switch (LBO->getOpcode()) {
    default:
      break;
    case Instruction::UDiv:
    case Instruction::LShr:
      if (ICmpInst::isSigned(Pred) || !Q.IIQ.isExact(LBO) ||
          !Q.IIQ.isExact(RBO))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse - 1))
          return V;
      break;
    case Instruction::SDiv:
      if (!ICmpInst::isEquality(Pred) || !Q.IIQ.isExact(LBO) ||
          !Q.IIQ.isExact(RBO))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse - 1))
        return V;
      break;
    case Instruction::AShr:
      if (!Q.IIQ.isExact(LBO) || !Q.IIQ.isExact(RBO))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse - 1))
        return V;
      break;
    case Instruction::Shl: {
      bool NUW = Q.IIQ.hasNoUnsignedWrap(LBO) && Q.IIQ.hasNoUnsignedWrap(RBO);
      bool NSW = Q.IIQ.hasNoSignedWrap(LBO) && Q.IIQ.hasNoSignedWrap(RBO);
      if (!NUW && !NSW)
        break;
      if (!NSW && ICmpInst::isSigned(Pred))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), Q, MaxRecurse - 1))
        return V;
      break;
    }
    }
  }
  return nullptr;
}

/// Simplify integer comparisons where at least one operand of the compare
/// matches an integer min/max idiom.
static Value *simplifyICmpWithMinMax(CmpInst::Predicate Pred, Value *LHS,
                                     Value *RHS, const SimplifyQuery &Q,
                                     unsigned MaxRecurse) {
  Type *ITy = GetCompareTy(LHS); // The return type.
  Value *A, *B;
  CmpInst::Predicate P = CmpInst::BAD_ICMP_PREDICATE;
  CmpInst::Predicate EqP; // Chosen so that "A == max/min(A,B)" iff "A EqP B".

  // Signed variants on "max(a,b)>=a -> true".
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS)
      std::swap(A, B);       // smax(A, B) pred A.
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_SMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS)
      std::swap(A, B);       // A pred smax(A, B).
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS)
      std::swap(A, B);       // smin(A, B) pred A.
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS)
      std::swap(A, B);       // A pred smin(A, B).
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_SLE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, Q, MaxRecurse - 1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_SGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, Q, MaxRecurse - 1))
          return V;
      break;
    }
    case CmpInst::ICMP_SGE:
      // Always true.
      return getTrue(ITy);
    case CmpInst::ICMP_SLT:
      // Always false.
      return getFalse(ITy);
    }
  }

  // Unsigned variants on "max(a,b)>=a -> true".
  P = CmpInst::BAD_ICMP_PREDICATE;
  if (match(LHS, m_UMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS)
      std::swap(A, B);       // umax(A, B) pred A.
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_UMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS)
      std::swap(A, B);       // A pred umax(A, B).
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS)
      std::swap(A, B);       // umin(A, B) pred A.
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS)
      std::swap(A, B);       // A pred umin(A, B).
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_ULE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, Q, MaxRecurse - 1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_UGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, Q, MaxRecurse - 1))
          return V;
      break;
    }
    case CmpInst::ICMP_UGE:
      return getTrue(ITy);
    case CmpInst::ICMP_ULT:
      return getFalse(ITy);
    }
  }

  // Comparing 1 each of min/max with a common operand?
  // Canonicalize min operand to RHS.
  if (match(LHS, m_UMin(m_Value(), m_Value())) ||
      match(LHS, m_SMin(m_Value(), m_Value()))) {
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  Value *C, *D;
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) &&
      match(RHS, m_SMin(m_Value(C), m_Value(D))) &&
      (A == C || A == D || B == C || B == D)) {
    // smax(A, B) >=s smin(A, D) --> true
    if (Pred == CmpInst::ICMP_SGE)
      return getTrue(ITy);
    // smax(A, B) <s smin(A, D) --> false
    if (Pred == CmpInst::ICMP_SLT)
      return getFalse(ITy);
  } else if (match(LHS, m_UMax(m_Value(A), m_Value(B))) &&
             match(RHS, m_UMin(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // umax(A, B) >=u umin(A, D) --> true
    if (Pred == CmpInst::ICMP_UGE)
      return getTrue(ITy);
    // umax(A, B) <u umin(A, D) --> false
    if (Pred == CmpInst::ICMP_ULT)
      return getFalse(ITy);
  }

  return nullptr;
}

static Value *simplifyICmpWithDominatingAssume(CmpInst::Predicate Predicate,
                                               Value *LHS, Value *RHS,
                                               const SimplifyQuery &Q) {
  // Gracefully handle instructions that have not been inserted yet.
  if (!Q.AC || !Q.CxtI || !Q.CxtI->getParent())
    return nullptr;

  for (Value *AssumeBaseOp : {LHS, RHS}) {
    for (auto &AssumeVH : Q.AC->assumptionsFor(AssumeBaseOp)) {
      if (!AssumeVH)
        continue;

      CallInst *Assume = cast<CallInst>(AssumeVH);
      if (Optional<bool> Imp =
              isImpliedCondition(Assume->getArgOperand(0), Predicate, LHS, RHS,
                                 Q.DL))
        if (isValidAssumeForContext(Assume, Q.CxtI, Q.DT))
          return ConstantInt::get(GetCompareTy(LHS), *Imp);
    }
  }

  return nullptr;
}

/// Given operands for an ICmpInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               const SimplifyQuery &Q, unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isIntPredicate(Pred) && "Not an integer compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, Q.DL, Q.TLI);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(!isa<UndefValue>(LHS) && "Unexpected icmp undef,%X");

  Type *ITy = GetCompareTy(LHS); // The return type.

  // icmp poison, X -> poison
  if (isa<PoisonValue>(RHS))
    return PoisonValue::get(ITy);

  // For EQ and NE, we can always pick a value for the undef to make the
  // predicate pass or fail, so we can return undef.
  // Matches behavior in llvm::ConstantFoldCompareInstruction.
  if (Q.isUndefValue(RHS) && ICmpInst::isEquality(Pred))
    return UndefValue::get(ITy);

  // icmp X, X -> true/false
  // icmp X, undef -> true/false because undef could be X.
  if (LHS == RHS || Q.isUndefValue(RHS))
    return ConstantInt::get(ITy, CmpInst::isTrueWhenEqual(Pred));

  if (Value *V = simplifyICmpOfBools(Pred, LHS, RHS, Q))
    return V;

  // TODO: Sink/common this with other potentially expensive calls that use
  //       ValueTracking? See comment below for isKnownNonEqual().
  if (Value *V = simplifyICmpWithZero(Pred, LHS, RHS, Q))
    return V;

  if (Value *V = simplifyICmpWithConstant(Pred, LHS, RHS, Q.IIQ))
    return V;

  // If both operands have range metadata, use the metadata
  // to simplify the comparison.
  if (isa<Instruction>(RHS) && isa<Instruction>(LHS)) {
    auto RHS_Instr = cast<Instruction>(RHS);
    auto LHS_Instr = cast<Instruction>(LHS);

    if (Q.IIQ.getMetadata(RHS_Instr, LLVMContext::MD_range) &&
        Q.IIQ.getMetadata(LHS_Instr, LLVMContext::MD_range)) {
      auto RHS_CR = getConstantRangeFromMetadata(
          *RHS_Instr->getMetadata(LLVMContext::MD_range));
      auto LHS_CR = getConstantRangeFromMetadata(
          *LHS_Instr->getMetadata(LLVMContext::MD_range));

      if (LHS_CR.icmp(Pred, RHS_CR))
        return ConstantInt::getTrue(RHS->getContext());

      if (LHS_CR.icmp(CmpInst::getInversePredicate(Pred), RHS_CR))
        return ConstantInt::getFalse(RHS->getContext());
    }
  }

  // Compare of cast, for example (zext X) != 0 -> X != 0
  if (isa<CastInst>(LHS) && (isa<Constant>(RHS) || isa<CastInst>(RHS))) {
    Instruction *LI = cast<CastInst>(LHS);
    Value *SrcOp = LI->getOperand(0);
    Type *SrcTy = SrcOp->getType();
    Type *DstTy = LI->getType();

    // Turn icmp (ptrtoint x), (ptrtoint/constant) into a compare of the input
    // if the integer type is the same size as the pointer type.
    if (MaxRecurse && isa<PtrToIntInst>(LI) &&
        Q.DL.getTypeSizeInBits(SrcTy) == DstTy->getPrimitiveSizeInBits()) {
      if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
        // Transfer the cast to the constant.
        if (Value *V = SimplifyICmpInst(Pred, SrcOp,
                                        ConstantExpr::getIntToPtr(RHSC, SrcTy),
                                        Q, MaxRecurse-1))
          return V;
      } else if (PtrToIntInst *RI = dyn_cast<PtrToIntInst>(RHS)) {
        if (RI->getOperand(0)->getType() == SrcTy)
          // Compare without the cast.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          Q, MaxRecurse-1))
            return V;
      }
    }

    if (isa<ZExtInst>(LHS)) {
      // Turn icmp (zext X), (zext Y) into a compare of X and Y if they have the
      // same type.
      if (ZExtInst *RI = dyn_cast<ZExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that signed predicates become unsigned.
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                          SrcOp, RI->getOperand(0), Q,
                                          MaxRecurse-1))
            return V;
      }
      // Fold (zext X) ule (sext X), (zext X) sge (sext X) to true.
      else if (SExtInst *RI = dyn_cast<SExtInst>(RHS)) {
        if (SrcOp == RI->getOperand(0)) {
          if (Pred == ICmpInst::ICMP_ULE || Pred == ICmpInst::ICMP_SGE)
            return ConstantInt::getTrue(ITy);
          if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SLT)
            return ConstantInt::getFalse(ITy);
        }
      }
      // Turn icmp (zext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::ZExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two zero-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                        SrcOp, Trunc, Q, MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are zero while RHS has a non-zero bit
        // there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default: llvm_unreachable("Unknown ICmp predicate!");
          // LHS <u RHS.
          case ICmpInst::ICMP_EQ:
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            return ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_NE:
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            return ConstantInt::getTrue(CI->getContext());

          // LHS is non-negative.  If RHS is negative then LHS >s LHS.  If RHS
          // is non-negative then LHS <s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());
          }
        }
      }
    }

    if (isa<SExtInst>(LHS)) {
      // Turn icmp (sext X), (sext Y) into a compare of X and Y if they have the
      // same type.
      if (SExtInst *RI = dyn_cast<SExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that the predicate does not change.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          Q, MaxRecurse-1))
            return V;
      }
      // Fold (sext X) uge (zext X), (sext X) sle (zext X) to true.
      else if (ZExtInst *RI = dyn_cast<ZExtInst>(RHS)) {
        if (SrcOp == RI->getOperand(0)) {
          if (Pred == ICmpInst::ICMP_UGE || Pred == ICmpInst::ICMP_SLE)
            return ConstantInt::getTrue(ITy);
          if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_SGT)
            return ConstantInt::getFalse(ITy);
        }
      }
      // Turn icmp (sext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::SExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two sign-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, Trunc, Q, MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are all equal, while RHS has varying
        // bits there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default: llvm_unreachable("Unknown ICmp predicate!");
          case ICmpInst::ICMP_EQ:
            return ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_NE:
            return ConstantInt::getTrue(CI->getContext());

          // If RHS is non-negative then LHS <s RHS.  If RHS is negative then
          // LHS >s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());

          // If LHS is non-negative then LHS <u RHS.  If LHS is negative then
          // LHS >u RHS.
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            // Comparison is true iff the LHS <s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SLT, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              Q, MaxRecurse-1))
                return V;
            break;
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            // Comparison is true iff the LHS >=s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SGE, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              Q, MaxRecurse-1))
                return V;
            break;
          }
        }
      }
    }
  }

  // icmp eq|ne X, Y -> false|true if X != Y
  // This is potentially expensive, and we have already computedKnownBits for
  // compares with 0 above here, so only try this for a non-zero compare.
  if (ICmpInst::isEquality(Pred) && !match(RHS, m_Zero()) &&
      isKnownNonEqual(LHS, RHS, Q.DL, Q.AC, Q.CxtI, Q.DT, Q.IIQ.UseInstrInfo)) {
    return Pred == ICmpInst::ICMP_NE ? getTrue(ITy) : getFalse(ITy);
  }

  if (Value *V = simplifyICmpWithBinOp(Pred, LHS, RHS, Q, MaxRecurse))
    return V;

  if (Value *V = simplifyICmpWithMinMax(Pred, LHS, RHS, Q, MaxRecurse))
    return V;

  if (Value *V = simplifyICmpWithDominatingAssume(Pred, LHS, RHS, Q))
    return V;

  // Simplify comparisons of related pointers using a powerful, recursive
  // GEP-walk when we have target data available..
  if (LHS->getType()->isPointerTy())
    if (auto *C = computePointerICmp(Pred, LHS, RHS, Q))
      return C;
  if (auto *CLHS = dyn_cast<PtrToIntOperator>(LHS))
    if (auto *CRHS = dyn_cast<PtrToIntOperator>(RHS))
      if (Q.DL.getTypeSizeInBits(CLHS->getPointerOperandType()) ==
              Q.DL.getTypeSizeInBits(CLHS->getType()) &&
          Q.DL.getTypeSizeInBits(CRHS->getPointerOperandType()) ==
              Q.DL.getTypeSizeInBits(CRHS->getType()))
        if (auto *C = computePointerICmp(Pred, CLHS->getPointerOperand(),
                                         CRHS->getPointerOperand(), Q))
          return C;

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const SimplifyQuery &Q) {
  return ::SimplifyICmpInst(Predicate, LHS, RHS, Q, RecursionLimit);
}

/// Given operands for an FCmpInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               FastMathFlags FMF, const SimplifyQuery &Q,
                               unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isFPPredicate(Pred) && "Not an FP compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, Q.DL, Q.TLI);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  // Fold trivial predicates.
  Type *RetTy = GetCompareTy(LHS);
  if (Pred == FCmpInst::FCMP_FALSE)
    return getFalse(RetTy);
  if (Pred == FCmpInst::FCMP_TRUE)
    return getTrue(RetTy);

  // Fold (un)ordered comparison if we can determine there are no NaNs.
  if (Pred == FCmpInst::FCMP_UNO || Pred == FCmpInst::FCMP_ORD)
    if (FMF.noNaNs() ||
        (isKnownNeverNaN(LHS, Q.TLI) && isKnownNeverNaN(RHS, Q.TLI)))
      return ConstantInt::get(RetTy, Pred == FCmpInst::FCMP_ORD);

  // NaN is unordered; NaN is not ordered.
  assert((FCmpInst::isOrdered(Pred) || FCmpInst::isUnordered(Pred)) &&
         "Comparison must be either ordered or unordered");
  if (match(RHS, m_NaN()))
    return ConstantInt::get(RetTy, CmpInst::isUnordered(Pred));

  // fcmp pred x, poison and  fcmp pred poison, x
  // fold to poison
  if (isa<PoisonValue>(LHS) || isa<PoisonValue>(RHS))
    return PoisonValue::get(RetTy);

  // fcmp pred x, undef  and  fcmp pred undef, x
  // fold to true if unordered, false if ordered
  if (Q.isUndefValue(LHS) || Q.isUndefValue(RHS)) {
    // Choosing NaN for the undef will always make unordered comparison succeed
    // and ordered comparison fail.
    return ConstantInt::get(RetTy, CmpInst::isUnordered(Pred));
  }

  // fcmp x,x -> true/false.  Not all compares are foldable.
  if (LHS == RHS) {
    if (CmpInst::isTrueWhenEqual(Pred))
      return getTrue(RetTy);
    if (CmpInst::isFalseWhenEqual(Pred))
      return getFalse(RetTy);
  }

  // Handle fcmp with constant RHS.
  // TODO: Use match with a specific FP value, so these work with vectors with
  // undef lanes.
  const APFloat *C;
  if (match(RHS, m_APFloat(C))) {
    // Check whether the constant is an infinity.
    if (C->isInfinity()) {
      if (C->isNegative()) {
        switch (Pred) {
        case FCmpInst::FCMP_OLT:
          // No value is ordered and less than negative infinity.
          return getFalse(RetTy);
        case FCmpInst::FCMP_UGE:
          // All values are unordered with or at least negative infinity.
          return getTrue(RetTy);
        default:
          break;
        }
      } else {
        switch (Pred) {
        case FCmpInst::FCMP_OGT:
          // No value is ordered and greater than infinity.
          return getFalse(RetTy);
        case FCmpInst::FCMP_ULE:
          // All values are unordered with and at most infinity.
          return getTrue(RetTy);
        default:
          break;
        }
      }

      // LHS == Inf
      if (Pred == FCmpInst::FCMP_OEQ && isKnownNeverInfinity(LHS, Q.TLI))
        return getFalse(RetTy);
      // LHS != Inf
      if (Pred == FCmpInst::FCMP_UNE && isKnownNeverInfinity(LHS, Q.TLI))
        return getTrue(RetTy);
      // LHS == Inf || LHS == NaN
      if (Pred == FCmpInst::FCMP_UEQ && isKnownNeverInfinity(LHS, Q.TLI) &&
          isKnownNeverNaN(LHS, Q.TLI))
        return getFalse(RetTy);
      // LHS != Inf && LHS != NaN
      if (Pred == FCmpInst::FCMP_ONE && isKnownNeverInfinity(LHS, Q.TLI) &&
          isKnownNeverNaN(LHS, Q.TLI))
        return getTrue(RetTy);
    }
    if (C->isNegative() && !C->isNegZero()) {
      assert(!C->isNaN() && "Unexpected NaN constant!");
      // TODO: We can catch more cases by using a range check rather than
      //       relying on CannotBeOrderedLessThanZero.
      switch (Pred) {
      case FCmpInst::FCMP_UGE:
      case FCmpInst::FCMP_UGT:
      case FCmpInst::FCMP_UNE:
        // (X >= 0) implies (X > C) when (C < 0)
        if (CannotBeOrderedLessThanZero(LHS, Q.TLI))
          return getTrue(RetTy);
        break;
      case FCmpInst::FCMP_OEQ:
      case FCmpInst::FCMP_OLE:
      case FCmpInst::FCMP_OLT:
        // (X >= 0) implies !(X < C) when (C < 0)
        if (CannotBeOrderedLessThanZero(LHS, Q.TLI))
          return getFalse(RetTy);
        break;
      default:
        break;
      }
    }

    // Check comparison of [minnum/maxnum with constant] with other constant.
    const APFloat *C2;
    if ((match(LHS, m_Intrinsic<Intrinsic::minnum>(m_Value(), m_APFloat(C2))) &&
         *C2 < *C) ||
        (match(LHS, m_Intrinsic<Intrinsic::maxnum>(m_Value(), m_APFloat(C2))) &&
         *C2 > *C)) {
      bool IsMaxNum =
          cast<IntrinsicInst>(LHS)->getIntrinsicID() == Intrinsic::maxnum;
      // The ordered relationship and minnum/maxnum guarantee that we do not
      // have NaN constants, so ordered/unordered preds are handled the same.
      switch (Pred) {
      case FCmpInst::FCMP_OEQ: case FCmpInst::FCMP_UEQ:
        // minnum(X, LesserC)  == C --> false
        // maxnum(X, GreaterC) == C --> false
        return getFalse(RetTy);
      case FCmpInst::FCMP_ONE: case FCmpInst::FCMP_UNE:
        // minnum(X, LesserC)  != C --> true
        // maxnum(X, GreaterC) != C --> true
        return getTrue(RetTy);
      case FCmpInst::FCMP_OGE: case FCmpInst::FCMP_UGE:
      case FCmpInst::FCMP_OGT: case FCmpInst::FCMP_UGT:
        // minnum(X, LesserC)  >= C --> false
        // minnum(X, LesserC)  >  C --> false
        // maxnum(X, GreaterC) >= C --> true
        // maxnum(X, GreaterC) >  C --> true
        return ConstantInt::get(RetTy, IsMaxNum);
      case FCmpInst::FCMP_OLE: case FCmpInst::FCMP_ULE:
      case FCmpInst::FCMP_OLT: case FCmpInst::FCMP_ULT:
        // minnum(X, LesserC)  <= C --> true
        // minnum(X, LesserC)  <  C --> true
        // maxnum(X, GreaterC) <= C --> false
        // maxnum(X, GreaterC) <  C --> false
        return ConstantInt::get(RetTy, !IsMaxNum);
      default:
        // TRUE/FALSE/ORD/UNO should be handled before this.
        llvm_unreachable("Unexpected fcmp predicate");
      }
    }
  }

  if (match(RHS, m_AnyZeroFP())) {
    switch (Pred) {
    case FCmpInst::FCMP_OGE:
    case FCmpInst::FCMP_ULT:
      // Positive or zero X >= 0.0 --> true
      // Positive or zero X <  0.0 --> false
      if ((FMF.noNaNs() || isKnownNeverNaN(LHS, Q.TLI)) &&
          CannotBeOrderedLessThanZero(LHS, Q.TLI))
        return Pred == FCmpInst::FCMP_OGE ? getTrue(RetTy) : getFalse(RetTy);
      break;
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_OLT:
      // Positive or zero or nan X >= 0.0 --> true
      // Positive or zero or nan X <  0.0 --> false
      if (CannotBeOrderedLessThanZero(LHS, Q.TLI))
        return Pred == FCmpInst::FCMP_UGE ? getTrue(RetTy) : getFalse(RetTy);
      break;
    default:
      break;
    }
  }

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, Q, MaxRecurse))
      return V;

  return nullptr;
}

Value *llvm::SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              FastMathFlags FMF, const SimplifyQuery &Q) {
  return ::SimplifyFCmpInst(Predicate, LHS, RHS, FMF, Q, RecursionLimit);
}

static Value *simplifyWithOpReplaced(Value *V, Value *Op, Value *RepOp,
                                     const SimplifyQuery &Q,
                                     bool AllowRefinement,
                                     unsigned MaxRecurse) {
  assert(!Op->getType()->isVectorTy() && "This is not safe for vectors");

  // Trivial replacement.
  if (V == Op)
    return RepOp;

  // We cannot replace a constant, and shouldn't even try.
  if (isa<Constant>(Op))
    return nullptr;

  auto *I = dyn_cast<Instruction>(V);
  if (!I || !is_contained(I->operands(), Op))
    return nullptr;

  // Replace Op with RepOp in instruction operands.
  SmallVector<Value *, 8> NewOps(I->getNumOperands());
  transform(I->operands(), NewOps.begin(),
            [&](Value *V) { return V == Op ? RepOp : V; });

  if (!AllowRefinement) {
    // General InstSimplify functions may refine the result, e.g. by returning
    // a constant for a potentially poison value. To avoid this, implement only
    // a few non-refining but profitable transforms here.

    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      unsigned Opcode = BO->getOpcode();
      // id op x -> x, x op id -> x
      if (NewOps[0] == ConstantExpr::getBinOpIdentity(Opcode, I->getType()))
        return NewOps[1];
      if (NewOps[1] == ConstantExpr::getBinOpIdentity(Opcode, I->getType(),
                                                      /* RHS */ true))
        return NewOps[0];

      // x & x -> x, x | x -> x
      if ((Opcode == Instruction::And || Opcode == Instruction::Or) &&
          NewOps[0] == NewOps[1])
        return NewOps[0];
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      // getelementptr x, 0 -> x
      if (NewOps.size() == 2 && match(NewOps[1], m_Zero()) &&
          !GEP->isInBounds())
        return NewOps[0];
    }
  } else if (MaxRecurse) {
    // The simplification queries below may return the original value. Consider:
    //   %div = udiv i32 %arg, %arg2
    //   %mul = mul nsw i32 %div, %arg2
    //   %cmp = icmp eq i32 %mul, %arg
    //   %sel = select i1 %cmp, i32 %div, i32 undef
    // Replacing %arg by %mul, %div becomes "udiv i32 %mul, %arg2", which
    // simplifies back to %arg. This can only happen because %mul does not
    // dominate %div. To ensure a consistent return value contract, we make sure
    // that this case returns nullptr as well.
    auto PreventSelfSimplify = [V](Value *Simplified) {
      return Simplified != V ? Simplified : nullptr;
    };

    if (auto *B = dyn_cast<BinaryOperator>(I))
      return PreventSelfSimplify(SimplifyBinOp(B->getOpcode(), NewOps[0],
                                               NewOps[1], Q, MaxRecurse - 1));

    if (CmpInst *C = dyn_cast<CmpInst>(I))
      return PreventSelfSimplify(SimplifyCmpInst(C->getPredicate(), NewOps[0],
                                                 NewOps[1], Q, MaxRecurse - 1));

    if (auto *GEP = dyn_cast<GetElementPtrInst>(I))
      return PreventSelfSimplify(SimplifyGEPInst(
          GEP->getSourceElementType(), NewOps[0], makeArrayRef(NewOps).slice(1),
          GEP->isInBounds(), Q, MaxRecurse - 1));

    if (isa<SelectInst>(I))
      return PreventSelfSimplify(
          SimplifySelectInst(NewOps[0], NewOps[1], NewOps[2], Q,
                             MaxRecurse - 1));
    // TODO: We could hand off more cases to instsimplify here.
  }

  // If all operands are constant after substituting Op for RepOp then we can
  // constant fold the instruction.
  SmallVector<Constant *, 8> ConstOps;
  for (Value *NewOp : NewOps) {
    if (Constant *ConstOp = dyn_cast<Constant>(NewOp))
      ConstOps.push_back(ConstOp);
    else
      return nullptr;
  }

  // Consider:
  //   %cmp = icmp eq i32 %x, 2147483647
  //   %add = add nsw i32 %x, 1
  //   %sel = select i1 %cmp, i32 -2147483648, i32 %add
  //
  // We can't replace %sel with %add unless we strip away the flags (which
  // will be done in InstCombine).
  // TODO: This may be unsound, because it only catches some forms of
  // refinement.
  if (!AllowRefinement && canCreatePoison(cast<Operator>(I)))
    return nullptr;

  if (CmpInst *C = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(C->getPredicate(), ConstOps[0],
                                           ConstOps[1], Q.DL, Q.TLI);

  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    if (!LI->isVolatile())
      return ConstantFoldLoadFromConstPtr(ConstOps[0], LI->getType(), Q.DL);

  return ConstantFoldInstOperands(I, ConstOps, Q.DL, Q.TLI);
}

Value *llvm::simplifyWithOpReplaced(Value *V, Value *Op, Value *RepOp,
                                    const SimplifyQuery &Q,
                                    bool AllowRefinement) {
  return ::simplifyWithOpReplaced(V, Op, RepOp, Q, AllowRefinement,
                                  RecursionLimit);
}

/// Try to simplify a select instruction when its condition operand is an
/// integer comparison where one operand of the compare is a constant.
static Value *simplifySelectBitTest(Value *TrueVal, Value *FalseVal, Value *X,
                                    const APInt *Y, bool TrueWhenUnset) {
  const APInt *C;

  // (X & Y) == 0 ? X & ~Y : X  --> X
  // (X & Y) != 0 ? X & ~Y : X  --> X & ~Y
  if (FalseVal == X && match(TrueVal, m_And(m_Specific(X), m_APInt(C))) &&
      *Y == ~*C)
    return TrueWhenUnset ? FalseVal : TrueVal;

  // (X & Y) == 0 ? X : X & ~Y  --> X & ~Y
  // (X & Y) != 0 ? X : X & ~Y  --> X
  if (TrueVal == X && match(FalseVal, m_And(m_Specific(X), m_APInt(C))) &&
      *Y == ~*C)
    return TrueWhenUnset ? FalseVal : TrueVal;

  if (Y->isPowerOf2()) {
    // (X & Y) == 0 ? X | Y : X  --> X | Y
    // (X & Y) != 0 ? X | Y : X  --> X
    if (FalseVal == X && match(TrueVal, m_Or(m_Specific(X), m_APInt(C))) &&
        *Y == *C)
      return TrueWhenUnset ? TrueVal : FalseVal;

    // (X & Y) == 0 ? X : X | Y  --> X
    // (X & Y) != 0 ? X : X | Y  --> X | Y
    if (TrueVal == X && match(FalseVal, m_Or(m_Specific(X), m_APInt(C))) &&
        *Y == *C)
      return TrueWhenUnset ? TrueVal : FalseVal;
  }

  return nullptr;
}

/// An alternative way to test if a bit is set or not uses sgt/slt instead of
/// eq/ne.
static Value *simplifySelectWithFakeICmpEq(Value *CmpLHS, Value *CmpRHS,
                                           ICmpInst::Predicate Pred,
                                           Value *TrueVal, Value *FalseVal) {
  Value *X;
  APInt Mask;
  if (!decomposeBitTestICmp(CmpLHS, CmpRHS, Pred, X, Mask))
    return nullptr;

  return simplifySelectBitTest(TrueVal, FalseVal, X, &Mask,
                               Pred == ICmpInst::ICMP_EQ);
}

/// Try to simplify a select instruction when its condition operand is an
/// integer comparison.
static Value *simplifySelectWithICmpCond(Value *CondVal, Value *TrueVal,
                                         Value *FalseVal, const SimplifyQuery &Q,
                                         unsigned MaxRecurse) {
  ICmpInst::Predicate Pred;
  Value *CmpLHS, *CmpRHS;
  if (!match(CondVal, m_ICmp(Pred, m_Value(CmpLHS), m_Value(CmpRHS))))
    return nullptr;

  // Canonicalize ne to eq predicate.
  if (Pred == ICmpInst::ICMP_NE) {
    Pred = ICmpInst::ICMP_EQ;
    std::swap(TrueVal, FalseVal);
  }

  // Check for integer min/max with a limit constant:
  // X > MIN_INT ? X : MIN_INT --> X
  // X < MAX_INT ? X : MAX_INT --> X
  if (TrueVal->getType()->isIntOrIntVectorTy()) {
    Value *X, *Y;
    SelectPatternFlavor SPF =
        matchDecomposedSelectPattern(cast<ICmpInst>(CondVal), TrueVal, FalseVal,
                                     X, Y).Flavor;
    if (SelectPatternResult::isMinOrMax(SPF) && Pred == getMinMaxPred(SPF)) {
      APInt LimitC = getMinMaxLimit(getInverseMinMaxFlavor(SPF),
                                    X->getType()->getScalarSizeInBits());
      if (match(Y, m_SpecificInt(LimitC)))
        return X;
    }
  }

  if (Pred == ICmpInst::ICMP_EQ && match(CmpRHS, m_Zero())) {
    Value *X;
    const APInt *Y;
    if (match(CmpLHS, m_And(m_Value(X), m_APInt(Y))))
      if (Value *V = simplifySelectBitTest(TrueVal, FalseVal, X, Y,
                                           /*TrueWhenUnset=*/true))
        return V;

    // Test for a bogus zero-shift-guard-op around funnel-shift or rotate.
    Value *ShAmt;
    auto isFsh = m_CombineOr(m_FShl(m_Value(X), m_Value(), m_Value(ShAmt)),
                             m_FShr(m_Value(), m_Value(X), m_Value(ShAmt)));
    // (ShAmt == 0) ? fshl(X, *, ShAmt) : X --> X
    // (ShAmt == 0) ? fshr(*, X, ShAmt) : X --> X
    if (match(TrueVal, isFsh) && FalseVal == X && CmpLHS == ShAmt)
      return X;

    // Test for a zero-shift-guard-op around rotates. These are used to
    // avoid UB from oversized shifts in raw IR rotate patterns, but the
    // intrinsics do not have that problem.
    // We do not allow this transform for the general funnel shift case because
    // that would not preserve the poison safety of the original code.
    auto isRotate =
        m_CombineOr(m_FShl(m_Value(X), m_Deferred(X), m_Value(ShAmt)),
                    m_FShr(m_Value(X), m_Deferred(X), m_Value(ShAmt)));
    // (ShAmt == 0) ? X : fshl(X, X, ShAmt) --> fshl(X, X, ShAmt)
    // (ShAmt == 0) ? X : fshr(X, X, ShAmt) --> fshr(X, X, ShAmt)
    if (match(FalseVal, isRotate) && TrueVal == X && CmpLHS == ShAmt &&
        Pred == ICmpInst::ICMP_EQ)
      return FalseVal;

    // X == 0 ? abs(X) : -abs(X) --> -abs(X)
    // X == 0 ? -abs(X) : abs(X) --> abs(X)
    if (match(TrueVal, m_Intrinsic<Intrinsic::abs>(m_Specific(CmpLHS))) &&
        match(FalseVal, m_Neg(m_Intrinsic<Intrinsic::abs>(m_Specific(CmpLHS)))))
      return FalseVal;
    if (match(TrueVal,
              m_Neg(m_Intrinsic<Intrinsic::abs>(m_Specific(CmpLHS)))) &&
        match(FalseVal, m_Intrinsic<Intrinsic::abs>(m_Specific(CmpLHS))))
      return FalseVal;
  }

  // Check for other compares that behave like bit test.
  if (Value *V = simplifySelectWithFakeICmpEq(CmpLHS, CmpRHS, Pred,
                                              TrueVal, FalseVal))
    return V;

  // If we have a scalar equality comparison, then we know the value in one of
  // the arms of the select. See if substituting this value into the arm and
  // simplifying the result yields the same value as the other arm.
  // Note that the equivalence/replacement opportunity does not hold for vectors
  // because each element of a vector select is chosen independently.
  if (Pred == ICmpInst::ICMP_EQ && !CondVal->getType()->isVectorTy()) {
    if (simplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, Q,
                               /* AllowRefinement */ false, MaxRecurse) ==
            TrueVal ||
        simplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, Q,
                               /* AllowRefinement */ false, MaxRecurse) ==
            TrueVal)
      return FalseVal;
    if (simplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, Q,
                               /* AllowRefinement */ true, MaxRecurse) ==
            FalseVal ||
        simplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, Q,
                               /* AllowRefinement */ true, MaxRecurse) ==
            FalseVal)
      return FalseVal;
  }

  return nullptr;
}

/// Try to simplify a select instruction when its condition operand is a
/// floating-point comparison.
static Value *simplifySelectWithFCmp(Value *Cond, Value *T, Value *F,
                                     const SimplifyQuery &Q) {
  FCmpInst::Predicate Pred;
  if (!match(Cond, m_FCmp(Pred, m_Specific(T), m_Specific(F))) &&
      !match(Cond, m_FCmp(Pred, m_Specific(F), m_Specific(T))))
    return nullptr;

  // This transform is safe if we do not have (do not care about) -0.0 or if
  // at least one operand is known to not be -0.0. Otherwise, the select can
  // change the sign of a zero operand.
  bool HasNoSignedZeros = Q.CxtI && isa<FPMathOperator>(Q.CxtI) &&
                          Q.CxtI->hasNoSignedZeros();
  const APFloat *C;
  if (HasNoSignedZeros || (match(T, m_APFloat(C)) && C->isNonZero()) ||
                          (match(F, m_APFloat(C)) && C->isNonZero())) {
    // (T == F) ? T : F --> F
    // (F == T) ? T : F --> F
    if (Pred == FCmpInst::FCMP_OEQ)
      return F;

    // (T != F) ? T : F --> T
    // (F != T) ? T : F --> T
    if (Pred == FCmpInst::FCMP_UNE)
      return T;
  }

  return nullptr;
}

/// Given operands for a SelectInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                                 const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (auto *CondC = dyn_cast<Constant>(Cond)) {
    if (auto *TrueC = dyn_cast<Constant>(TrueVal))
      if (auto *FalseC = dyn_cast<Constant>(FalseVal))
        return ConstantFoldSelectInstruction(CondC, TrueC, FalseC);

    // select poison, X, Y -> poison
    if (isa<PoisonValue>(CondC))
      return PoisonValue::get(TrueVal->getType());

    // select undef, X, Y -> X or Y
    if (Q.isUndefValue(CondC))
      return isa<Constant>(FalseVal) ? FalseVal : TrueVal;

    // select true,  X, Y --> X
    // select false, X, Y --> Y
    // For vectors, allow undef/poison elements in the condition to match the
    // defined elements, so we can eliminate the select.
    if (match(CondC, m_One()))
      return TrueVal;
    if (match(CondC, m_Zero()))
      return FalseVal;
  }

  assert(Cond->getType()->isIntOrIntVectorTy(1) &&
         "Select must have bool or bool vector condition");
  assert(TrueVal->getType() == FalseVal->getType() &&
         "Select must have same types for true/false ops");

  if (Cond->getType() == TrueVal->getType()) {
    // select i1 Cond, i1 true, i1 false --> i1 Cond
    if (match(TrueVal, m_One()) && match(FalseVal, m_ZeroInt()))
      return Cond;

    // (X || Y) && (X || !Y) --> X (commuted 8 ways)
    Value *X, *Y;
    if (match(FalseVal, m_ZeroInt())) {
      if (match(Cond, m_c_LogicalOr(m_Value(X), m_Not(m_Value(Y)))) &&
          match(TrueVal, m_c_LogicalOr(m_Specific(X), m_Specific(Y))))
        return X;
      if (match(TrueVal, m_c_LogicalOr(m_Value(X), m_Not(m_Value(Y)))) &&
          match(Cond, m_c_LogicalOr(m_Specific(X), m_Specific(Y))))
        return X;
    }
  }

  // select ?, X, X -> X
  if (TrueVal == FalseVal)
    return TrueVal;

  // If the true or false value is poison, we can fold to the other value.
  // If the true or false value is undef, we can fold to the other value as
  // long as the other value isn't poison.
  // select ?, poison, X -> X
  // select ?, undef,  X -> X
  if (isa<PoisonValue>(TrueVal) ||
      (Q.isUndefValue(TrueVal) &&
       isGuaranteedNotToBePoison(FalseVal, Q.AC, Q.CxtI, Q.DT)))
    return FalseVal;
  // select ?, X, poison -> X
  // select ?, X, undef  -> X
  if (isa<PoisonValue>(FalseVal) ||
      (Q.isUndefValue(FalseVal) &&
       isGuaranteedNotToBePoison(TrueVal, Q.AC, Q.CxtI, Q.DT)))
    return TrueVal;

  // Deal with partial undef vector constants: select ?, VecC, VecC' --> VecC''
  Constant *TrueC, *FalseC;
  if (isa<FixedVectorType>(TrueVal->getType()) &&
      match(TrueVal, m_Constant(TrueC)) &&
      match(FalseVal, m_Constant(FalseC))) {
    unsigned NumElts =
        cast<FixedVectorType>(TrueC->getType())->getNumElements();
    SmallVector<Constant *, 16> NewC;
    for (unsigned i = 0; i != NumElts; ++i) {
      // Bail out on incomplete vector constants.
      Constant *TEltC = TrueC->getAggregateElement(i);
      Constant *FEltC = FalseC->getAggregateElement(i);
      if (!TEltC || !FEltC)
        break;

      // If the elements match (undef or not), that value is the result. If only
      // one element is undef, choose the defined element as the safe result.
      if (TEltC == FEltC)
        NewC.push_back(TEltC);
      else if (isa<PoisonValue>(TEltC) ||
               (Q.isUndefValue(TEltC) && isGuaranteedNotToBePoison(FEltC)))
        NewC.push_back(FEltC);
      else if (isa<PoisonValue>(FEltC) ||
               (Q.isUndefValue(FEltC) && isGuaranteedNotToBePoison(TEltC)))
        NewC.push_back(TEltC);
      else
        break;
    }
    if (NewC.size() == NumElts)
      return ConstantVector::get(NewC);
  }

  if (Value *V =
          simplifySelectWithICmpCond(Cond, TrueVal, FalseVal, Q, MaxRecurse))
    return V;

  if (Value *V = simplifySelectWithFCmp(Cond, TrueVal, FalseVal, Q))
    return V;

  if (Value *V = foldSelectWithBinaryOp(Cond, TrueVal, FalseVal))
    return V;

  Optional<bool> Imp = isImpliedByDomCondition(Cond, Q.CxtI, Q.DL);
  if (Imp)
    return *Imp ? TrueVal : FalseVal;

  return nullptr;
}

Value *llvm::SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                                const SimplifyQuery &Q) {
  return ::SimplifySelectInst(Cond, TrueVal, FalseVal, Q, RecursionLimit);
}

/// Given operands for an GetElementPtrInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyGEPInst(Type *SrcTy, Value *Ptr,
                              ArrayRef<Value *> Indices, bool InBounds,
                              const SimplifyQuery &Q, unsigned) {
  // The type of the GEP pointer operand.
  unsigned AS =
      cast<PointerType>(Ptr->getType()->getScalarType())->getAddressSpace();

  // getelementptr P -> P.
  if (Indices.empty())
    return Ptr;

  // Compute the (pointer) type returned by the GEP instruction.
  Type *LastType = GetElementPtrInst::getIndexedType(SrcTy, Indices);
  Type *GEPTy = PointerType::get(LastType, AS);
  if (VectorType *VT = dyn_cast<VectorType>(Ptr->getType()))
    GEPTy = VectorType::get(GEPTy, VT->getElementCount());
  else {
    for (Value *Op : Indices) {
      // If one of the operands is a vector, the result type is a vector of
      // pointers. All vector operands must have the same number of elements.
      if (VectorType *VT = dyn_cast<VectorType>(Op->getType())) {
        GEPTy = VectorType::get(GEPTy, VT->getElementCount());
        break;
      }
    }
  }

  // For opaque pointers an all-zero GEP is a no-op. For typed pointers,
  // it may be equivalent to a bitcast.
  if (Ptr->getType()->isOpaquePointerTy() &&
      all_of(Indices, [](const auto *V) { return match(V, m_Zero()); }))
    return Ptr;

  // getelementptr poison, idx -> poison
  // getelementptr baseptr, poison -> poison
  if (isa<PoisonValue>(Ptr) ||
      any_of(Indices, [](const auto *V) { return isa<PoisonValue>(V); }))
    return PoisonValue::get(GEPTy);

  if (Q.isUndefValue(Ptr))
    // If inbounds, we can choose an out-of-bounds pointer as a base pointer.
    return InBounds ? PoisonValue::get(GEPTy) : UndefValue::get(GEPTy);

  bool IsScalableVec =
      isa<ScalableVectorType>(SrcTy) || any_of(Indices, [](const Value *V) {
        return isa<ScalableVectorType>(V->getType());
      });

  if (Indices.size() == 1) {
    // getelementptr P, 0 -> P.
    if (match(Indices[0], m_Zero()) && Ptr->getType() == GEPTy)
      return Ptr;

    Type *Ty = SrcTy;
    if (!IsScalableVec && Ty->isSized()) {
      Value *P;
      uint64_t C;
      uint64_t TyAllocSize = Q.DL.getTypeAllocSize(Ty);
      // getelementptr P, N -> P if P points to a type of zero size.
      if (TyAllocSize == 0 && Ptr->getType() == GEPTy)
        return Ptr;

      // The following transforms are only safe if the ptrtoint cast
      // doesn't truncate the pointers.
      if (Indices[0]->getType()->getScalarSizeInBits() ==
          Q.DL.getPointerSizeInBits(AS)) {
        auto CanSimplify = [GEPTy, &P, Ptr]() -> bool {
          return P->getType() == GEPTy &&
                 getUnderlyingObject(P) == getUnderlyingObject(Ptr);
        };
        // getelementptr V, (sub P, V) -> P if P points to a type of size 1.
        if (TyAllocSize == 1 &&
            match(Indices[0],
                  m_Sub(m_PtrToInt(m_Value(P)), m_PtrToInt(m_Specific(Ptr)))) &&
            CanSimplify())
          return P;

        // getelementptr V, (ashr (sub P, V), C) -> P if P points to a type of
        // size 1 << C.
        if (match(Indices[0], m_AShr(m_Sub(m_PtrToInt(m_Value(P)),
                                           m_PtrToInt(m_Specific(Ptr))),
                                     m_ConstantInt(C))) &&
            TyAllocSize == 1ULL << C && CanSimplify())
          return P;

        // getelementptr V, (sdiv (sub P, V), C) -> P if P points to a type of
        // size C.
        if (match(Indices[0], m_SDiv(m_Sub(m_PtrToInt(m_Value(P)),
                                           m_PtrToInt(m_Specific(Ptr))),
                                     m_SpecificInt(TyAllocSize))) &&
            CanSimplify())
          return P;
      }
    }
  }

  if (!IsScalableVec && Q.DL.getTypeAllocSize(LastType) == 1 &&
      all_of(Indices.drop_back(1),
             [](Value *Idx) { return match(Idx, m_Zero()); })) {
    unsigned IdxWidth =
        Q.DL.getIndexSizeInBits(Ptr->getType()->getPointerAddressSpace());
    if (Q.DL.getTypeSizeInBits(Indices.back()->getType()) == IdxWidth) {
      APInt BasePtrOffset(IdxWidth, 0);
      Value *StrippedBasePtr =
          Ptr->stripAndAccumulateInBoundsConstantOffsets(Q.DL, BasePtrOffset);

      // Avoid creating inttoptr of zero here: While LLVMs treatment of
      // inttoptr is generally conservative, this particular case is folded to
      // a null pointer, which will have incorrect provenance.

      // gep (gep V, C), (sub 0, V) -> C
      if (match(Indices.back(),
                m_Sub(m_Zero(), m_PtrToInt(m_Specific(StrippedBasePtr)))) &&
          !BasePtrOffset.isZero()) {
        auto *CI = ConstantInt::get(GEPTy->getContext(), BasePtrOffset);
        return ConstantExpr::getIntToPtr(CI, GEPTy);
      }
      // gep (gep V, C), (xor V, -1) -> C-1
      if (match(Indices.back(),
                m_Xor(m_PtrToInt(m_Specific(StrippedBasePtr)), m_AllOnes())) &&
          !BasePtrOffset.isOne()) {
        auto *CI = ConstantInt::get(GEPTy->getContext(), BasePtrOffset - 1);
        return ConstantExpr::getIntToPtr(CI, GEPTy);
      }
    }
  }

  // Check to see if this is constant foldable.
  if (!isa<Constant>(Ptr) ||
      !all_of(Indices, [](Value *V) { return isa<Constant>(V); }))
    return nullptr;

  auto *CE = ConstantExpr::getGetElementPtr(SrcTy, cast<Constant>(Ptr), Indices,
                                            InBounds);
  return ConstantFoldConstant(CE, Q.DL);
}

Value *llvm::SimplifyGEPInst(Type *SrcTy, Value *Ptr, ArrayRef<Value *> Indices,
                             bool InBounds, const SimplifyQuery &Q) {
  return ::SimplifyGEPInst(SrcTy, Ptr, Indices, InBounds, Q, RecursionLimit);
}

/// Given operands for an InsertValueInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                      ArrayRef<unsigned> Idxs, const SimplifyQuery &Q,
                                      unsigned) {
  if (Constant *CAgg = dyn_cast<Constant>(Agg))
    if (Constant *CVal = dyn_cast<Constant>(Val))
      return ConstantFoldInsertValueInstruction(CAgg, CVal, Idxs);

  // insertvalue x, undef, n -> x
  if (Q.isUndefValue(Val))
    return Agg;

  // insertvalue x, (extractvalue y, n), n
  if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(Val))
    if (EV->getAggregateOperand()->getType() == Agg->getType() &&
        EV->getIndices() == Idxs) {
      // insertvalue undef, (extractvalue y, n), n -> y
      if (Q.isUndefValue(Agg))
        return EV->getAggregateOperand();

      // insertvalue y, (extractvalue y, n), n -> y
      if (Agg == EV->getAggregateOperand())
        return Agg;
    }

  return nullptr;
}

Value *llvm::SimplifyInsertValueInst(Value *Agg, Value *Val,
                                     ArrayRef<unsigned> Idxs,
                                     const SimplifyQuery &Q) {
  return ::SimplifyInsertValueInst(Agg, Val, Idxs, Q, RecursionLimit);
}

Value *llvm::SimplifyInsertElementInst(Value *Vec, Value *Val, Value *Idx,
                                       const SimplifyQuery &Q) {
  // Try to constant fold.
  auto *VecC = dyn_cast<Constant>(Vec);
  auto *ValC = dyn_cast<Constant>(Val);
  auto *IdxC = dyn_cast<Constant>(Idx);
  if (VecC && ValC && IdxC)
    return ConstantExpr::getInsertElement(VecC, ValC, IdxC);

  // For fixed-length vector, fold into poison if index is out of bounds.
  if (auto *CI = dyn_cast<ConstantInt>(Idx)) {
    if (isa<FixedVectorType>(Vec->getType()) &&
        CI->uge(cast<FixedVectorType>(Vec->getType())->getNumElements()))
      return PoisonValue::get(Vec->getType());
  }

  // If index is undef, it might be out of bounds (see above case)
  if (Q.isUndefValue(Idx))
    return PoisonValue::get(Vec->getType());

  // If the scalar is poison, or it is undef and there is no risk of
  // propagating poison from the vector value, simplify to the vector value.
  if (isa<PoisonValue>(Val) ||
      (Q.isUndefValue(Val) && isGuaranteedNotToBePoison(Vec)))
    return Vec;

  // If we are extracting a value from a vector, then inserting it into the same
  // place, that's the input vector:
  // insertelt Vec, (extractelt Vec, Idx), Idx --> Vec
  if (match(Val, m_ExtractElt(m_Specific(Vec), m_Specific(Idx))))
    return Vec;

  return nullptr;
}

/// Given operands for an ExtractValueInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                       const SimplifyQuery &, unsigned) {
  if (auto *CAgg = dyn_cast<Constant>(Agg))
    return ConstantFoldExtractValueInstruction(CAgg, Idxs);

  // extractvalue x, (insertvalue y, elt, n), n -> elt
  unsigned NumIdxs = Idxs.size();
  for (auto *IVI = dyn_cast<InsertValueInst>(Agg); IVI != nullptr;
       IVI = dyn_cast<InsertValueInst>(IVI->getAggregateOperand())) {
    ArrayRef<unsigned> InsertValueIdxs = IVI->getIndices();
    unsigned NumInsertValueIdxs = InsertValueIdxs.size();
    unsigned NumCommonIdxs = std::min(NumInsertValueIdxs, NumIdxs);
    if (InsertValueIdxs.slice(0, NumCommonIdxs) ==
        Idxs.slice(0, NumCommonIdxs)) {
      if (NumIdxs == NumInsertValueIdxs)
        return IVI->getInsertedValueOperand();
      break;
    }
  }

  return nullptr;
}

Value *llvm::SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                      const SimplifyQuery &Q) {
  return ::SimplifyExtractValueInst(Agg, Idxs, Q, RecursionLimit);
}

/// Given operands for an ExtractElementInst, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyExtractElementInst(Value *Vec, Value *Idx,
                                         const SimplifyQuery &Q, unsigned) {
  auto *VecVTy = cast<VectorType>(Vec->getType());
  if (auto *CVec = dyn_cast<Constant>(Vec)) {
    if (auto *CIdx = dyn_cast<Constant>(Idx))
      return ConstantExpr::getExtractElement(CVec, CIdx);

    if (Q.isUndefValue(Vec))
      return UndefValue::get(VecVTy->getElementType());
  }

  // An undef extract index can be arbitrarily chosen to be an out-of-range
  // index value, which would result in the instruction being poison.
  if (Q.isUndefValue(Idx))
    return PoisonValue::get(VecVTy->getElementType());

  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (auto *IdxC = dyn_cast<ConstantInt>(Idx)) {
    // For fixed-length vector, fold into undef if index is out of bounds.
    unsigned MinNumElts = VecVTy->getElementCount().getKnownMinValue();
    if (isa<FixedVectorType>(VecVTy) && IdxC->getValue().uge(MinNumElts))
      return PoisonValue::get(VecVTy->getElementType());
    // Handle case where an element is extracted from a splat.
    if (IdxC->getValue().ult(MinNumElts))
      if (auto *Splat = getSplatValue(Vec))
        return Splat;
    if (Value *Elt = findScalarElement(Vec, IdxC->getZExtValue()))
      return Elt;
  } else {
    // The index is not relevant if our vector is a splat.
    if (Value *Splat = getSplatValue(Vec))
      return Splat;
  }
  return nullptr;
}

Value *llvm::SimplifyExtractElementInst(Value *Vec, Value *Idx,
                                        const SimplifyQuery &Q) {
  return ::SimplifyExtractElementInst(Vec, Idx, Q, RecursionLimit);
}

/// See if we can fold the given phi. If not, returns null.
static Value *SimplifyPHINode(PHINode *PN, ArrayRef<Value *> IncomingValues,
                              const SimplifyQuery &Q) {
  // WARNING: no matter how worthwhile it may seem, we can not perform PHI CSE
  //          here, because the PHI we may succeed simplifying to was not
  //          def-reachable from the original PHI!

  // If all of the PHI's incoming values are the same then replace the PHI node
  // with the common value.
  Value *CommonValue = nullptr;
  bool HasUndefInput = false;
  for (Value *Incoming : IncomingValues) {
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PN) continue;
    if (Q.isUndefValue(Incoming)) {
      // Remember that we saw an undef value, but otherwise ignore them.
      HasUndefInput = true;
      continue;
    }
    if (CommonValue && Incoming != CommonValue)
      return nullptr;  // Not the same, bail out.
    CommonValue = Incoming;
  }

  // If CommonValue is null then all of the incoming values were either undef or
  // equal to the phi node itself.
  if (!CommonValue)
    return UndefValue::get(PN->getType());

  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot return X as the result of the PHI node unless it
  // dominates the PHI block.
  if (HasUndefInput)
    return valueDominatesPHI(CommonValue, PN, Q.DT) ? CommonValue : nullptr;

  return CommonValue;
}

static Value *SimplifyCastInst(unsigned CastOpc, Value *Op,
                               Type *Ty, const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (auto *C = dyn_cast<Constant>(Op))
    return ConstantFoldCastOperand(CastOpc, C, Ty, Q.DL);

  if (auto *CI = dyn_cast<CastInst>(Op)) {
    auto *Src = CI->getOperand(0);
    Type *SrcTy = Src->getType();
    Type *MidTy = CI->getType();
    Type *DstTy = Ty;
    if (Src->getType() == Ty) {
      auto FirstOp = static_cast<Instruction::CastOps>(CI->getOpcode());
      auto SecondOp = static_cast<Instruction::CastOps>(CastOpc);
      Type *SrcIntPtrTy =
          SrcTy->isPtrOrPtrVectorTy() ? Q.DL.getIntPtrType(SrcTy) : nullptr;
      Type *MidIntPtrTy =
          MidTy->isPtrOrPtrVectorTy() ? Q.DL.getIntPtrType(MidTy) : nullptr;
      Type *DstIntPtrTy =
          DstTy->isPtrOrPtrVectorTy() ? Q.DL.getIntPtrType(DstTy) : nullptr;
      if (CastInst::isEliminableCastPair(FirstOp, SecondOp, SrcTy, MidTy, DstTy,
                                         SrcIntPtrTy, MidIntPtrTy,
                                         DstIntPtrTy) == Instruction::BitCast)
        return Src;
    }
  }

  // bitcast x -> x
  if (CastOpc == Instruction::BitCast)
    if (Op->getType() == Ty)
      return Op;

  return nullptr;
}

Value *llvm::SimplifyCastInst(unsigned CastOpc, Value *Op, Type *Ty,
                              const SimplifyQuery &Q) {
  return ::SimplifyCastInst(CastOpc, Op, Ty, Q, RecursionLimit);
}

/// For the given destination element of a shuffle, peek through shuffles to
/// match a root vector source operand that contains that element in the same
/// vector lane (ie, the same mask index), so we can eliminate the shuffle(s).
static Value *foldIdentityShuffles(int DestElt, Value *Op0, Value *Op1,
                                   int MaskVal, Value *RootVec,
                                   unsigned MaxRecurse) {
  if (!MaxRecurse--)
    return nullptr;

  // Bail out if any mask value is undefined. That kind of shuffle may be
  // simplified further based on demanded bits or other folds.
  if (MaskVal == -1)
    return nullptr;

  // The mask value chooses which source operand we need to look at next.
  int InVecNumElts = cast<FixedVectorType>(Op0->getType())->getNumElements();
  int RootElt = MaskVal;
  Value *SourceOp = Op0;
  if (MaskVal >= InVecNumElts) {
    RootElt = MaskVal - InVecNumElts;
    SourceOp = Op1;
  }

  // If the source operand is a shuffle itself, look through it to find the
  // matching root vector.
  if (auto *SourceShuf = dyn_cast<ShuffleVectorInst>(SourceOp)) {
    return foldIdentityShuffles(
        DestElt, SourceShuf->getOperand(0), SourceShuf->getOperand(1),
        SourceShuf->getMaskValue(RootElt), RootVec, MaxRecurse);
  }

  // TODO: Look through bitcasts? What if the bitcast changes the vector element
  // size?

  // The source operand is not a shuffle. Initialize the root vector value for
  // this shuffle if that has not been done yet.
  if (!RootVec)
    RootVec = SourceOp;

  // Give up as soon as a source operand does not match the existing root value.
  if (RootVec != SourceOp)
    return nullptr;

  // The element must be coming from the same lane in the source vector
  // (although it may have crossed lanes in intermediate shuffles).
  if (RootElt != DestElt)
    return nullptr;

  return RootVec;
}

static Value *SimplifyShuffleVectorInst(Value *Op0, Value *Op1,
                                        ArrayRef<int> Mask, Type *RetTy,
                                        const SimplifyQuery &Q,
                                        unsigned MaxRecurse) {
  if (all_of(Mask, [](int Elem) { return Elem == UndefMaskElem; }))
    return UndefValue::get(RetTy);

  auto *InVecTy = cast<VectorType>(Op0->getType());
  unsigned MaskNumElts = Mask.size();
  ElementCount InVecEltCount = InVecTy->getElementCount();

  bool Scalable = InVecEltCount.isScalable();

  SmallVector<int, 32> Indices;
  Indices.assign(Mask.begin(), Mask.end());

  // Canonicalization: If mask does not select elements from an input vector,
  // replace that input vector with poison.
  if (!Scalable) {
    bool MaskSelects0 = false, MaskSelects1 = false;
    unsigned InVecNumElts = InVecEltCount.getKnownMinValue();
    for (unsigned i = 0; i != MaskNumElts; ++i) {
      if (Indices[i] == -1)
        continue;
      if ((unsigned)Indices[i] < InVecNumElts)
        MaskSelects0 = true;
      else
        MaskSelects1 = true;
    }
    if (!MaskSelects0)
      Op0 = PoisonValue::get(InVecTy);
    if (!MaskSelects1)
      Op1 = PoisonValue::get(InVecTy);
  }

  auto *Op0Const = dyn_cast<Constant>(Op0);
  auto *Op1Const = dyn_cast<Constant>(Op1);

  // If all operands are constant, constant fold the shuffle. This
  // transformation depends on the value of the mask which is not known at
  // compile time for scalable vectors
  if (Op0Const && Op1Const)
    return ConstantExpr::getShuffleVector(Op0Const, Op1Const, Mask);

  // Canonicalization: if only one input vector is constant, it shall be the
  // second one. This transformation depends on the value of the mask which
  // is not known at compile time for scalable vectors
  if (!Scalable && Op0Const && !Op1Const) {
    std::swap(Op0, Op1);
    ShuffleVectorInst::commuteShuffleMask(Indices,
                                          InVecEltCount.getKnownMinValue());
  }

  // A splat of an inserted scalar constant becomes a vector constant:
  // shuf (inselt ?, C, IndexC), undef, <IndexC, IndexC...> --> <C, C...>
  // NOTE: We may have commuted above, so analyze the updated Indices, not the
  //       original mask constant.
  // NOTE: This transformation depends on the value of the mask which is not
  // known at compile time for scalable vectors
  Constant *C;
  ConstantInt *IndexC;
  if (!Scalable && match(Op0, m_InsertElt(m_Value(), m_Constant(C),
                                          m_ConstantInt(IndexC)))) {
    // Match a splat shuffle mask of the insert index allowing undef elements.
    int InsertIndex = IndexC->getZExtValue();
    if (all_of(Indices, [InsertIndex](int MaskElt) {
          return MaskElt == InsertIndex || MaskElt == -1;
        })) {
      assert(isa<UndefValue>(Op1) && "Expected undef operand 1 for splat");

      // Shuffle mask undefs become undefined constant result elements.
      SmallVector<Constant *, 16> VecC(MaskNumElts, C);
      for (unsigned i = 0; i != MaskNumElts; ++i)
        if (Indices[i] == -1)
          VecC[i] = UndefValue::get(C->getType());
      return ConstantVector::get(VecC);
    }
  }

  // A shuffle of a splat is always the splat itself. Legal if the shuffle's
  // value type is same as the input vectors' type.
  if (auto *OpShuf = dyn_cast<ShuffleVectorInst>(Op0))
    if (Q.isUndefValue(Op1) && RetTy == InVecTy &&
        is_splat(OpShuf->getShuffleMask()))
      return Op0;

  // All remaining transformation depend on the value of the mask, which is
  // not known at compile time for scalable vectors.
  if (Scalable)
    return nullptr;

  // Don't fold a shuffle with undef mask elements. This may get folded in a
  // better way using demanded bits or other analysis.
  // TODO: Should we allow this?
  if (is_contained(Indices, -1))
    return nullptr;

  // Check if every element of this shuffle can be mapped back to the
  // corresponding element of a single root vector. If so, we don't need this
  // shuffle. This handles simple identity shuffles as well as chains of
  // shuffles that may widen/narrow and/or move elements across lanes and back.
  Value *RootVec = nullptr;
  for (unsigned i = 0; i != MaskNumElts; ++i) {
    // Note that recursion is limited for each vector element, so if any element
    // exceeds the limit, this will fail to simplify.
    RootVec =
        foldIdentityShuffles(i, Op0, Op1, Indices[i], RootVec, MaxRecurse);

    // We can't replace a widening/narrowing shuffle with one of its operands.
    if (!RootVec || RootVec->getType() != RetTy)
      return nullptr;
  }
  return RootVec;
}

/// Given operands for a ShuffleVectorInst, fold the result or return null.
Value *llvm::SimplifyShuffleVectorInst(Value *Op0, Value *Op1,
                                       ArrayRef<int> Mask, Type *RetTy,
                                       const SimplifyQuery &Q) {
  return ::SimplifyShuffleVectorInst(Op0, Op1, Mask, RetTy, Q, RecursionLimit);
}

static Constant *foldConstant(Instruction::UnaryOps Opcode,
                              Value *&Op, const SimplifyQuery &Q) {
  if (auto *C = dyn_cast<Constant>(Op))
    return ConstantFoldUnaryOpOperand(Opcode, C, Q.DL);
  return nullptr;
}

/// Given the operand for an FNeg, see if we can fold the result.  If not, this
/// returns null.
static Value *simplifyFNegInst(Value *Op, FastMathFlags FMF,
                               const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (Constant *C = foldConstant(Instruction::FNeg, Op, Q))
    return C;

  Value *X;
  // fneg (fneg X) ==> X
  if (match(Op, m_FNeg(m_Value(X))))
    return X;

  return nullptr;
}

Value *llvm::SimplifyFNegInst(Value *Op, FastMathFlags FMF,
                              const SimplifyQuery &Q) {
  return ::simplifyFNegInst(Op, FMF, Q, RecursionLimit);
}

static Constant *propagateNaN(Constant *In) {
  // If the input is a vector with undef elements, just return a default NaN.
  if (!In->isNaN())
    return ConstantFP::getNaN(In->getType());

  // Propagate the existing NaN constant when possible.
  // TODO: Should we quiet a signaling NaN?
  return In;
}

/// Perform folds that are common to any floating-point operation. This implies
/// transforms based on poison/undef/NaN because the operation itself makes no
/// difference to the result.
static Constant *simplifyFPOp(ArrayRef<Value *> Ops, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  // Poison is independent of anything else. It always propagates from an
  // operand to a math result.
  if (any_of(Ops, [](Value *V) { return match(V, m_Poison()); }))
    return PoisonValue::get(Ops[0]->getType());

  for (Value *V : Ops) {
    bool IsNan = match(V, m_NaN());
    bool IsInf = match(V, m_Inf());
    bool IsUndef = Q.isUndefValue(V);

    // If this operation has 'nnan' or 'ninf' and at least 1 disallowed operand
    // (an undef operand can be chosen to be Nan/Inf), then the result of
    // this operation is poison.
    if (FMF.noNaNs() && (IsNan || IsUndef))
      return PoisonValue::get(V->getType());
    if (FMF.noInfs() && (IsInf || IsUndef))
      return PoisonValue::get(V->getType());

    if (isDefaultFPEnvironment(ExBehavior, Rounding)) {
      if (IsUndef || IsNan)
        return propagateNaN(cast<Constant>(V));
    } else if (ExBehavior != fp::ebStrict) {
      if (IsNan)
        return propagateNaN(cast<Constant>(V));
    }
  }
  return nullptr;
}

// TODO: Move this out to a header file:
static inline bool canIgnoreSNaN(fp::ExceptionBehavior EB, FastMathFlags FMF) {
  return (EB == fp::ebIgnore || FMF.noNaNs());
}

/// Given operands for an FAdd, see if we can fold the result.  If not, this
/// returns null.
static Value *
SimplifyFAddInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                 const SimplifyQuery &Q, unsigned MaxRecurse,
                 fp::ExceptionBehavior ExBehavior = fp::ebIgnore,
                 RoundingMode Rounding = RoundingMode::NearestTiesToEven) {
  if (isDefaultFPEnvironment(ExBehavior, Rounding))
    if (Constant *C = foldOrCommuteConstant(Instruction::FAdd, Op0, Op1, Q))
      return C;

  if (Constant *C = simplifyFPOp({Op0, Op1}, FMF, Q, ExBehavior, Rounding))
    return C;

  // fadd X, -0 ==> X
  // With strict/constrained FP, we have these possible edge cases that do
  // not simplify to Op0:
  // fadd SNaN, -0.0 --> QNaN
  // fadd +0.0, -0.0 --> -0.0 (but only with round toward negative)
  if (canIgnoreSNaN(ExBehavior, FMF) &&
      (!canRoundingModeBe(Rounding, RoundingMode::TowardNegative) ||
       FMF.noSignedZeros()))
    if (match(Op1, m_NegZeroFP()))
      return Op0;

  // fadd X, 0 ==> X, when we know X is not -0
  if (canIgnoreSNaN(ExBehavior, FMF))
    if (match(Op1, m_PosZeroFP()) &&
        (FMF.noSignedZeros() || CannotBeNegativeZero(Op0, Q.TLI)))
      return Op0;

  if (!isDefaultFPEnvironment(ExBehavior, Rounding))
    return nullptr;

  // With nnan: -X + X --> 0.0 (and commuted variant)
  // We don't have to explicitly exclude infinities (ninf): INF + -INF == NaN.
  // Negative zeros are allowed because we always end up with positive zero:
  // X = -0.0: (-0.0 - (-0.0)) + (-0.0) == ( 0.0) + (-0.0) == 0.0
  // X = -0.0: ( 0.0 - (-0.0)) + (-0.0) == ( 0.0) + (-0.0) == 0.0
  // X =  0.0: (-0.0 - ( 0.0)) + ( 0.0) == (-0.0) + ( 0.0) == 0.0
  // X =  0.0: ( 0.0 - ( 0.0)) + ( 0.0) == ( 0.0) + ( 0.0) == 0.0
  if (FMF.noNaNs()) {
    if (match(Op0, m_FSub(m_AnyZeroFP(), m_Specific(Op1))) ||
        match(Op1, m_FSub(m_AnyZeroFP(), m_Specific(Op0))))
      return ConstantFP::getNullValue(Op0->getType());

    if (match(Op0, m_FNeg(m_Specific(Op1))) ||
        match(Op1, m_FNeg(m_Specific(Op0))))
      return ConstantFP::getNullValue(Op0->getType());
  }

  // (X - Y) + Y --> X
  // Y + (X - Y) --> X
  Value *X;
  if (FMF.noSignedZeros() && FMF.allowReassoc() &&
      (match(Op0, m_FSub(m_Value(X), m_Specific(Op1))) ||
       match(Op1, m_FSub(m_Value(X), m_Specific(Op0)))))
    return X;

  return nullptr;
}

/// Given operands for an FSub, see if we can fold the result.  If not, this
/// returns null.
static Value *
SimplifyFSubInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                 const SimplifyQuery &Q, unsigned MaxRecurse,
                 fp::ExceptionBehavior ExBehavior = fp::ebIgnore,
                 RoundingMode Rounding = RoundingMode::NearestTiesToEven) {
  if (isDefaultFPEnvironment(ExBehavior, Rounding))
    if (Constant *C = foldOrCommuteConstant(Instruction::FSub, Op0, Op1, Q))
      return C;

  if (Constant *C = simplifyFPOp({Op0, Op1}, FMF, Q, ExBehavior, Rounding))
    return C;

  if (!isDefaultFPEnvironment(ExBehavior, Rounding))
    return nullptr;

  // fsub X, +0 ==> X
  if (match(Op1, m_PosZeroFP()))
    return Op0;

  // fsub X, -0 ==> X, when we know X is not -0
  if (match(Op1, m_NegZeroFP()) &&
      (FMF.noSignedZeros() || CannotBeNegativeZero(Op0, Q.TLI)))
    return Op0;

  // fsub -0.0, (fsub -0.0, X) ==> X
  // fsub -0.0, (fneg X) ==> X
  Value *X;
  if (match(Op0, m_NegZeroFP()) &&
      match(Op1, m_FNeg(m_Value(X))))
    return X;

  // fsub 0.0, (fsub 0.0, X) ==> X if signed zeros are ignored.
  // fsub 0.0, (fneg X) ==> X if signed zeros are ignored.
  if (FMF.noSignedZeros() && match(Op0, m_AnyZeroFP()) &&
      (match(Op1, m_FSub(m_AnyZeroFP(), m_Value(X))) ||
       match(Op1, m_FNeg(m_Value(X)))))
    return X;

  // fsub nnan x, x ==> 0.0
  if (FMF.noNaNs() && Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // Y - (Y - X) --> X
  // (X + Y) - Y --> X
  if (FMF.noSignedZeros() && FMF.allowReassoc() &&
      (match(Op1, m_FSub(m_Specific(Op0), m_Value(X))) ||
       match(Op0, m_c_FAdd(m_Specific(Op1), m_Value(X)))))
    return X;

  return nullptr;
}

static Value *SimplifyFMAFMul(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q, unsigned MaxRecurse,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  if (Constant *C = simplifyFPOp({Op0, Op1}, FMF, Q, ExBehavior, Rounding))
    return C;

  if (!isDefaultFPEnvironment(ExBehavior, Rounding))
    return nullptr;

  // fmul X, 1.0 ==> X
  if (match(Op1, m_FPOne()))
    return Op0;

  // fmul 1.0, X ==> X
  if (match(Op0, m_FPOne()))
    return Op1;

  // fmul nnan nsz X, 0 ==> 0
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op1, m_AnyZeroFP()))
    return ConstantFP::getNullValue(Op0->getType());

  // fmul nnan nsz 0, X ==> 0
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op0, m_AnyZeroFP()))
    return ConstantFP::getNullValue(Op1->getType());

  // sqrt(X) * sqrt(X) --> X, if we can:
  // 1. Remove the intermediate rounding (reassociate).
  // 2. Ignore non-zero negative numbers because sqrt would produce NAN.
  // 3. Ignore -0.0 because sqrt(-0.0) == -0.0, but -0.0 * -0.0 == 0.0.
  Value *X;
  if (Op0 == Op1 && match(Op0, m_Intrinsic<Intrinsic::sqrt>(m_Value(X))) &&
      FMF.allowReassoc() && FMF.noNaNs() && FMF.noSignedZeros())
    return X;

  return nullptr;
}

/// Given the operands for an FMul, see if we can fold the result
static Value *
SimplifyFMulInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                 const SimplifyQuery &Q, unsigned MaxRecurse,
                 fp::ExceptionBehavior ExBehavior = fp::ebIgnore,
                 RoundingMode Rounding = RoundingMode::NearestTiesToEven) {
  if (isDefaultFPEnvironment(ExBehavior, Rounding))
    if (Constant *C = foldOrCommuteConstant(Instruction::FMul, Op0, Op1, Q))
      return C;

  // Now apply simplifications that do not require rounding.
  return SimplifyFMAFMul(Op0, Op1, FMF, Q, MaxRecurse, ExBehavior, Rounding);
}

Value *llvm::SimplifyFAddInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  return ::SimplifyFAddInst(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                            Rounding);
}

Value *llvm::SimplifyFSubInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  return ::SimplifyFSubInst(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                            Rounding);
}

Value *llvm::SimplifyFMulInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  return ::SimplifyFMulInst(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                            Rounding);
}

Value *llvm::SimplifyFMAFMul(Value *Op0, Value *Op1, FastMathFlags FMF,
                             const SimplifyQuery &Q,
                             fp::ExceptionBehavior ExBehavior,
                             RoundingMode Rounding) {
  return ::SimplifyFMAFMul(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                           Rounding);
}

static Value *
SimplifyFDivInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                 const SimplifyQuery &Q, unsigned,
                 fp::ExceptionBehavior ExBehavior = fp::ebIgnore,
                 RoundingMode Rounding = RoundingMode::NearestTiesToEven) {
  if (isDefaultFPEnvironment(ExBehavior, Rounding))
    if (Constant *C = foldOrCommuteConstant(Instruction::FDiv, Op0, Op1, Q))
      return C;

  if (Constant *C = simplifyFPOp({Op0, Op1}, FMF, Q, ExBehavior, Rounding))
    return C;

  if (!isDefaultFPEnvironment(ExBehavior, Rounding))
    return nullptr;

  // X / 1.0 -> X
  if (match(Op1, m_FPOne()))
    return Op0;

  // 0 / X -> 0
  // Requires that NaNs are off (X could be zero) and signed zeroes are
  // ignored (X could be positive or negative, so the output sign is unknown).
  if (FMF.noNaNs() && FMF.noSignedZeros() && match(Op0, m_AnyZeroFP()))
    return ConstantFP::getNullValue(Op0->getType());

  if (FMF.noNaNs()) {
    // X / X -> 1.0 is legal when NaNs are ignored.
    // We can ignore infinities because INF/INF is NaN.
    if (Op0 == Op1)
      return ConstantFP::get(Op0->getType(), 1.0);

    // (X * Y) / Y --> X if we can reassociate to the above form.
    Value *X;
    if (FMF.allowReassoc() && match(Op0, m_c_FMul(m_Value(X), m_Specific(Op1))))
      return X;

    // -X /  X -> -1.0 and
    //  X / -X -> -1.0 are legal when NaNs are ignored.
    // We can ignore signed zeros because +-0.0/+-0.0 is NaN and ignored.
    if (match(Op0, m_FNegNSZ(m_Specific(Op1))) ||
        match(Op1, m_FNegNSZ(m_Specific(Op0))))
      return ConstantFP::get(Op0->getType(), -1.0);
  }

  return nullptr;
}

Value *llvm::SimplifyFDivInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  return ::SimplifyFDivInst(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                            Rounding);
}

static Value *
SimplifyFRemInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                 const SimplifyQuery &Q, unsigned,
                 fp::ExceptionBehavior ExBehavior = fp::ebIgnore,
                 RoundingMode Rounding = RoundingMode::NearestTiesToEven) {
  if (isDefaultFPEnvironment(ExBehavior, Rounding))
    if (Constant *C = foldOrCommuteConstant(Instruction::FRem, Op0, Op1, Q))
      return C;

  if (Constant *C = simplifyFPOp({Op0, Op1}, FMF, Q, ExBehavior, Rounding))
    return C;

  if (!isDefaultFPEnvironment(ExBehavior, Rounding))
    return nullptr;

  // Unlike fdiv, the result of frem always matches the sign of the dividend.
  // The constant match may include undef elements in a vector, so return a full
  // zero constant as the result.
  if (FMF.noNaNs()) {
    // +0 % X -> 0
    if (match(Op0, m_PosZeroFP()))
      return ConstantFP::getNullValue(Op0->getType());
    // -0 % X -> -0
    if (match(Op0, m_NegZeroFP()))
      return ConstantFP::getNegativeZero(Op0->getType());
  }

  return nullptr;
}

Value *llvm::SimplifyFRemInst(Value *Op0, Value *Op1, FastMathFlags FMF,
                              const SimplifyQuery &Q,
                              fp::ExceptionBehavior ExBehavior,
                              RoundingMode Rounding) {
  return ::SimplifyFRemInst(Op0, Op1, FMF, Q, RecursionLimit, ExBehavior,
                            Rounding);
}

//=== Helper functions for higher up the class hierarchy.

/// Given the operand for a UnaryOperator, see if we can fold the result.
/// If not, this returns null.
static Value *simplifyUnOp(unsigned Opcode, Value *Op, const SimplifyQuery &Q,
                           unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::FNeg:
    return simplifyFNegInst(Op, FastMathFlags(), Q, MaxRecurse);
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

/// Given the operand for a UnaryOperator, see if we can fold the result.
/// If not, this returns null.
/// Try to use FastMathFlags when folding the result.
static Value *simplifyFPUnOp(unsigned Opcode, Value *Op,
                             const FastMathFlags &FMF,
                             const SimplifyQuery &Q, unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::FNeg:
    return simplifyFNegInst(Op, FMF, Q, MaxRecurse);
  default:
    return simplifyUnOp(Opcode, Op, Q, MaxRecurse);
  }
}

Value *llvm::SimplifyUnOp(unsigned Opcode, Value *Op, const SimplifyQuery &Q) {
  return ::simplifyUnOp(Opcode, Op, Q, RecursionLimit);
}

Value *llvm::SimplifyUnOp(unsigned Opcode, Value *Op, FastMathFlags FMF,
                          const SimplifyQuery &Q) {
  return ::simplifyFPUnOp(Opcode, Op, FMF, Q, RecursionLimit);
}

/// Given operands for a BinaryOperator, see if we can fold the result.
/// If not, this returns null.
static Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                            const SimplifyQuery &Q, unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::Add:
    return SimplifyAddInst(LHS, RHS, false, false, Q, MaxRecurse);
  case Instruction::Sub:
    return SimplifySubInst(LHS, RHS, false, false, Q, MaxRecurse);
  case Instruction::Mul:
    return SimplifyMulInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::SDiv:
    return SimplifySDivInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::UDiv:
    return SimplifyUDivInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::SRem:
    return SimplifySRemInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::URem:
    return SimplifyURemInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::Shl:
    return SimplifyShlInst(LHS, RHS, false, false, Q, MaxRecurse);
  case Instruction::LShr:
    return SimplifyLShrInst(LHS, RHS, false, Q, MaxRecurse);
  case Instruction::AShr:
    return SimplifyAShrInst(LHS, RHS, false, Q, MaxRecurse);
  case Instruction::And:
    return SimplifyAndInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::Or:
    return SimplifyOrInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::Xor:
    return SimplifyXorInst(LHS, RHS, Q, MaxRecurse);
  case Instruction::FAdd:
    return SimplifyFAddInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::FSub:
    return SimplifyFSubInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::FMul:
    return SimplifyFMulInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::FDiv:
    return SimplifyFDivInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  case Instruction::FRem:
    return SimplifyFRemInst(LHS, RHS, FastMathFlags(), Q, MaxRecurse);
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

/// Given operands for a BinaryOperator, see if we can fold the result.
/// If not, this returns null.
/// Try to use FastMathFlags when folding the result.
static Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                            const FastMathFlags &FMF, const SimplifyQuery &Q,
                            unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::FAdd:
    return SimplifyFAddInst(LHS, RHS, FMF, Q, MaxRecurse);
  case Instruction::FSub:
    return SimplifyFSubInst(LHS, RHS, FMF, Q, MaxRecurse);
  case Instruction::FMul:
    return SimplifyFMulInst(LHS, RHS, FMF, Q, MaxRecurse);
  case Instruction::FDiv:
    return SimplifyFDivInst(LHS, RHS, FMF, Q, MaxRecurse);
  default:
    return SimplifyBinOp(Opcode, LHS, RHS, Q, MaxRecurse);
  }
}

Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                           const SimplifyQuery &Q) {
  return ::SimplifyBinOp(Opcode, LHS, RHS, Q, RecursionLimit);
}

Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                           FastMathFlags FMF, const SimplifyQuery &Q) {
  return ::SimplifyBinOp(Opcode, LHS, RHS, FMF, Q, RecursionLimit);
}

/// Given operands for a CmpInst, see if we can fold the result.
static Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const SimplifyQuery &Q, unsigned MaxRecurse) {
  if (CmpInst::isIntPredicate((CmpInst::Predicate)Predicate))
    return SimplifyICmpInst(Predicate, LHS, RHS, Q, MaxRecurse);
  return SimplifyFCmpInst(Predicate, LHS, RHS, FastMathFlags(), Q, MaxRecurse);
}

Value *llvm::SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                             const SimplifyQuery &Q) {
  return ::SimplifyCmpInst(Predicate, LHS, RHS, Q, RecursionLimit);
}

static bool IsIdempotent(Intrinsic::ID ID) {
  switch (ID) {
  default: return false;

  // Unary idempotent: f(f(x)) = f(x)
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::canonicalize:
    return true;
  }
}

static Value *SimplifyRelativeLoad(Constant *Ptr, Constant *Offset,
                                   const DataLayout &DL) {
  GlobalValue *PtrSym;
  APInt PtrOffset;
  if (!IsConstantOffsetFromGlobal(Ptr, PtrSym, PtrOffset, DL))
    return nullptr;

  Type *Int8PtrTy = Type::getInt8PtrTy(Ptr->getContext());
  Type *Int32Ty = Type::getInt32Ty(Ptr->getContext());
  Type *Int32PtrTy = Int32Ty->getPointerTo();
  Type *Int64Ty = Type::getInt64Ty(Ptr->getContext());

  auto *OffsetConstInt = dyn_cast<ConstantInt>(Offset);
  if (!OffsetConstInt || OffsetConstInt->getType()->getBitWidth() > 64)
    return nullptr;

  uint64_t OffsetInt = OffsetConstInt->getSExtValue();
  if (OffsetInt % 4 != 0)
    return nullptr;

  Constant *C = ConstantExpr::getGetElementPtr(
      Int32Ty, ConstantExpr::getBitCast(Ptr, Int32PtrTy),
      ConstantInt::get(Int64Ty, OffsetInt / 4));
  Constant *Loaded = ConstantFoldLoadFromConstPtr(C, Int32Ty, DL);
  if (!Loaded)
    return nullptr;

  auto *LoadedCE = dyn_cast<ConstantExpr>(Loaded);
  if (!LoadedCE)
    return nullptr;

  if (LoadedCE->getOpcode() == Instruction::Trunc) {
    LoadedCE = dyn_cast<ConstantExpr>(LoadedCE->getOperand(0));
    if (!LoadedCE)
      return nullptr;
  }

  if (LoadedCE->getOpcode() != Instruction::Sub)
    return nullptr;

  auto *LoadedLHS = dyn_cast<ConstantExpr>(LoadedCE->getOperand(0));
  if (!LoadedLHS || LoadedLHS->getOpcode() != Instruction::PtrToInt)
    return nullptr;
  auto *LoadedLHSPtr = LoadedLHS->getOperand(0);

  Constant *LoadedRHS = LoadedCE->getOperand(1);
  GlobalValue *LoadedRHSSym;
  APInt LoadedRHSOffset;
  if (!IsConstantOffsetFromGlobal(LoadedRHS, LoadedRHSSym, LoadedRHSOffset,
                                  DL) ||
      PtrSym != LoadedRHSSym || PtrOffset != LoadedRHSOffset)
    return nullptr;

  return ConstantExpr::getBitCast(LoadedLHSPtr, Int8PtrTy);
}

static Value *simplifyUnaryIntrinsic(Function *F, Value *Op0,
                                     const SimplifyQuery &Q) {
  // Idempotent functions return the same result when called repeatedly.
  Intrinsic::ID IID = F->getIntrinsicID();
  if (IsIdempotent(IID))
    if (auto *II = dyn_cast<IntrinsicInst>(Op0))
      if (II->getIntrinsicID() == IID)
        return II;

  Value *X;
  switch (IID) {
  case Intrinsic::fabs:
    if (SignBitMustBeZero(Op0, Q.TLI)) return Op0;
    break;
  case Intrinsic::bswap:
    // bswap(bswap(x)) -> x
    if (match(Op0, m_BSwap(m_Value(X)))) return X;
    break;
  case Intrinsic::bitreverse:
    // bitreverse(bitreverse(x)) -> x
    if (match(Op0, m_BitReverse(m_Value(X)))) return X;
    break;
  case Intrinsic::ctpop: {
    // If everything but the lowest bit is zero, that bit is the pop-count. Ex:
    // ctpop(and X, 1) --> and X, 1
    unsigned BitWidth = Op0->getType()->getScalarSizeInBits();
    if (MaskedValueIsZero(Op0, APInt::getHighBitsSet(BitWidth, BitWidth - 1),
                          Q.DL, 0, Q.AC, Q.CxtI, Q.DT))
      return Op0;
    break;
  }
  case Intrinsic::exp:
    // exp(log(x)) -> x
    if (Q.CxtI->hasAllowReassoc() &&
        match(Op0, m_Intrinsic<Intrinsic::log>(m_Value(X)))) return X;
    break;
  case Intrinsic::exp2:
    // exp2(log2(x)) -> x
    if (Q.CxtI->hasAllowReassoc() &&
        match(Op0, m_Intrinsic<Intrinsic::log2>(m_Value(X)))) return X;
    break;
  case Intrinsic::log:
    // log(exp(x)) -> x
    if (Q.CxtI->hasAllowReassoc() &&
        match(Op0, m_Intrinsic<Intrinsic::exp>(m_Value(X)))) return X;
    break;
  case Intrinsic::log2:
    // log2(exp2(x)) -> x
    if (Q.CxtI->hasAllowReassoc() &&
        (match(Op0, m_Intrinsic<Intrinsic::exp2>(m_Value(X))) ||
         match(Op0, m_Intrinsic<Intrinsic::pow>(m_SpecificFP(2.0),
                                                m_Value(X))))) return X;
    break;
  case Intrinsic::log10:
    // log10(pow(10.0, x)) -> x
    if (Q.CxtI->hasAllowReassoc() &&
        match(Op0, m_Intrinsic<Intrinsic::pow>(m_SpecificFP(10.0),
                                               m_Value(X)))) return X;
    break;
  case Intrinsic::floor:
  case Intrinsic::trunc:
  case Intrinsic::ceil:
  case Intrinsic::round:
  case Intrinsic::roundeven:
  case Intrinsic::nearbyint:
  case Intrinsic::rint: {
    // floor (sitofp x) -> sitofp x
    // floor (uitofp x) -> uitofp x
    //
    // Converting from int always results in a finite integral number or
    // infinity. For either of those inputs, these rounding functions always
    // return the same value, so the rounding can be eliminated.
    if (match(Op0, m_SIToFP(m_Value())) || match(Op0, m_UIToFP(m_Value())))
      return Op0;
    break;
  }
  case Intrinsic::experimental_vector_reverse:
    // experimental.vector.reverse(experimental.vector.reverse(x)) -> x
    if (match(Op0,
              m_Intrinsic<Intrinsic::experimental_vector_reverse>(m_Value(X))))
      return X;
    // experimental.vector.reverse(splat(X)) -> splat(X)
    if (isSplatValue(Op0))
      return Op0;
    break;
  default:
    break;
  }

  return nullptr;
}

/// Given a min/max intrinsic, see if it can be removed based on having an
/// operand that is another min/max intrinsic with shared operand(s). The caller
/// is expected to swap the operand arguments to handle commutation.
static Value *foldMinMaxSharedOp(Intrinsic::ID IID, Value *Op0, Value *Op1) {
  Value *X, *Y;
  if (!match(Op0, m_MaxOrMin(m_Value(X), m_Value(Y))))
    return nullptr;

  auto *MM0 = dyn_cast<IntrinsicInst>(Op0);
  if (!MM0)
    return nullptr;
  Intrinsic::ID IID0 = MM0->getIntrinsicID();

  if (Op1 == X || Op1 == Y ||
      match(Op1, m_c_MaxOrMin(m_Specific(X), m_Specific(Y)))) {
    // max (max X, Y), X --> max X, Y
    if (IID0 == IID)
      return MM0;
    // max (min X, Y), X --> X
    if (IID0 == getInverseMinMaxIntrinsic(IID))
      return Op1;
  }
  return nullptr;
}

static Value *simplifyBinaryIntrinsic(Function *F, Value *Op0, Value *Op1,
                                      const SimplifyQuery &Q) {
  Intrinsic::ID IID = F->getIntrinsicID();
  Type *ReturnType = F->getReturnType();
  unsigned BitWidth = ReturnType->getScalarSizeInBits();
  switch (IID) {
  case Intrinsic::abs:
    // abs(abs(x)) -> abs(x). We don't need to worry about the nsw arg here.
    // It is always ok to pick the earlier abs. We'll just lose nsw if its only
    // on the outer abs.
    if (match(Op0, m_Intrinsic<Intrinsic::abs>(m_Value(), m_Value())))
      return Op0;
    break;

  case Intrinsic::cttz: {
    Value *X;
    if (match(Op0, m_Shl(m_One(), m_Value(X))))
      return X;
    break;
  }
  case Intrinsic::ctlz: {
    Value *X;
    if (match(Op0, m_LShr(m_Negative(), m_Value(X))))
      return X;
    if (match(Op0, m_AShr(m_Negative(), m_Value())))
      return Constant::getNullValue(ReturnType);
    break;
  }
  case Intrinsic::smax:
  case Intrinsic::smin:
  case Intrinsic::umax:
  case Intrinsic::umin: {
    // If the arguments are the same, this is a no-op.
    if (Op0 == Op1)
      return Op0;

    // Canonicalize constant operand as Op1.
    if (isa<Constant>(Op0))
      std::swap(Op0, Op1);

    // Assume undef is the limit value.
    if (Q.isUndefValue(Op1))
      return ConstantInt::get(
          ReturnType, MinMaxIntrinsic::getSaturationPoint(IID, BitWidth));

    const APInt *C;
    if (match(Op1, m_APIntAllowUndef(C))) {
      // Clamp to limit value. For example:
      // umax(i8 %x, i8 255) --> 255
      if (*C == MinMaxIntrinsic::getSaturationPoint(IID, BitWidth))
        return ConstantInt::get(ReturnType, *C);

      // If the constant op is the opposite of the limit value, the other must
      // be larger/smaller or equal. For example:
      // umin(i8 %x, i8 255) --> %x
      if (*C == MinMaxIntrinsic::getSaturationPoint(
                    getInverseMinMaxIntrinsic(IID), BitWidth))
        return Op0;

      // Remove nested call if constant operands allow it. Example:
      // max (max X, 7), 5 -> max X, 7
      auto *MinMax0 = dyn_cast<IntrinsicInst>(Op0);
      if (MinMax0 && MinMax0->getIntrinsicID() == IID) {
        // TODO: loosen undef/splat restrictions for vector constants.
        Value *M00 = MinMax0->getOperand(0), *M01 = MinMax0->getOperand(1);
        const APInt *InnerC;
        if ((match(M00, m_APInt(InnerC)) || match(M01, m_APInt(InnerC))) &&
            ICmpInst::compare(*InnerC, *C,
                              ICmpInst::getNonStrictPredicate(
                                  MinMaxIntrinsic::getPredicate(IID))))
          return Op0;
      }
    }

    if (Value *V = foldMinMaxSharedOp(IID, Op0, Op1))
      return V;
    if (Value *V = foldMinMaxSharedOp(IID, Op1, Op0))
      return V;

    ICmpInst::Predicate Pred =
        ICmpInst::getNonStrictPredicate(MinMaxIntrinsic::getPredicate(IID));
    if (isICmpTrue(Pred, Op0, Op1, Q.getWithoutUndef(), RecursionLimit))
      return Op0;
    if (isICmpTrue(Pred, Op1, Op0, Q.getWithoutUndef(), RecursionLimit))
      return Op1;

    if (Optional<bool> Imp =
            isImpliedByDomCondition(Pred, Op0, Op1, Q.CxtI, Q.DL))
      return *Imp ? Op0 : Op1;
    if (Optional<bool> Imp =
            isImpliedByDomCondition(Pred, Op1, Op0, Q.CxtI, Q.DL))
      return *Imp ? Op1 : Op0;

    break;
  }
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
    // X - X -> { 0, false }
    // X - undef -> { 0, false }
    // undef - X -> { 0, false }
    if (Op0 == Op1 || Q.isUndefValue(Op0) || Q.isUndefValue(Op1))
      return Constant::getNullValue(ReturnType);
    break;
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
    // X + undef -> { -1, false }
    // undef + x -> { -1, false }
    if (Q.isUndefValue(Op0) || Q.isUndefValue(Op1)) {
      return ConstantStruct::get(
          cast<StructType>(ReturnType),
          {Constant::getAllOnesValue(ReturnType->getStructElementType(0)),
           Constant::getNullValue(ReturnType->getStructElementType(1))});
    }
    break;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    // 0 * X -> { 0, false }
    // X * 0 -> { 0, false }
    if (match(Op0, m_Zero()) || match(Op1, m_Zero()))
      return Constant::getNullValue(ReturnType);
    // undef * X -> { 0, false }
    // X * undef -> { 0, false }
    if (Q.isUndefValue(Op0) || Q.isUndefValue(Op1))
      return Constant::getNullValue(ReturnType);
    break;
  case Intrinsic::uadd_sat:
    // sat(MAX + X) -> MAX
    // sat(X + MAX) -> MAX
    if (match(Op0, m_AllOnes()) || match(Op1, m_AllOnes()))
      return Constant::getAllOnesValue(ReturnType);
    LLVM_FALLTHROUGH;
  case Intrinsic::sadd_sat:
    // sat(X + undef) -> -1
    // sat(undef + X) -> -1
    // For unsigned: Assume undef is MAX, thus we saturate to MAX (-1).
    // For signed: Assume undef is ~X, in which case X + ~X = -1.
    if (Q.isUndefValue(Op0) || Q.isUndefValue(Op1))
      return Constant::getAllOnesValue(ReturnType);

    // X + 0 -> X
    if (match(Op1, m_Zero()))
      return Op0;
    // 0 + X -> X
    if (match(Op0, m_Zero()))
      return Op1;
    break;
  case Intrinsic::usub_sat:
    // sat(0 - X) -> 0, sat(X - MAX) -> 0
    if (match(Op0, m_Zero()) || match(Op1, m_AllOnes()))
      return Constant::getNullValue(ReturnType);
    LLVM_FALLTHROUGH;
  case Intrinsic::ssub_sat:
    // X - X -> 0, X - undef -> 0, undef - X -> 0
    if (Op0 == Op1 || Q.isUndefValue(Op0) || Q.isUndefValue(Op1))
      return Constant::getNullValue(ReturnType);
    // X - 0 -> X
    if (match(Op1, m_Zero()))
      return Op0;
    break;
  case Intrinsic::load_relative:
    if (auto *C0 = dyn_cast<Constant>(Op0))
      if (auto *C1 = dyn_cast<Constant>(Op1))
        return SimplifyRelativeLoad(C0, C1, Q.DL);
    break;
  case Intrinsic::powi:
    if (auto *Power = dyn_cast<ConstantInt>(Op1)) {
      // powi(x, 0) -> 1.0
      if (Power->isZero())
        return ConstantFP::get(Op0->getType(), 1.0);
      // powi(x, 1) -> x
      if (Power->isOne())
        return Op0;
    }
    break;
  case Intrinsic::copysign:
    // copysign X, X --> X
    if (Op0 == Op1)
      return Op0;
    // copysign -X, X --> X
    // copysign X, -X --> -X
    if (match(Op0, m_FNeg(m_Specific(Op1))) ||
        match(Op1, m_FNeg(m_Specific(Op0))))
      return Op1;
    break;
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
  case Intrinsic::maximum:
  case Intrinsic::minimum: {
    // If the arguments are the same, this is a no-op.
    if (Op0 == Op1) return Op0;

    // Canonicalize constant operand as Op1.
    if (isa<Constant>(Op0))
      std::swap(Op0, Op1);

    // If an argument is undef, return the other argument.
    if (Q.isUndefValue(Op1))
      return Op0;

    bool PropagateNaN = IID == Intrinsic::minimum || IID == Intrinsic::maximum;
    bool IsMin = IID == Intrinsic::minimum || IID == Intrinsic::minnum;

    // minnum(X, nan) -> X
    // maxnum(X, nan) -> X
    // minimum(X, nan) -> nan
    // maximum(X, nan) -> nan
    if (match(Op1, m_NaN()))
      return PropagateNaN ? propagateNaN(cast<Constant>(Op1)) : Op0;

    // In the following folds, inf can be replaced with the largest finite
    // float, if the ninf flag is set.
    const APFloat *C;
    if (match(Op1, m_APFloat(C)) &&
        (C->isInfinity() || (Q.CxtI->hasNoInfs() && C->isLargest()))) {
      // minnum(X, -inf) -> -inf
      // maxnum(X, +inf) -> +inf
      // minimum(X, -inf) -> -inf if nnan
      // maximum(X, +inf) -> +inf if nnan
      if (C->isNegative() == IsMin && (!PropagateNaN || Q.CxtI->hasNoNaNs()))
        return ConstantFP::get(ReturnType, *C);

      // minnum(X, +inf) -> X if nnan
      // maxnum(X, -inf) -> X if nnan
      // minimum(X, +inf) -> X
      // maximum(X, -inf) -> X
      if (C->isNegative() != IsMin && (PropagateNaN || Q.CxtI->hasNoNaNs()))
        return Op0;
    }

    // Min/max of the same operation with common operand:
    // m(m(X, Y)), X --> m(X, Y) (4 commuted variants)
    if (auto *M0 = dyn_cast<IntrinsicInst>(Op0))
      if (M0->getIntrinsicID() == IID &&
          (M0->getOperand(0) == Op1 || M0->getOperand(1) == Op1))
        return Op0;
    if (auto *M1 = dyn_cast<IntrinsicInst>(Op1))
      if (M1->getIntrinsicID() == IID &&
          (M1->getOperand(0) == Op0 || M1->getOperand(1) == Op0))
        return Op1;

    break;
  }
  case Intrinsic::experimental_vector_extract: {
    Type *ReturnType = F->getReturnType();

    // (extract_vector (insert_vector _, X, 0), 0) -> X
    unsigned IdxN = cast<ConstantInt>(Op1)->getZExtValue();
    Value *X = nullptr;
    if (match(Op0, m_Intrinsic<Intrinsic::experimental_vector_insert>(
                       m_Value(), m_Value(X), m_Zero())) &&
        IdxN == 0 && X->getType() == ReturnType)
      return X;

    break;
  }
  default:
    break;
  }

  return nullptr;
}

static Value *simplifyIntrinsic(CallBase *Call, const SimplifyQuery &Q) {

  unsigned NumOperands = Call->arg_size();
  Function *F = cast<Function>(Call->getCalledFunction());
  Intrinsic::ID IID = F->getIntrinsicID();

  // Most of the intrinsics with no operands have some kind of side effect.
  // Don't simplify.
  if (!NumOperands) {
    switch (IID) {
    case Intrinsic::vscale: {
      // Call may not be inserted into the IR yet at point of calling simplify.
      if (!Call->getParent() || !Call->getParent()->getParent())
        return nullptr;
      auto Attr = Call->getFunction()->getFnAttribute(Attribute::VScaleRange);
      if (!Attr.isValid())
        return nullptr;
      unsigned VScaleMin = Attr.getVScaleRangeMin();
      Optional<unsigned> VScaleMax = Attr.getVScaleRangeMax();
      if (VScaleMax && VScaleMin == VScaleMax)
        return ConstantInt::get(F->getReturnType(), VScaleMin);
      return nullptr;
    }
    default:
      return nullptr;
    }
  }

  if (NumOperands == 1)
    return simplifyUnaryIntrinsic(F, Call->getArgOperand(0), Q);

  if (NumOperands == 2)
    return simplifyBinaryIntrinsic(F, Call->getArgOperand(0),
                                   Call->getArgOperand(1), Q);

  // Handle intrinsics with 3 or more arguments.
  switch (IID) {
  case Intrinsic::masked_load:
  case Intrinsic::masked_gather: {
    Value *MaskArg = Call->getArgOperand(2);
    Value *PassthruArg = Call->getArgOperand(3);
    // If the mask is all zeros or undef, the "passthru" argument is the result.
    if (maskIsAllZeroOrUndef(MaskArg))
      return PassthruArg;
    return nullptr;
  }
  case Intrinsic::fshl:
  case Intrinsic::fshr: {
    Value *Op0 = Call->getArgOperand(0), *Op1 = Call->getArgOperand(1),
          *ShAmtArg = Call->getArgOperand(2);

    // If both operands are undef, the result is undef.
    if (Q.isUndefValue(Op0) && Q.isUndefValue(Op1))
      return UndefValue::get(F->getReturnType());

    // If shift amount is undef, assume it is zero.
    if (Q.isUndefValue(ShAmtArg))
      return Call->getArgOperand(IID == Intrinsic::fshl ? 0 : 1);

    const APInt *ShAmtC;
    if (match(ShAmtArg, m_APInt(ShAmtC))) {
      // If there's effectively no shift, return the 1st arg or 2nd arg.
      APInt BitWidth = APInt(ShAmtC->getBitWidth(), ShAmtC->getBitWidth());
      if (ShAmtC->urem(BitWidth).isZero())
        return Call->getArgOperand(IID == Intrinsic::fshl ? 0 : 1);
    }

    // Rotating zero by anything is zero.
    if (match(Op0, m_Zero()) && match(Op1, m_Zero()))
      return ConstantInt::getNullValue(F->getReturnType());

    // Rotating -1 by anything is -1.
    if (match(Op0, m_AllOnes()) && match(Op1, m_AllOnes()))
      return ConstantInt::getAllOnesValue(F->getReturnType());

    return nullptr;
  }
  case Intrinsic::experimental_constrained_fma: {
    Value *Op0 = Call->getArgOperand(0);
    Value *Op1 = Call->getArgOperand(1);
    Value *Op2 = Call->getArgOperand(2);
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    if (Value *V = simplifyFPOp({Op0, Op1, Op2}, {}, Q,
                                FPI->getExceptionBehavior().getValue(),
                                FPI->getRoundingMode().getValue()))
      return V;
    return nullptr;
  }
  case Intrinsic::fma:
  case Intrinsic::fmuladd: {
    Value *Op0 = Call->getArgOperand(0);
    Value *Op1 = Call->getArgOperand(1);
    Value *Op2 = Call->getArgOperand(2);
    if (Value *V = simplifyFPOp({Op0, Op1, Op2}, {}, Q, fp::ebIgnore,
                                RoundingMode::NearestTiesToEven))
      return V;
    return nullptr;
  }
  case Intrinsic::smul_fix:
  case Intrinsic::smul_fix_sat: {
    Value *Op0 = Call->getArgOperand(0);
    Value *Op1 = Call->getArgOperand(1);
    Value *Op2 = Call->getArgOperand(2);
    Type *ReturnType = F->getReturnType();

    // Canonicalize constant operand as Op1 (ConstantFolding handles the case
    // when both Op0 and Op1 are constant so we do not care about that special
    // case here).
    if (isa<Constant>(Op0))
      std::swap(Op0, Op1);

    // X * 0 -> 0
    if (match(Op1, m_Zero()))
      return Constant::getNullValue(ReturnType);

    // X * undef -> 0
    if (Q.isUndefValue(Op1))
      return Constant::getNullValue(ReturnType);

    // X * (1 << Scale) -> X
    APInt ScaledOne =
        APInt::getOneBitSet(ReturnType->getScalarSizeInBits(),
                            cast<ConstantInt>(Op2)->getZExtValue());
    if (ScaledOne.isNonNegative() && match(Op1, m_SpecificInt(ScaledOne)))
      return Op0;

    return nullptr;
  }
  case Intrinsic::experimental_vector_insert: {
    Value *Vec = Call->getArgOperand(0);
    Value *SubVec = Call->getArgOperand(1);
    Value *Idx = Call->getArgOperand(2);
    Type *ReturnType = F->getReturnType();

    // (insert_vector Y, (extract_vector X, 0), 0) -> X
    // where: Y is X, or Y is undef
    unsigned IdxN = cast<ConstantInt>(Idx)->getZExtValue();
    Value *X = nullptr;
    if (match(SubVec, m_Intrinsic<Intrinsic::experimental_vector_extract>(
                          m_Value(X), m_Zero())) &&
        (Q.isUndefValue(Vec) || Vec == X) && IdxN == 0 &&
        X->getType() == ReturnType)
      return X;

    return nullptr;
  }
  case Intrinsic::experimental_constrained_fadd: {
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    return SimplifyFAddInst(FPI->getArgOperand(0), FPI->getArgOperand(1),
                            FPI->getFastMathFlags(), Q,
                            FPI->getExceptionBehavior().getValue(),
                            FPI->getRoundingMode().getValue());
    break;
  }
  case Intrinsic::experimental_constrained_fsub: {
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    return SimplifyFSubInst(FPI->getArgOperand(0), FPI->getArgOperand(1),
                            FPI->getFastMathFlags(), Q,
                            FPI->getExceptionBehavior().getValue(),
                            FPI->getRoundingMode().getValue());
    break;
  }
  case Intrinsic::experimental_constrained_fmul: {
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    return SimplifyFMulInst(FPI->getArgOperand(0), FPI->getArgOperand(1),
                            FPI->getFastMathFlags(), Q,
                            FPI->getExceptionBehavior().getValue(),
                            FPI->getRoundingMode().getValue());
    break;
  }
  case Intrinsic::experimental_constrained_fdiv: {
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    return SimplifyFDivInst(FPI->getArgOperand(0), FPI->getArgOperand(1),
                            FPI->getFastMathFlags(), Q,
                            FPI->getExceptionBehavior().getValue(),
                            FPI->getRoundingMode().getValue());
    break;
  }
  case Intrinsic::experimental_constrained_frem: {
    auto *FPI = cast<ConstrainedFPIntrinsic>(Call);
    return SimplifyFRemInst(FPI->getArgOperand(0), FPI->getArgOperand(1),
                            FPI->getFastMathFlags(), Q,
                            FPI->getExceptionBehavior().getValue(),
                            FPI->getRoundingMode().getValue());
    break;
  }
  default:
    return nullptr;
  }
}

static Value *tryConstantFoldCall(CallBase *Call, const SimplifyQuery &Q) {
  auto *F = dyn_cast<Function>(Call->getCalledOperand());
  if (!F || !canConstantFoldCallTo(Call, F))
    return nullptr;

  SmallVector<Constant *, 4> ConstantArgs;
  unsigned NumArgs = Call->arg_size();
  ConstantArgs.reserve(NumArgs);
  for (auto &Arg : Call->args()) {
    Constant *C = dyn_cast<Constant>(&Arg);
    if (!C) {
      if (isa<MetadataAsValue>(Arg.get()))
        continue;
      return nullptr;
    }
    ConstantArgs.push_back(C);
  }

  return ConstantFoldCall(Call, F, ConstantArgs, Q.TLI);
}

Value *llvm::SimplifyCall(CallBase *Call, const SimplifyQuery &Q) {
  // musttail calls can only be simplified if they are also DCEd.
  // As we can't guarantee this here, don't simplify them.
  if (Call->isMustTailCall())
    return nullptr;

  // call undef -> poison
  // call null -> poison
  Value *Callee = Call->getCalledOperand();
  if (isa<UndefValue>(Callee) || isa<ConstantPointerNull>(Callee))
    return PoisonValue::get(Call->getType());

  if (Value *V = tryConstantFoldCall(Call, Q))
    return V;

  auto *F = dyn_cast<Function>(Callee);
  if (F && F->isIntrinsic())
    if (Value *Ret = simplifyIntrinsic(Call, Q))
      return Ret;

  return nullptr;
}

/// Given operands for a Freeze, see if we can fold the result.
static Value *SimplifyFreezeInst(Value *Op0, const SimplifyQuery &Q) {
  // Use a utility function defined in ValueTracking.
  if (llvm::isGuaranteedNotToBeUndefOrPoison(Op0, Q.AC, Q.CxtI, Q.DT))
    return Op0;
  // We have room for improvement.
  return nullptr;
}

Value *llvm::SimplifyFreezeInst(Value *Op0, const SimplifyQuery &Q) {
  return ::SimplifyFreezeInst(Op0, Q);
}

static Value *SimplifyLoadInst(LoadInst *LI, Value *PtrOp,
                               const SimplifyQuery &Q) {
  if (LI->isVolatile())
    return nullptr;

  APInt Offset(Q.DL.getIndexTypeSizeInBits(PtrOp->getType()), 0);
  auto *PtrOpC = dyn_cast<Constant>(PtrOp);
  // Try to convert operand into a constant by stripping offsets while looking
  // through invariant.group intrinsics. Don't bother if the underlying object
  // is not constant, as calculating GEP offsets is expensive.
  if (!PtrOpC && isa<Constant>(getUnderlyingObject(PtrOp))) {
    PtrOp = PtrOp->stripAndAccumulateConstantOffsets(
        Q.DL, Offset, /* AllowNonInbounts */ true,
        /* AllowInvariantGroup */ true);
    // Index size may have changed due to address space casts.
    Offset = Offset.sextOrTrunc(Q.DL.getIndexTypeSizeInBits(PtrOp->getType()));
    PtrOpC = dyn_cast<Constant>(PtrOp);
  }

  if (PtrOpC)
    return ConstantFoldLoadFromConstPtr(PtrOpC, LI->getType(), Offset, Q.DL);
  return nullptr;
}

/// See if we can compute a simplified version of this instruction.
/// If not, this returns null.

static Value *simplifyInstructionWithOperands(Instruction *I,
                                              ArrayRef<Value *> NewOps,
                                              const SimplifyQuery &SQ,
                                              OptimizationRemarkEmitter *ORE) {
  const SimplifyQuery Q = SQ.CxtI ? SQ : SQ.getWithInstruction(I);
  Value *Result = nullptr;

  switch (I->getOpcode()) {
  default:
    if (llvm::all_of(NewOps, [](Value *V) { return isa<Constant>(V); })) {
      SmallVector<Constant *, 8> NewConstOps(NewOps.size());
      transform(NewOps, NewConstOps.begin(),
                [](Value *V) { return cast<Constant>(V); });
      Result = ConstantFoldInstOperands(I, NewConstOps, Q.DL, Q.TLI);
    }
    break;
  case Instruction::FNeg:
    Result = SimplifyFNegInst(NewOps[0], I->getFastMathFlags(), Q);
    break;
  case Instruction::FAdd:
    Result = SimplifyFAddInst(NewOps[0], NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::Add:
    Result = SimplifyAddInst(
        NewOps[0], NewOps[1], Q.IIQ.hasNoSignedWrap(cast<BinaryOperator>(I)),
        Q.IIQ.hasNoUnsignedWrap(cast<BinaryOperator>(I)), Q);
    break;
  case Instruction::FSub:
    Result = SimplifyFSubInst(NewOps[0], NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::Sub:
    Result = SimplifySubInst(
        NewOps[0], NewOps[1], Q.IIQ.hasNoSignedWrap(cast<BinaryOperator>(I)),
        Q.IIQ.hasNoUnsignedWrap(cast<BinaryOperator>(I)), Q);
    break;
  case Instruction::FMul:
    Result = SimplifyFMulInst(NewOps[0], NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::Mul:
    Result = SimplifyMulInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::SDiv:
    Result = SimplifySDivInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::UDiv:
    Result = SimplifyUDivInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::FDiv:
    Result = SimplifyFDivInst(NewOps[0], NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::SRem:
    Result = SimplifySRemInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::URem:
    Result = SimplifyURemInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::FRem:
    Result = SimplifyFRemInst(NewOps[0], NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::Shl:
    Result = SimplifyShlInst(
        NewOps[0], NewOps[1], Q.IIQ.hasNoSignedWrap(cast<BinaryOperator>(I)),
        Q.IIQ.hasNoUnsignedWrap(cast<BinaryOperator>(I)), Q);
    break;
  case Instruction::LShr:
    Result = SimplifyLShrInst(NewOps[0], NewOps[1],
                              Q.IIQ.isExact(cast<BinaryOperator>(I)), Q);
    break;
  case Instruction::AShr:
    Result = SimplifyAShrInst(NewOps[0], NewOps[1],
                              Q.IIQ.isExact(cast<BinaryOperator>(I)), Q);
    break;
  case Instruction::And:
    Result = SimplifyAndInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::Or:
    Result = SimplifyOrInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::Xor:
    Result = SimplifyXorInst(NewOps[0], NewOps[1], Q);
    break;
  case Instruction::ICmp:
    Result = SimplifyICmpInst(cast<ICmpInst>(I)->getPredicate(), NewOps[0],
                              NewOps[1], Q);
    break;
  case Instruction::FCmp:
    Result = SimplifyFCmpInst(cast<FCmpInst>(I)->getPredicate(), NewOps[0],
                              NewOps[1], I->getFastMathFlags(), Q);
    break;
  case Instruction::Select:
    Result = SimplifySelectInst(NewOps[0], NewOps[1], NewOps[2], Q);
    break;
  case Instruction::GetElementPtr: {
    auto *GEPI = cast<GetElementPtrInst>(I);
    Result =
        SimplifyGEPInst(GEPI->getSourceElementType(), NewOps[0],
                        makeArrayRef(NewOps).slice(1), GEPI->isInBounds(), Q);
    break;
  }
  case Instruction::InsertValue: {
    InsertValueInst *IV = cast<InsertValueInst>(I);
    Result = SimplifyInsertValueInst(NewOps[0], NewOps[1], IV->getIndices(), Q);
    break;
  }
  case Instruction::InsertElement: {
    Result = SimplifyInsertElementInst(NewOps[0], NewOps[1], NewOps[2], Q);
    break;
  }
  case Instruction::ExtractValue: {
    auto *EVI = cast<ExtractValueInst>(I);
    Result = SimplifyExtractValueInst(NewOps[0], EVI->getIndices(), Q);
    break;
  }
  case Instruction::ExtractElement: {
    Result = SimplifyExtractElementInst(NewOps[0], NewOps[1], Q);
    break;
  }
  case Instruction::ShuffleVector: {
    auto *SVI = cast<ShuffleVectorInst>(I);
    Result = SimplifyShuffleVectorInst(
        NewOps[0], NewOps[1], SVI->getShuffleMask(), SVI->getType(), Q);
    break;
  }
  case Instruction::PHI:
    Result = SimplifyPHINode(cast<PHINode>(I), NewOps, Q);
    break;
  case Instruction::Call: {
    // TODO: Use NewOps
    Result = SimplifyCall(cast<CallInst>(I), Q);
    break;
  }
  case Instruction::Freeze:
    Result = llvm::SimplifyFreezeInst(NewOps[0], Q);
    break;
#define HANDLE_CAST_INST(num, opc, clas) case Instruction::opc:
#include "llvm/IR/Instruction.def"
#undef HANDLE_CAST_INST
    Result = SimplifyCastInst(I->getOpcode(), NewOps[0], I->getType(), Q);
    break;
  case Instruction::Alloca:
    // No simplifications for Alloca and it can't be constant folded.
    Result = nullptr;
    break;
  case Instruction::Load:
    Result = SimplifyLoadInst(cast<LoadInst>(I), NewOps[0], Q);
    break;
  }

  /// If called on unreachable code, the above logic may report that the
  /// instruction simplified to itself.  Make life easier for users by
  /// detecting that case here, returning a safe value instead.
  return Result == I ? UndefValue::get(I->getType()) : Result;
}

Value *llvm::SimplifyInstructionWithOperands(Instruction *I,
                                             ArrayRef<Value *> NewOps,
                                             const SimplifyQuery &SQ,
                                             OptimizationRemarkEmitter *ORE) {
  assert(NewOps.size() == I->getNumOperands() &&
         "Number of operands should match the instruction!");
  return ::simplifyInstructionWithOperands(I, NewOps, SQ, ORE);
}

Value *llvm::SimplifyInstruction(Instruction *I, const SimplifyQuery &SQ,
                                 OptimizationRemarkEmitter *ORE) {
  SmallVector<Value *, 8> Ops(I->operands());
  return ::simplifyInstructionWithOperands(I, Ops, SQ, ORE);
}

/// Implementation of recursive simplification through an instruction's
/// uses.
///
/// This is the common implementation of the recursive simplification routines.
/// If we have a pre-simplified value in 'SimpleV', that is forcibly used to
/// replace the instruction 'I'. Otherwise, we simply add 'I' to the list of
/// instructions to process and attempt to simplify it using
/// InstructionSimplify. Recursively visited users which could not be
/// simplified themselves are to the optional UnsimplifiedUsers set for
/// further processing by the caller.
///
/// This routine returns 'true' only when *it* simplifies something. The passed
/// in simplified value does not count toward this.
static bool replaceAndRecursivelySimplifyImpl(
    Instruction *I, Value *SimpleV, const TargetLibraryInfo *TLI,
    const DominatorTree *DT, AssumptionCache *AC,
    SmallSetVector<Instruction *, 8> *UnsimplifiedUsers = nullptr) {
  bool Simplified = false;
  SmallSetVector<Instruction *, 8> Worklist;
  const DataLayout &DL = I->getModule()->getDataLayout();

  // If we have an explicit value to collapse to, do that round of the
  // simplification loop by hand initially.
  if (SimpleV) {
    for (User *U : I->users())
      if (U != I)
        Worklist.insert(cast<Instruction>(U));

    // Replace the instruction with its simplified value.
    I->replaceAllUsesWith(SimpleV);

    // Gracefully handle edge cases where the instruction is not wired into any
    // parent block.
    if (I->getParent() && !I->isEHPad() && !I->isTerminator() &&
        !I->mayHaveSideEffects())
      I->eraseFromParent();
  } else {
    Worklist.insert(I);
  }

  // Note that we must test the size on each iteration, the worklist can grow.
  for (unsigned Idx = 0; Idx != Worklist.size(); ++Idx) {
    I = Worklist[Idx];

    // See if this instruction simplifies.
    SimpleV = SimplifyInstruction(I, {DL, TLI, DT, AC});
    if (!SimpleV) {
      if (UnsimplifiedUsers)
        UnsimplifiedUsers->insert(I);
      continue;
    }

    Simplified = true;

    // Stash away all the uses of the old instruction so we can check them for
    // recursive simplifications after a RAUW. This is cheaper than checking all
    // uses of To on the recursive step in most cases.
    for (User *U : I->users())
      Worklist.insert(cast<Instruction>(U));

    // Replace the instruction with its simplified value.
    I->replaceAllUsesWith(SimpleV);

    // Gracefully handle edge cases where the instruction is not wired into any
    // parent block.
    if (I->getParent() && !I->isEHPad() && !I->isTerminator() &&
        !I->mayHaveSideEffects())
      I->eraseFromParent();
  }
  return Simplified;
}

bool llvm::replaceAndRecursivelySimplify(
    Instruction *I, Value *SimpleV, const TargetLibraryInfo *TLI,
    const DominatorTree *DT, AssumptionCache *AC,
    SmallSetVector<Instruction *, 8> *UnsimplifiedUsers) {
  assert(I != SimpleV && "replaceAndRecursivelySimplify(X,X) is not valid!");
  assert(SimpleV && "Must provide a simplified value.");
  return replaceAndRecursivelySimplifyImpl(I, SimpleV, TLI, DT, AC,
                                           UnsimplifiedUsers);
}

namespace llvm {
const SimplifyQuery getBestSimplifyQuery(Pass &P, Function &F) {
  auto *DTWP = P.getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *TLIWP = P.getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
  auto *TLI = TLIWP ? &TLIWP->getTLI(F) : nullptr;
  auto *ACWP = P.getAnalysisIfAvailable<AssumptionCacheTracker>();
  auto *AC = ACWP ? &ACWP->getAssumptionCache(F) : nullptr;
  return {F.getParent()->getDataLayout(), TLI, DT, AC};
}

const SimplifyQuery getBestSimplifyQuery(LoopStandardAnalysisResults &AR,
                                         const DataLayout &DL) {
  return {DL, &AR.TLI, &AR.DT, &AR.AC};
}

template <class T, class... TArgs>
const SimplifyQuery getBestSimplifyQuery(AnalysisManager<T, TArgs...> &AM,
                                         Function &F) {
  auto *DT = AM.template getCachedResult<DominatorTreeAnalysis>(F);
  auto *TLI = AM.template getCachedResult<TargetLibraryAnalysis>(F);
  auto *AC = AM.template getCachedResult<AssumptionAnalysis>(F);
  return {F.getParent()->getDataLayout(), TLI, DT, AC};
}
template const SimplifyQuery getBestSimplifyQuery(AnalysisManager<Function> &,
                                                  Function &);
}

void InstSimplifyFolder::anchor() {}
