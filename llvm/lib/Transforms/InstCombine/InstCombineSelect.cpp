//===- InstCombineSelect.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitSelect function.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CmpInstAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/OverflowInstAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <cassert>
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

static Value *createMinMax(InstCombiner::BuilderTy &Builder,
                           SelectPatternFlavor SPF, Value *A, Value *B) {
  CmpInst::Predicate Pred = getMinMaxPred(SPF);
  assert(CmpInst::isIntPredicate(Pred) && "Expected integer predicate");
  return Builder.CreateSelect(Builder.CreateICmp(Pred, A, B), A, B);
}

/// Replace a select operand based on an equality comparison with the identity
/// constant of a binop.
static Instruction *foldSelectBinOpIdentity(SelectInst &Sel,
                                            const TargetLibraryInfo &TLI,
                                            InstCombinerImpl &IC) {
  // The select condition must be an equality compare with a constant operand.
  Value *X;
  Constant *C;
  CmpInst::Predicate Pred;
  if (!match(Sel.getCondition(), m_Cmp(Pred, m_Value(X), m_Constant(C))))
    return nullptr;

  bool IsEq;
  if (ICmpInst::isEquality(Pred))
    IsEq = Pred == ICmpInst::ICMP_EQ;
  else if (Pred == FCmpInst::FCMP_OEQ)
    IsEq = true;
  else if (Pred == FCmpInst::FCMP_UNE)
    IsEq = false;
  else
    return nullptr;

  // A select operand must be a binop.
  BinaryOperator *BO;
  if (!match(Sel.getOperand(IsEq ? 1 : 2), m_BinOp(BO)))
    return nullptr;

  // The compare constant must be the identity constant for that binop.
  // If this a floating-point compare with 0.0, any zero constant will do.
  Type *Ty = BO->getType();
  Constant *IdC = ConstantExpr::getBinOpIdentity(BO->getOpcode(), Ty, true);
  if (IdC != C) {
    if (!IdC || !CmpInst::isFPPredicate(Pred))
      return nullptr;
    if (!match(IdC, m_AnyZeroFP()) || !match(C, m_AnyZeroFP()))
      return nullptr;
  }

  // Last, match the compare variable operand with a binop operand.
  Value *Y;
  if (!BO->isCommutative() && !match(BO, m_BinOp(m_Value(Y), m_Specific(X))))
    return nullptr;
  if (!match(BO, m_c_BinOp(m_Value(Y), m_Specific(X))))
    return nullptr;

  // +0.0 compares equal to -0.0, and so it does not behave as required for this
  // transform. Bail out if we can not exclude that possibility.
  if (isa<FPMathOperator>(BO))
    if (!BO->hasNoSignedZeros() && !CannotBeNegativeZero(Y, &TLI))
      return nullptr;

  // BO = binop Y, X
  // S = { select (cmp eq X, C), BO, ? } or { select (cmp ne X, C), ?, BO }
  // =>
  // S = { select (cmp eq X, C),  Y, ? } or { select (cmp ne X, C), ?,  Y }
  return IC.replaceOperand(Sel, IsEq ? 1 : 2, Y);
}

/// This folds:
///  select (icmp eq (and X, C1)), TC, FC
///    iff C1 is a power 2 and the difference between TC and FC is a power-of-2.
/// To something like:
///  (shr (and (X, C1)), (log2(C1) - log2(TC-FC))) + FC
/// Or:
///  (shl (and (X, C1)), (log2(TC-FC) - log2(C1))) + FC
/// With some variations depending if FC is larger than TC, or the shift
/// isn't needed, or the bit widths don't match.
static Value *foldSelectICmpAnd(SelectInst &Sel, ICmpInst *Cmp,
                                InstCombiner::BuilderTy &Builder) {
  const APInt *SelTC, *SelFC;
  if (!match(Sel.getTrueValue(), m_APInt(SelTC)) ||
      !match(Sel.getFalseValue(), m_APInt(SelFC)))
    return nullptr;

  // If this is a vector select, we need a vector compare.
  Type *SelType = Sel.getType();
  if (SelType->isVectorTy() != Cmp->getType()->isVectorTy())
    return nullptr;

  Value *V;
  APInt AndMask;
  bool CreateAnd = false;
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  if (ICmpInst::isEquality(Pred)) {
    if (!match(Cmp->getOperand(1), m_Zero()))
      return nullptr;

    V = Cmp->getOperand(0);
    const APInt *AndRHS;
    if (!match(V, m_And(m_Value(), m_Power2(AndRHS))))
      return nullptr;

    AndMask = *AndRHS;
  } else if (decomposeBitTestICmp(Cmp->getOperand(0), Cmp->getOperand(1),
                                  Pred, V, AndMask)) {
    assert(ICmpInst::isEquality(Pred) && "Not equality test?");
    if (!AndMask.isPowerOf2())
      return nullptr;

    CreateAnd = true;
  } else {
    return nullptr;
  }

  // In general, when both constants are non-zero, we would need an offset to
  // replace the select. This would require more instructions than we started
  // with. But there's one special-case that we handle here because it can
  // simplify/reduce the instructions.
  APInt TC = *SelTC;
  APInt FC = *SelFC;
  if (!TC.isNullValue() && !FC.isNullValue()) {
    // If the select constants differ by exactly one bit and that's the same
    // bit that is masked and checked by the select condition, the select can
    // be replaced by bitwise logic to set/clear one bit of the constant result.
    if (TC.getBitWidth() != AndMask.getBitWidth() || (TC ^ FC) != AndMask)
      return nullptr;
    if (CreateAnd) {
      // If we have to create an 'and', then we must kill the cmp to not
      // increase the instruction count.
      if (!Cmp->hasOneUse())
        return nullptr;
      V = Builder.CreateAnd(V, ConstantInt::get(SelType, AndMask));
    }
    bool ExtraBitInTC = TC.ugt(FC);
    if (Pred == ICmpInst::ICMP_EQ) {
      // If the masked bit in V is clear, clear or set the bit in the result:
      // (V & AndMaskC) == 0 ? TC : FC --> (V & AndMaskC) ^ TC
      // (V & AndMaskC) == 0 ? TC : FC --> (V & AndMaskC) | TC
      Constant *C = ConstantInt::get(SelType, TC);
      return ExtraBitInTC ? Builder.CreateXor(V, C) : Builder.CreateOr(V, C);
    }
    if (Pred == ICmpInst::ICMP_NE) {
      // If the masked bit in V is set, set or clear the bit in the result:
      // (V & AndMaskC) != 0 ? TC : FC --> (V & AndMaskC) | FC
      // (V & AndMaskC) != 0 ? TC : FC --> (V & AndMaskC) ^ FC
      Constant *C = ConstantInt::get(SelType, FC);
      return ExtraBitInTC ? Builder.CreateOr(V, C) : Builder.CreateXor(V, C);
    }
    llvm_unreachable("Only expecting equality predicates");
  }

  // Make sure one of the select arms is a power-of-2.
  if (!TC.isPowerOf2() && !FC.isPowerOf2())
    return nullptr;

  // Determine which shift is needed to transform result of the 'and' into the
  // desired result.
  const APInt &ValC = !TC.isNullValue() ? TC : FC;
  unsigned ValZeros = ValC.logBase2();
  unsigned AndZeros = AndMask.logBase2();

  // Insert the 'and' instruction on the input to the truncate.
  if (CreateAnd)
    V = Builder.CreateAnd(V, ConstantInt::get(V->getType(), AndMask));

  // If types don't match, we can still convert the select by introducing a zext
  // or a trunc of the 'and'.
  if (ValZeros > AndZeros) {
    V = Builder.CreateZExtOrTrunc(V, SelType);
    V = Builder.CreateShl(V, ValZeros - AndZeros);
  } else if (ValZeros < AndZeros) {
    V = Builder.CreateLShr(V, AndZeros - ValZeros);
    V = Builder.CreateZExtOrTrunc(V, SelType);
  } else {
    V = Builder.CreateZExtOrTrunc(V, SelType);
  }

  // Okay, now we know that everything is set up, we just don't know whether we
  // have a icmp_ne or icmp_eq and whether the true or false val is the zero.
  bool ShouldNotVal = !TC.isNullValue();
  ShouldNotVal ^= Pred == ICmpInst::ICMP_NE;
  if (ShouldNotVal)
    V = Builder.CreateXor(V, ValC);

  return V;
}

/// We want to turn code that looks like this:
///   %C = or %A, %B
///   %D = select %cond, %C, %A
/// into:
///   %C = select %cond, %B, 0
///   %D = or %A, %C
///
/// Assuming that the specified instruction is an operand to the select, return
/// a bitmask indicating which operands of this instruction are foldable if they
/// equal the other incoming value of the select.
static unsigned getSelectFoldableOperands(BinaryOperator *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return 3;              // Can fold through either operand.
  case Instruction::Sub:   // Can only fold on the amount subtracted.
  case Instruction::Shl:   // Can only fold on the shift amount.
  case Instruction::LShr:
  case Instruction::AShr:
    return 1;
  default:
    return 0;              // Cannot fold
  }
}

/// We have (select c, TI, FI), and we know that TI and FI have the same opcode.
Instruction *InstCombinerImpl::foldSelectOpOp(SelectInst &SI, Instruction *TI,
                                              Instruction *FI) {
  // Don't break up min/max patterns. The hasOneUse checks below prevent that
  // for most cases, but vector min/max with bitcasts can be transformed. If the
  // one-use restrictions are eased for other patterns, we still don't want to
  // obfuscate min/max.
  if ((match(&SI, m_SMin(m_Value(), m_Value())) ||
       match(&SI, m_SMax(m_Value(), m_Value())) ||
       match(&SI, m_UMin(m_Value(), m_Value())) ||
       match(&SI, m_UMax(m_Value(), m_Value()))))
    return nullptr;

  // If this is a cast from the same type, merge.
  Value *Cond = SI.getCondition();
  Type *CondTy = Cond->getType();
  if (TI->getNumOperands() == 1 && TI->isCast()) {
    Type *FIOpndTy = FI->getOperand(0)->getType();
    if (TI->getOperand(0)->getType() != FIOpndTy)
      return nullptr;

    // The select condition may be a vector. We may only change the operand
    // type if the vector width remains the same (and matches the condition).
    if (auto *CondVTy = dyn_cast<VectorType>(CondTy)) {
      if (!FIOpndTy->isVectorTy() ||
          CondVTy->getElementCount() !=
              cast<VectorType>(FIOpndTy)->getElementCount())
        return nullptr;

      // TODO: If the backend knew how to deal with casts better, we could
      // remove this limitation. For now, there's too much potential to create
      // worse codegen by promoting the select ahead of size-altering casts
      // (PR28160).
      //
      // Note that ValueTracking's matchSelectPattern() looks through casts
      // without checking 'hasOneUse' when it matches min/max patterns, so this
      // transform may end up happening anyway.
      if (TI->getOpcode() != Instruction::BitCast &&
          (!TI->hasOneUse() || !FI->hasOneUse()))
        return nullptr;
    } else if (!TI->hasOneUse() || !FI->hasOneUse()) {
      // TODO: The one-use restrictions for a scalar select could be eased if
      // the fold of a select in visitLoadInst() was enhanced to match a pattern
      // that includes a cast.
      return nullptr;
    }

    // Fold this by inserting a select from the input values.
    Value *NewSI =
        Builder.CreateSelect(Cond, TI->getOperand(0), FI->getOperand(0),
                             SI.getName() + ".v", &SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI,
                            TI->getType());
  }

  // Cond ? -X : -Y --> -(Cond ? X : Y)
  Value *X, *Y;
  if (match(TI, m_FNeg(m_Value(X))) && match(FI, m_FNeg(m_Value(Y))) &&
      (TI->hasOneUse() || FI->hasOneUse())) {
    Value *NewSel = Builder.CreateSelect(Cond, X, Y, SI.getName() + ".v", &SI);
    return UnaryOperator::CreateFNegFMF(NewSel, TI);
  }

  // Min/max intrinsic with a common operand can have the common operand pulled
  // after the select. This is the same transform as below for binops, but
  // specialized for intrinsic matching and without the restrictive uses clause.
  auto *TII = dyn_cast<IntrinsicInst>(TI);
  auto *FII = dyn_cast<IntrinsicInst>(FI);
  if (TII && FII && TII->getIntrinsicID() == FII->getIntrinsicID() &&
      (TII->hasOneUse() || FII->hasOneUse())) {
    Value *T0, *T1, *F0, *F1;
    if (match(TII, m_MaxOrMin(m_Value(T0), m_Value(T1))) &&
        match(FII, m_MaxOrMin(m_Value(F0), m_Value(F1)))) {
      if (T0 == F0) {
        Value *NewSel = Builder.CreateSelect(Cond, T1, F1, "minmaxop", &SI);
        return CallInst::Create(TII->getCalledFunction(), {NewSel, T0});
      }
      if (T0 == F1) {
        Value *NewSel = Builder.CreateSelect(Cond, T1, F0, "minmaxop", &SI);
        return CallInst::Create(TII->getCalledFunction(), {NewSel, T0});
      }
      if (T1 == F0) {
        Value *NewSel = Builder.CreateSelect(Cond, T0, F1, "minmaxop", &SI);
        return CallInst::Create(TII->getCalledFunction(), {NewSel, T1});
      }
      if (T1 == F1) {
        Value *NewSel = Builder.CreateSelect(Cond, T0, F0, "minmaxop", &SI);
        return CallInst::Create(TII->getCalledFunction(), {NewSel, T1});
      }
    }
  }

  // Only handle binary operators (including two-operand getelementptr) with
  // one-use here. As with the cast case above, it may be possible to relax the
  // one-use constraint, but that needs be examined carefully since it may not
  // reduce the total number of instructions.
  if (TI->getNumOperands() != 2 || FI->getNumOperands() != 2 ||
      (!isa<BinaryOperator>(TI) && !isa<GetElementPtrInst>(TI)) ||
      !TI->hasOneUse() || !FI->hasOneUse())
    return nullptr;

  // Figure out if the operations have any operands in common.
  Value *MatchOp, *OtherOpT, *OtherOpF;
  bool MatchIsOpZero;
  if (TI->getOperand(0) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = false;
  } else if (!TI->isCommutative()) {
    return nullptr;
  } else if (TI->getOperand(0) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else {
    return nullptr;
  }

  // If the select condition is a vector, the operands of the original select's
  // operands also must be vectors. This may not be the case for getelementptr
  // for example.
  if (CondTy->isVectorTy() && (!OtherOpT->getType()->isVectorTy() ||
                               !OtherOpF->getType()->isVectorTy()))
    return nullptr;

  // If we reach here, they do have operations in common.
  Value *NewSI = Builder.CreateSelect(Cond, OtherOpT, OtherOpF,
                                      SI.getName() + ".v", &SI);
  Value *Op0 = MatchIsOpZero ? MatchOp : NewSI;
  Value *Op1 = MatchIsOpZero ? NewSI : MatchOp;
  if (auto *BO = dyn_cast<BinaryOperator>(TI)) {
    BinaryOperator *NewBO = BinaryOperator::Create(BO->getOpcode(), Op0, Op1);
    NewBO->copyIRFlags(TI);
    NewBO->andIRFlags(FI);
    return NewBO;
  }
  if (auto *TGEP = dyn_cast<GetElementPtrInst>(TI)) {
    auto *FGEP = cast<GetElementPtrInst>(FI);
    Type *ElementType = TGEP->getResultElementType();
    return TGEP->isInBounds() && FGEP->isInBounds()
               ? GetElementPtrInst::CreateInBounds(ElementType, Op0, {Op1})
               : GetElementPtrInst::Create(ElementType, Op0, {Op1});
  }
  llvm_unreachable("Expected BinaryOperator or GEP");
  return nullptr;
}

static bool isSelect01(const APInt &C1I, const APInt &C2I) {
  if (!C1I.isNullValue() && !C2I.isNullValue()) // One side must be zero.
    return false;
  return C1I.isOneValue() || C1I.isAllOnesValue() ||
         C2I.isOneValue() || C2I.isAllOnesValue();
}

/// Try to fold the select into one of the operands to allow further
/// optimization.
Instruction *InstCombinerImpl::foldSelectIntoOp(SelectInst &SI, Value *TrueVal,
                                                Value *FalseVal) {
  // See the comment above GetSelectFoldableOperands for a description of the
  // transformation we are doing here.
  if (auto *TVI = dyn_cast<BinaryOperator>(TrueVal)) {
    if (TVI->hasOneUse() && !isa<Constant>(FalseVal)) {
      if (unsigned SFO = getSelectFoldableOperands(TVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = ConstantExpr::getBinOpIdentity(TVI->getOpcode(),
                                                       TVI->getType(), true);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          const APInt *OOpC;
          bool OOpIsAPInt = match(OOp, m_APInt(OOpC));
          if (!isa<Constant>(OOp) ||
              (OOpIsAPInt && isSelect01(C->getUniqueInteger(), *OOpC))) {
            Value *NewSel = Builder.CreateSelect(SI.getCondition(), OOp, C);
            NewSel->takeName(TVI);
            BinaryOperator *BO = BinaryOperator::Create(TVI->getOpcode(),
                                                        FalseVal, NewSel);
            BO->copyIRFlags(TVI);
            return BO;
          }
        }
      }
    }
  }

  if (auto *FVI = dyn_cast<BinaryOperator>(FalseVal)) {
    if (FVI->hasOneUse() && !isa<Constant>(TrueVal)) {
      if (unsigned SFO = getSelectFoldableOperands(FVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = ConstantExpr::getBinOpIdentity(FVI->getOpcode(),
                                                       FVI->getType(), true);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          const APInt *OOpC;
          bool OOpIsAPInt = match(OOp, m_APInt(OOpC));
          if (!isa<Constant>(OOp) ||
              (OOpIsAPInt && isSelect01(C->getUniqueInteger(), *OOpC))) {
            Value *NewSel = Builder.CreateSelect(SI.getCondition(), C, OOp);
            NewSel->takeName(FVI);
            BinaryOperator *BO = BinaryOperator::Create(FVI->getOpcode(),
                                                        TrueVal, NewSel);
            BO->copyIRFlags(FVI);
            return BO;
          }
        }
      }
    }
  }

  return nullptr;
}

/// We want to turn:
///   (select (icmp eq (and X, Y), 0), (and (lshr X, Z), 1), 1)
/// into:
///   zext (icmp ne i32 (and X, (or Y, (shl 1, Z))), 0)
/// Note:
///   Z may be 0 if lshr is missing.
/// Worst-case scenario is that we will replace 5 instructions with 5 different
/// instructions, but we got rid of select.
static Instruction *foldSelectICmpAndAnd(Type *SelType, const ICmpInst *Cmp,
                                         Value *TVal, Value *FVal,
                                         InstCombiner::BuilderTy &Builder) {
  if (!(Cmp->hasOneUse() && Cmp->getOperand(0)->hasOneUse() &&
        Cmp->getPredicate() == ICmpInst::ICMP_EQ &&
        match(Cmp->getOperand(1), m_Zero()) && match(FVal, m_One())))
    return nullptr;

  // The TrueVal has general form of:  and %B, 1
  Value *B;
  if (!match(TVal, m_OneUse(m_And(m_Value(B), m_One()))))
    return nullptr;

  // Where %B may be optionally shifted:  lshr %X, %Z.
  Value *X, *Z;
  const bool HasShift = match(B, m_OneUse(m_LShr(m_Value(X), m_Value(Z))));
  if (!HasShift)
    X = B;

  Value *Y;
  if (!match(Cmp->getOperand(0), m_c_And(m_Specific(X), m_Value(Y))))
    return nullptr;

  // ((X & Y) == 0) ? ((X >> Z) & 1) : 1 --> (X & (Y | (1 << Z))) != 0
  // ((X & Y) == 0) ? (X & 1) : 1 --> (X & (Y | 1)) != 0
  Constant *One = ConstantInt::get(SelType, 1);
  Value *MaskB = HasShift ? Builder.CreateShl(One, Z) : One;
  Value *FullMask = Builder.CreateOr(Y, MaskB);
  Value *MaskedX = Builder.CreateAnd(X, FullMask);
  Value *ICmpNeZero = Builder.CreateIsNotNull(MaskedX);
  return new ZExtInst(ICmpNeZero, SelType);
}

/// We want to turn:
///   (select (icmp sgt x, C), lshr (X, Y), ashr (X, Y)); iff C s>= -1
///   (select (icmp slt x, C), ashr (X, Y), lshr (X, Y)); iff C s>= 0
/// into:
///   ashr (X, Y)
static Value *foldSelectICmpLshrAshr(const ICmpInst *IC, Value *TrueVal,
                                     Value *FalseVal,
                                     InstCombiner::BuilderTy &Builder) {
  ICmpInst::Predicate Pred = IC->getPredicate();
  Value *CmpLHS = IC->getOperand(0);
  Value *CmpRHS = IC->getOperand(1);
  if (!CmpRHS->getType()->isIntOrIntVectorTy())
    return nullptr;

  Value *X, *Y;
  unsigned Bitwidth = CmpRHS->getType()->getScalarSizeInBits();
  if ((Pred != ICmpInst::ICMP_SGT ||
       !match(CmpRHS,
              m_SpecificInt_ICMP(ICmpInst::ICMP_SGE, APInt(Bitwidth, -1)))) &&
      (Pred != ICmpInst::ICMP_SLT ||
       !match(CmpRHS,
              m_SpecificInt_ICMP(ICmpInst::ICMP_SGE, APInt(Bitwidth, 0)))))
    return nullptr;

  // Canonicalize so that ashr is in FalseVal.
  if (Pred == ICmpInst::ICMP_SLT)
    std::swap(TrueVal, FalseVal);

  if (match(TrueVal, m_LShr(m_Value(X), m_Value(Y))) &&
      match(FalseVal, m_AShr(m_Specific(X), m_Specific(Y))) &&
      match(CmpLHS, m_Specific(X))) {
    const auto *Ashr = cast<Instruction>(FalseVal);
    // if lshr is not exact and ashr is, this new ashr must not be exact.
    bool IsExact = Ashr->isExact() && cast<Instruction>(TrueVal)->isExact();
    return Builder.CreateAShr(X, Y, IC->getName(), IsExact);
  }

  return nullptr;
}

/// We want to turn:
///   (select (icmp eq (and X, C1), 0), Y, (or Y, C2))
/// into:
///   (or (shl (and X, C1), C3), Y)
/// iff:
///   C1 and C2 are both powers of 2
/// where:
///   C3 = Log(C2) - Log(C1)
///
/// This transform handles cases where:
/// 1. The icmp predicate is inverted
/// 2. The select operands are reversed
/// 3. The magnitude of C2 and C1 are flipped
static Value *foldSelectICmpAndOr(const ICmpInst *IC, Value *TrueVal,
                                  Value *FalseVal,
                                  InstCombiner::BuilderTy &Builder) {
  // Only handle integer compares. Also, if this is a vector select, we need a
  // vector compare.
  if (!TrueVal->getType()->isIntOrIntVectorTy() ||
      TrueVal->getType()->isVectorTy() != IC->getType()->isVectorTy())
    return nullptr;

  Value *CmpLHS = IC->getOperand(0);
  Value *CmpRHS = IC->getOperand(1);

  Value *V;
  unsigned C1Log;
  bool IsEqualZero;
  bool NeedAnd = false;
  if (IC->isEquality()) {
    if (!match(CmpRHS, m_Zero()))
      return nullptr;

    const APInt *C1;
    if (!match(CmpLHS, m_And(m_Value(), m_Power2(C1))))
      return nullptr;

    V = CmpLHS;
    C1Log = C1->logBase2();
    IsEqualZero = IC->getPredicate() == ICmpInst::ICMP_EQ;
  } else if (IC->getPredicate() == ICmpInst::ICMP_SLT ||
             IC->getPredicate() == ICmpInst::ICMP_SGT) {
    // We also need to recognize (icmp slt (trunc (X)), 0) and
    // (icmp sgt (trunc (X)), -1).
    IsEqualZero = IC->getPredicate() == ICmpInst::ICMP_SGT;
    if ((IsEqualZero && !match(CmpRHS, m_AllOnes())) ||
        (!IsEqualZero && !match(CmpRHS, m_Zero())))
      return nullptr;

    if (!match(CmpLHS, m_OneUse(m_Trunc(m_Value(V)))))
      return nullptr;

    C1Log = CmpLHS->getType()->getScalarSizeInBits() - 1;
    NeedAnd = true;
  } else {
    return nullptr;
  }

  const APInt *C2;
  bool OrOnTrueVal = false;
  bool OrOnFalseVal = match(FalseVal, m_Or(m_Specific(TrueVal), m_Power2(C2)));
  if (!OrOnFalseVal)
    OrOnTrueVal = match(TrueVal, m_Or(m_Specific(FalseVal), m_Power2(C2)));

  if (!OrOnFalseVal && !OrOnTrueVal)
    return nullptr;

  Value *Y = OrOnFalseVal ? TrueVal : FalseVal;

  unsigned C2Log = C2->logBase2();

  bool NeedXor = (!IsEqualZero && OrOnFalseVal) || (IsEqualZero && OrOnTrueVal);
  bool NeedShift = C1Log != C2Log;
  bool NeedZExtTrunc = Y->getType()->getScalarSizeInBits() !=
                       V->getType()->getScalarSizeInBits();

  // Make sure we don't create more instructions than we save.
  Value *Or = OrOnFalseVal ? FalseVal : TrueVal;
  if ((NeedShift + NeedXor + NeedZExtTrunc) >
      (IC->hasOneUse() + Or->hasOneUse()))
    return nullptr;

  if (NeedAnd) {
    // Insert the AND instruction on the input to the truncate.
    APInt C1 = APInt::getOneBitSet(V->getType()->getScalarSizeInBits(), C1Log);
    V = Builder.CreateAnd(V, ConstantInt::get(V->getType(), C1));
  }

  if (C2Log > C1Log) {
    V = Builder.CreateZExtOrTrunc(V, Y->getType());
    V = Builder.CreateShl(V, C2Log - C1Log);
  } else if (C1Log > C2Log) {
    V = Builder.CreateLShr(V, C1Log - C2Log);
    V = Builder.CreateZExtOrTrunc(V, Y->getType());
  } else
    V = Builder.CreateZExtOrTrunc(V, Y->getType());

  if (NeedXor)
    V = Builder.CreateXor(V, *C2);

  return Builder.CreateOr(V, Y);
}

/// Canonicalize a set or clear of a masked set of constant bits to
/// select-of-constants form.
static Instruction *foldSetClearBits(SelectInst &Sel,
                                     InstCombiner::BuilderTy &Builder) {
  Value *Cond = Sel.getCondition();
  Value *T = Sel.getTrueValue();
  Value *F = Sel.getFalseValue();
  Type *Ty = Sel.getType();
  Value *X;
  const APInt *NotC, *C;

  // Cond ? (X & ~C) : (X | C) --> (X & ~C) | (Cond ? 0 : C)
  if (match(T, m_And(m_Value(X), m_APInt(NotC))) &&
      match(F, m_OneUse(m_Or(m_Specific(X), m_APInt(C)))) && *NotC == ~(*C)) {
    Constant *Zero = ConstantInt::getNullValue(Ty);
    Constant *OrC = ConstantInt::get(Ty, *C);
    Value *NewSel = Builder.CreateSelect(Cond, Zero, OrC, "masksel", &Sel);
    return BinaryOperator::CreateOr(T, NewSel);
  }

  // Cond ? (X | C) : (X & ~C) --> (X & ~C) | (Cond ? C : 0)
  if (match(F, m_And(m_Value(X), m_APInt(NotC))) &&
      match(T, m_OneUse(m_Or(m_Specific(X), m_APInt(C)))) && *NotC == ~(*C)) {
    Constant *Zero = ConstantInt::getNullValue(Ty);
    Constant *OrC = ConstantInt::get(Ty, *C);
    Value *NewSel = Builder.CreateSelect(Cond, OrC, Zero, "masksel", &Sel);
    return BinaryOperator::CreateOr(F, NewSel);
  }

  return nullptr;
}

/// Transform patterns such as (a > b) ? a - b : 0 into usub.sat(a, b).
/// There are 8 commuted/swapped variants of this pattern.
/// TODO: Also support a - UMIN(a,b) patterns.
static Value *canonicalizeSaturatedSubtract(const ICmpInst *ICI,
                                            const Value *TrueVal,
                                            const Value *FalseVal,
                                            InstCombiner::BuilderTy &Builder) {
  ICmpInst::Predicate Pred = ICI->getPredicate();
  if (!ICmpInst::isUnsigned(Pred))
    return nullptr;

  // (b > a) ? 0 : a - b -> (b <= a) ? a - b : 0
  if (match(TrueVal, m_Zero())) {
    Pred = ICmpInst::getInversePredicate(Pred);
    std::swap(TrueVal, FalseVal);
  }
  if (!match(FalseVal, m_Zero()))
    return nullptr;

  Value *A = ICI->getOperand(0);
  Value *B = ICI->getOperand(1);
  if (Pred == ICmpInst::ICMP_ULE || Pred == ICmpInst::ICMP_ULT) {
    // (b < a) ? a - b : 0 -> (a > b) ? a - b : 0
    std::swap(A, B);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  assert((Pred == ICmpInst::ICMP_UGE || Pred == ICmpInst::ICMP_UGT) &&
         "Unexpected isUnsigned predicate!");

  // Ensure the sub is of the form:
  //  (a > b) ? a - b : 0 -> usub.sat(a, b)
  //  (a > b) ? b - a : 0 -> -usub.sat(a, b)
  // Checking for both a-b and a+(-b) as a constant.
  bool IsNegative = false;
  const APInt *C;
  if (match(TrueVal, m_Sub(m_Specific(B), m_Specific(A))) ||
      (match(A, m_APInt(C)) &&
       match(TrueVal, m_Add(m_Specific(B), m_SpecificInt(-*C)))))
    IsNegative = true;
  else if (!match(TrueVal, m_Sub(m_Specific(A), m_Specific(B))) &&
           !(match(B, m_APInt(C)) &&
             match(TrueVal, m_Add(m_Specific(A), m_SpecificInt(-*C)))))
    return nullptr;

  // If we are adding a negate and the sub and icmp are used anywhere else, we
  // would end up with more instructions.
  if (IsNegative && !TrueVal->hasOneUse() && !ICI->hasOneUse())
    return nullptr;

  // (a > b) ? a - b : 0 -> usub.sat(a, b)
  // (a > b) ? b - a : 0 -> -usub.sat(a, b)
  Value *Result = Builder.CreateBinaryIntrinsic(Intrinsic::usub_sat, A, B);
  if (IsNegative)
    Result = Builder.CreateNeg(Result);
  return Result;
}

static Value *canonicalizeSaturatedAdd(ICmpInst *Cmp, Value *TVal, Value *FVal,
                                       InstCombiner::BuilderTy &Builder) {
  if (!Cmp->hasOneUse())
    return nullptr;

  // Match unsigned saturated add with constant.
  Value *Cmp0 = Cmp->getOperand(0);
  Value *Cmp1 = Cmp->getOperand(1);
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *X;
  const APInt *C, *CmpC;
  if (Pred == ICmpInst::ICMP_ULT &&
      match(TVal, m_Add(m_Value(X), m_APInt(C))) && X == Cmp0 &&
      match(FVal, m_AllOnes()) && match(Cmp1, m_APInt(CmpC)) && *CmpC == ~*C) {
    // (X u< ~C) ? (X + C) : -1 --> uadd.sat(X, C)
    return Builder.CreateBinaryIntrinsic(
        Intrinsic::uadd_sat, X, ConstantInt::get(X->getType(), *C));
  }

  // Match unsigned saturated add of 2 variables with an unnecessary 'not'.
  // There are 8 commuted variants.
  // Canonicalize -1 (saturated result) to true value of the select.
  if (match(FVal, m_AllOnes())) {
    std::swap(TVal, FVal);
    Pred = CmpInst::getInversePredicate(Pred);
  }
  if (!match(TVal, m_AllOnes()))
    return nullptr;

  // Canonicalize predicate to less-than or less-or-equal-than.
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_UGE) {
    std::swap(Cmp0, Cmp1);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_ULE)
    return nullptr;

  // Match unsigned saturated add of 2 variables with an unnecessary 'not'.
  // Strictness of the comparison is irrelevant.
  Value *Y;
  if (match(Cmp0, m_Not(m_Value(X))) &&
      match(FVal, m_c_Add(m_Specific(X), m_Value(Y))) && Y == Cmp1) {
    // (~X u< Y) ? -1 : (X + Y) --> uadd.sat(X, Y)
    // (~X u< Y) ? -1 : (Y + X) --> uadd.sat(X, Y)
    return Builder.CreateBinaryIntrinsic(Intrinsic::uadd_sat, X, Y);
  }
  // The 'not' op may be included in the sum but not the compare.
  // Strictness of the comparison is irrelevant.
  X = Cmp0;
  Y = Cmp1;
  if (match(FVal, m_c_Add(m_Not(m_Specific(X)), m_Specific(Y)))) {
    // (X u< Y) ? -1 : (~X + Y) --> uadd.sat(~X, Y)
    // (X u< Y) ? -1 : (Y + ~X) --> uadd.sat(Y, ~X)
    BinaryOperator *BO = cast<BinaryOperator>(FVal);
    return Builder.CreateBinaryIntrinsic(
        Intrinsic::uadd_sat, BO->getOperand(0), BO->getOperand(1));
  }
  // The overflow may be detected via the add wrapping round.
  // This is only valid for strict comparison!
  if (Pred == ICmpInst::ICMP_ULT &&
      match(Cmp0, m_c_Add(m_Specific(Cmp1), m_Value(Y))) &&
      match(FVal, m_c_Add(m_Specific(Cmp1), m_Specific(Y)))) {
    // ((X + Y) u< X) ? -1 : (X + Y) --> uadd.sat(X, Y)
    // ((X + Y) u< Y) ? -1 : (X + Y) --> uadd.sat(X, Y)
    return Builder.CreateBinaryIntrinsic(Intrinsic::uadd_sat, Cmp1, Y);
  }

  return nullptr;
}

/// Fold the following code sequence:
/// \code
///   int a = ctlz(x & -x);
//    x ? 31 - a : a;
/// \code
///
/// into:
///   cttz(x)
static Instruction *foldSelectCtlzToCttz(ICmpInst *ICI, Value *TrueVal,
                                         Value *FalseVal,
                                         InstCombiner::BuilderTy &Builder) {
  unsigned BitWidth = TrueVal->getType()->getScalarSizeInBits();
  if (!ICI->isEquality() || !match(ICI->getOperand(1), m_Zero()))
    return nullptr;

  if (ICI->getPredicate() == ICmpInst::ICMP_NE)
    std::swap(TrueVal, FalseVal);

  if (!match(FalseVal,
             m_Xor(m_Deferred(TrueVal), m_SpecificInt(BitWidth - 1))))
    return nullptr;

  if (!match(TrueVal, m_Intrinsic<Intrinsic::ctlz>()))
    return nullptr;

  Value *X = ICI->getOperand(0);
  auto *II = cast<IntrinsicInst>(TrueVal);
  if (!match(II->getOperand(0), m_c_And(m_Specific(X), m_Neg(m_Specific(X)))))
    return nullptr;

  Function *F = Intrinsic::getDeclaration(II->getModule(), Intrinsic::cttz,
                                          II->getType());
  return CallInst::Create(F, {X, II->getArgOperand(1)});
}

/// Attempt to fold a cttz/ctlz followed by a icmp plus select into a single
/// call to cttz/ctlz with flag 'is_zero_undef' cleared.
///
/// For example, we can fold the following code sequence:
/// \code
///   %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 true)
///   %1 = icmp ne i32 %x, 0
///   %2 = select i1 %1, i32 %0, i32 32
/// \code
///
/// into:
///   %0 = tail call i32 @llvm.cttz.i32(i32 %x, i1 false)
static Value *foldSelectCttzCtlz(ICmpInst *ICI, Value *TrueVal, Value *FalseVal,
                                 InstCombiner::BuilderTy &Builder) {
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);

  // Check if the condition value compares a value for equality against zero.
  if (!ICI->isEquality() || !match(CmpRHS, m_Zero()))
    return nullptr;

  Value *SelectArg = FalseVal;
  Value *ValueOnZero = TrueVal;
  if (Pred == ICmpInst::ICMP_NE)
    std::swap(SelectArg, ValueOnZero);

  // Skip zero extend/truncate.
  Value *Count = nullptr;
  if (!match(SelectArg, m_ZExt(m_Value(Count))) &&
      !match(SelectArg, m_Trunc(m_Value(Count))))
    Count = SelectArg;

  // Check that 'Count' is a call to intrinsic cttz/ctlz. Also check that the
  // input to the cttz/ctlz is used as LHS for the compare instruction.
  if (!match(Count, m_Intrinsic<Intrinsic::cttz>(m_Specific(CmpLHS))) &&
      !match(Count, m_Intrinsic<Intrinsic::ctlz>(m_Specific(CmpLHS))))
    return nullptr;

  IntrinsicInst *II = cast<IntrinsicInst>(Count);

  // Check if the value propagated on zero is a constant number equal to the
  // sizeof in bits of 'Count'.
  unsigned SizeOfInBits = Count->getType()->getScalarSizeInBits();
  if (match(ValueOnZero, m_SpecificInt(SizeOfInBits))) {
    // Explicitly clear the 'undef_on_zero' flag. It's always valid to go from
    // true to false on this flag, so we can replace it for all users.
    II->setArgOperand(1, ConstantInt::getFalse(II->getContext()));
    return SelectArg;
  }

  // The ValueOnZero is not the bitwidth. But if the cttz/ctlz (and optional
  // zext/trunc) have one use (ending at the select), the cttz/ctlz result will
  // not be used if the input is zero. Relax to 'undef_on_zero' for that case.
  if (II->hasOneUse() && SelectArg->hasOneUse() &&
      !match(II->getArgOperand(1), m_One()))
    II->setArgOperand(1, ConstantInt::getTrue(II->getContext()));

  return nullptr;
}

/// Return true if we find and adjust an icmp+select pattern where the compare
/// is with a constant that can be incremented or decremented to match the
/// minimum or maximum idiom.
static bool adjustMinMax(SelectInst &Sel, ICmpInst &Cmp) {
  ICmpInst::Predicate Pred = Cmp.getPredicate();
  Value *CmpLHS = Cmp.getOperand(0);
  Value *CmpRHS = Cmp.getOperand(1);
  Value *TrueVal = Sel.getTrueValue();
  Value *FalseVal = Sel.getFalseValue();

  // We may move or edit the compare, so make sure the select is the only user.
  const APInt *CmpC;
  if (!Cmp.hasOneUse() || !match(CmpRHS, m_APInt(CmpC)))
    return false;

  // These transforms only work for selects of integers or vector selects of
  // integer vectors.
  Type *SelTy = Sel.getType();
  auto *SelEltTy = dyn_cast<IntegerType>(SelTy->getScalarType());
  if (!SelEltTy || SelTy->isVectorTy() != Cmp.getType()->isVectorTy())
    return false;

  Constant *AdjustedRHS;
  if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT)
    AdjustedRHS = ConstantInt::get(CmpRHS->getType(), *CmpC + 1);
  else if (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_SLT)
    AdjustedRHS = ConstantInt::get(CmpRHS->getType(), *CmpC - 1);
  else
    return false;

  // X > C ? X : C+1  -->  X < C+1 ? C+1 : X
  // X < C ? X : C-1  -->  X > C-1 ? C-1 : X
  if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
      (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
    ; // Nothing to do here. Values match without any sign/zero extension.
  }
  // Types do not match. Instead of calculating this with mixed types, promote
  // all to the larger type. This enables scalar evolution to analyze this
  // expression.
  else if (CmpRHS->getType()->getScalarSizeInBits() < SelEltTy->getBitWidth()) {
    Constant *SextRHS = ConstantExpr::getSExt(AdjustedRHS, SelTy);

    // X = sext x; x >s c ? X : C+1 --> X = sext x; X <s C+1 ? C+1 : X
    // X = sext x; x <s c ? X : C-1 --> X = sext x; X >s C-1 ? C-1 : X
    // X = sext x; x >u c ? X : C+1 --> X = sext x; X <u C+1 ? C+1 : X
    // X = sext x; x <u c ? X : C-1 --> X = sext x; X >u C-1 ? C-1 : X
    if (match(TrueVal, m_SExt(m_Specific(CmpLHS))) && SextRHS == FalseVal) {
      CmpLHS = TrueVal;
      AdjustedRHS = SextRHS;
    } else if (match(FalseVal, m_SExt(m_Specific(CmpLHS))) &&
               SextRHS == TrueVal) {
      CmpLHS = FalseVal;
      AdjustedRHS = SextRHS;
    } else if (Cmp.isUnsigned()) {
      Constant *ZextRHS = ConstantExpr::getZExt(AdjustedRHS, SelTy);
      // X = zext x; x >u c ? X : C+1 --> X = zext x; X <u C+1 ? C+1 : X
      // X = zext x; x <u c ? X : C-1 --> X = zext x; X >u C-1 ? C-1 : X
      // zext + signed compare cannot be changed:
      //    0xff <s 0x00, but 0x00ff >s 0x0000
      if (match(TrueVal, m_ZExt(m_Specific(CmpLHS))) && ZextRHS == FalseVal) {
        CmpLHS = TrueVal;
        AdjustedRHS = ZextRHS;
      } else if (match(FalseVal, m_ZExt(m_Specific(CmpLHS))) &&
                 ZextRHS == TrueVal) {
        CmpLHS = FalseVal;
        AdjustedRHS = ZextRHS;
      } else {
        return false;
      }
    } else {
      return false;
    }
  } else {
    return false;
  }

  Pred = ICmpInst::getSwappedPredicate(Pred);
  CmpRHS = AdjustedRHS;
  std::swap(FalseVal, TrueVal);
  Cmp.setPredicate(Pred);
  Cmp.setOperand(0, CmpLHS);
  Cmp.setOperand(1, CmpRHS);
  Sel.setOperand(1, TrueVal);
  Sel.setOperand(2, FalseVal);
  Sel.swapProfMetadata();

  // Move the compare instruction right before the select instruction. Otherwise
  // the sext/zext value may be defined after the compare instruction uses it.
  Cmp.moveBefore(&Sel);

  return true;
}

/// If this is an integer min/max (icmp + select) with a constant operand,
/// create the canonical icmp for the min/max operation and canonicalize the
/// constant to the 'false' operand of the select:
/// select (icmp Pred X, C1), C2, X --> select (icmp Pred' X, C2), X, C2
/// Note: if C1 != C2, this will change the icmp constant to the existing
/// constant operand of the select.
static Instruction *canonicalizeMinMaxWithConstant(SelectInst &Sel,
                                                   ICmpInst &Cmp,
                                                   InstCombinerImpl &IC) {
  if (!Cmp.hasOneUse() || !isa<Constant>(Cmp.getOperand(1)))
    return nullptr;

  // Canonicalize the compare predicate based on whether we have min or max.
  Value *LHS, *RHS;
  SelectPatternResult SPR = matchSelectPattern(&Sel, LHS, RHS);
  if (!SelectPatternResult::isMinOrMax(SPR.Flavor))
    return nullptr;

  // Is this already canonical?
  ICmpInst::Predicate CanonicalPred = getMinMaxPred(SPR.Flavor);
  if (Cmp.getOperand(0) == LHS && Cmp.getOperand(1) == RHS &&
      Cmp.getPredicate() == CanonicalPred)
    return nullptr;

  // Bail out on unsimplified X-0 operand (due to some worklist management bug),
  // as this may cause an infinite combine loop. Let the sub be folded first.
  if (match(LHS, m_Sub(m_Value(), m_Zero())) ||
      match(RHS, m_Sub(m_Value(), m_Zero())))
    return nullptr;

  // Create the canonical compare and plug it into the select.
  IC.replaceOperand(Sel, 0, IC.Builder.CreateICmp(CanonicalPred, LHS, RHS));

  // If the select operands did not change, we're done.
  if (Sel.getTrueValue() == LHS && Sel.getFalseValue() == RHS)
    return &Sel;

  // If we are swapping the select operands, swap the metadata too.
  assert(Sel.getTrueValue() == RHS && Sel.getFalseValue() == LHS &&
         "Unexpected results from matchSelectPattern");
  Sel.swapValues();
  Sel.swapProfMetadata();
  return &Sel;
}

static Instruction *canonicalizeAbsNabs(SelectInst &Sel, ICmpInst &Cmp,
                                        InstCombinerImpl &IC) {
  if (!Cmp.hasOneUse() || !isa<Constant>(Cmp.getOperand(1)))
    return nullptr;

  Value *LHS, *RHS;
  SelectPatternFlavor SPF = matchSelectPattern(&Sel, LHS, RHS).Flavor;
  if (SPF != SelectPatternFlavor::SPF_ABS &&
      SPF != SelectPatternFlavor::SPF_NABS)
    return nullptr;

  // Note that NSW flag can only be propagated for normal, non-negated abs!
  bool IntMinIsPoison = SPF == SelectPatternFlavor::SPF_ABS &&
                        match(RHS, m_NSWNeg(m_Specific(LHS)));
  Constant *IntMinIsPoisonC =
      ConstantInt::get(Type::getInt1Ty(Sel.getContext()), IntMinIsPoison);
  Instruction *Abs =
      IC.Builder.CreateBinaryIntrinsic(Intrinsic::abs, LHS, IntMinIsPoisonC);

  if (SPF == SelectPatternFlavor::SPF_NABS)
    return BinaryOperator::CreateNeg(Abs); // Always without NSW flag!

  return IC.replaceInstUsesWith(Sel, Abs);
}

/// If we have a select with an equality comparison, then we know the value in
/// one of the arms of the select. See if substituting this value into an arm
/// and simplifying the result yields the same value as the other arm.
///
/// To make this transform safe, we must drop poison-generating flags
/// (nsw, etc) if we simplified to a binop because the select may be guarding
/// that poison from propagating. If the existing binop already had no
/// poison-generating flags, then this transform can be done by instsimplify.
///
/// Consider:
///   %cmp = icmp eq i32 %x, 2147483647
///   %add = add nsw i32 %x, 1
///   %sel = select i1 %cmp, i32 -2147483648, i32 %add
///
/// We can't replace %sel with %add unless we strip away the flags.
/// TODO: Wrapping flags could be preserved in some cases with better analysis.
Instruction *InstCombinerImpl::foldSelectValueEquivalence(SelectInst &Sel,
                                                          ICmpInst &Cmp) {
  // Value equivalence substitution requires an all-or-nothing replacement.
  // It does not make sense for a vector compare where each lane is chosen
  // independently.
  if (!Cmp.isEquality() || Cmp.getType()->isVectorTy())
    return nullptr;

  // Canonicalize the pattern to ICMP_EQ by swapping the select operands.
  Value *TrueVal = Sel.getTrueValue(), *FalseVal = Sel.getFalseValue();
  bool Swapped = false;
  if (Cmp.getPredicate() == ICmpInst::ICMP_NE) {
    std::swap(TrueVal, FalseVal);
    Swapped = true;
  }

  // In X == Y ? f(X) : Z, try to evaluate f(Y) and replace the operand.
  // Make sure Y cannot be undef though, as we might pick different values for
  // undef in the icmp and in f(Y). Additionally, take care to avoid replacing
  // X == Y ? X : Z with X == Y ? Y : Z, as that would lead to an infinite
  // replacement cycle.
  Value *CmpLHS = Cmp.getOperand(0), *CmpRHS = Cmp.getOperand(1);
  if (TrueVal != CmpLHS &&
      isGuaranteedNotToBeUndefOrPoison(CmpRHS, SQ.AC, &Sel, &DT)) {
    if (Value *V = simplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, SQ,
                                          /* AllowRefinement */ true))
      return replaceOperand(Sel, Swapped ? 2 : 1, V);

    // Even if TrueVal does not simplify, we can directly replace a use of
    // CmpLHS with CmpRHS, as long as the instruction is not used anywhere
    // else and is safe to speculatively execute (we may end up executing it
    // with different operands, which should not cause side-effects or trigger
    // undefined behavior). Only do this if CmpRHS is a constant, as
    // profitability is not clear for other cases.
    // FIXME: The replacement could be performed recursively.
    if (match(CmpRHS, m_ImmConstant()) && !match(CmpLHS, m_ImmConstant()))
      if (auto *I = dyn_cast<Instruction>(TrueVal))
        if (I->hasOneUse() && isSafeToSpeculativelyExecute(I))
          for (Use &U : I->operands())
            if (U == CmpLHS) {
              replaceUse(U, CmpRHS);
              return &Sel;
            }
  }
  if (TrueVal != CmpRHS &&
      isGuaranteedNotToBeUndefOrPoison(CmpLHS, SQ.AC, &Sel, &DT))
    if (Value *V = simplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, SQ,
                                          /* AllowRefinement */ true))
      return replaceOperand(Sel, Swapped ? 2 : 1, V);

  auto *FalseInst = dyn_cast<Instruction>(FalseVal);
  if (!FalseInst)
    return nullptr;

  // InstSimplify already performed this fold if it was possible subject to
  // current poison-generating flags. Try the transform again with
  // poison-generating flags temporarily dropped.
  bool WasNUW = false, WasNSW = false, WasExact = false, WasInBounds = false;
  if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(FalseVal)) {
    WasNUW = OBO->hasNoUnsignedWrap();
    WasNSW = OBO->hasNoSignedWrap();
    FalseInst->setHasNoUnsignedWrap(false);
    FalseInst->setHasNoSignedWrap(false);
  }
  if (auto *PEO = dyn_cast<PossiblyExactOperator>(FalseVal)) {
    WasExact = PEO->isExact();
    FalseInst->setIsExact(false);
  }
  if (auto *GEP = dyn_cast<GetElementPtrInst>(FalseVal)) {
    WasInBounds = GEP->isInBounds();
    GEP->setIsInBounds(false);
  }

  // Try each equivalence substitution possibility.
  // We have an 'EQ' comparison, so the select's false value will propagate.
  // Example:
  // (X == 42) ? 43 : (X + 1) --> (X == 42) ? (X + 1) : (X + 1) --> X + 1
  if (simplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, SQ,
                             /* AllowRefinement */ false) == TrueVal ||
      simplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, SQ,
                             /* AllowRefinement */ false) == TrueVal) {
    return replaceInstUsesWith(Sel, FalseVal);
  }

  // Restore poison-generating flags if the transform did not apply.
  if (WasNUW)
    FalseInst->setHasNoUnsignedWrap();
  if (WasNSW)
    FalseInst->setHasNoSignedWrap();
  if (WasExact)
    FalseInst->setIsExact();
  if (WasInBounds)
    cast<GetElementPtrInst>(FalseInst)->setIsInBounds();

  return nullptr;
}

// See if this is a pattern like:
//   %old_cmp1 = icmp slt i32 %x, C2
//   %old_replacement = select i1 %old_cmp1, i32 %target_low, i32 %target_high
//   %old_x_offseted = add i32 %x, C1
//   %old_cmp0 = icmp ult i32 %old_x_offseted, C0
//   %r = select i1 %old_cmp0, i32 %x, i32 %old_replacement
// This can be rewritten as more canonical pattern:
//   %new_cmp1 = icmp slt i32 %x, -C1
//   %new_cmp2 = icmp sge i32 %x, C0-C1
//   %new_clamped_low = select i1 %new_cmp1, i32 %target_low, i32 %x
//   %r = select i1 %new_cmp2, i32 %target_high, i32 %new_clamped_low
// Iff -C1 s<= C2 s<= C0-C1
// Also ULT predicate can also be UGT iff C0 != -1 (+invert result)
//      SLT predicate can also be SGT iff C2 != INT_MAX (+invert res.)
static Instruction *canonicalizeClampLike(SelectInst &Sel0, ICmpInst &Cmp0,
                                          InstCombiner::BuilderTy &Builder) {
  Value *X = Sel0.getTrueValue();
  Value *Sel1 = Sel0.getFalseValue();

  // First match the condition of the outermost select.
  // Said condition must be one-use.
  if (!Cmp0.hasOneUse())
    return nullptr;
  Value *Cmp00 = Cmp0.getOperand(0);
  Constant *C0;
  if (!match(Cmp0.getOperand(1),
             m_CombineAnd(m_AnyIntegralConstant(), m_Constant(C0))))
    return nullptr;
  // Canonicalize Cmp0 into the form we expect.
  // FIXME: we shouldn't care about lanes that are 'undef' in the end?
  switch (Cmp0.getPredicate()) {
  case ICmpInst::Predicate::ICMP_ULT:
    break; // Great!
  case ICmpInst::Predicate::ICMP_ULE:
    // We'd have to increment C0 by one, and for that it must not have all-ones
    // element, but then it would have been canonicalized to 'ult' before
    // we get here. So we can't do anything useful with 'ule'.
    return nullptr;
  case ICmpInst::Predicate::ICMP_UGT:
    // We want to canonicalize it to 'ult', so we'll need to increment C0,
    // which again means it must not have any all-ones elements.
    if (!match(C0,
               m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE,
                                  APInt::getAllOnesValue(
                                      C0->getType()->getScalarSizeInBits()))))
      return nullptr; // Can't do, have all-ones element[s].
    C0 = InstCombiner::AddOne(C0);
    std::swap(X, Sel1);
    break;
  case ICmpInst::Predicate::ICMP_UGE:
    // The only way we'd get this predicate if this `icmp` has extra uses,
    // but then we won't be able to do this fold.
    return nullptr;
  default:
    return nullptr; // Unknown predicate.
  }

  // Now that we've canonicalized the ICmp, we know the X we expect;
  // the select in other hand should be one-use.
  if (!Sel1->hasOneUse())
    return nullptr;

  // We now can finish matching the condition of the outermost select:
  // it should either be the X itself, or an addition of some constant to X.
  Constant *C1;
  if (Cmp00 == X)
    C1 = ConstantInt::getNullValue(Sel0.getType());
  else if (!match(Cmp00,
                  m_Add(m_Specific(X),
                        m_CombineAnd(m_AnyIntegralConstant(), m_Constant(C1)))))
    return nullptr;

  Value *Cmp1;
  ICmpInst::Predicate Pred1;
  Constant *C2;
  Value *ReplacementLow, *ReplacementHigh;
  if (!match(Sel1, m_Select(m_Value(Cmp1), m_Value(ReplacementLow),
                            m_Value(ReplacementHigh))) ||
      !match(Cmp1,
             m_ICmp(Pred1, m_Specific(X),
                    m_CombineAnd(m_AnyIntegralConstant(), m_Constant(C2)))))
    return nullptr;

  if (!Cmp1->hasOneUse() && (Cmp00 == X || !Cmp00->hasOneUse()))
    return nullptr; // Not enough one-use instructions for the fold.
  // FIXME: this restriction could be relaxed if Cmp1 can be reused as one of
  //        two comparisons we'll need to build.

  // Canonicalize Cmp1 into the form we expect.
  // FIXME: we shouldn't care about lanes that are 'undef' in the end?
  switch (Pred1) {
  case ICmpInst::Predicate::ICMP_SLT:
    break;
  case ICmpInst::Predicate::ICMP_SLE:
    // We'd have to increment C2 by one, and for that it must not have signed
    // max element, but then it would have been canonicalized to 'slt' before
    // we get here. So we can't do anything useful with 'sle'.
    return nullptr;
  case ICmpInst::Predicate::ICMP_SGT:
    // We want to canonicalize it to 'slt', so we'll need to increment C2,
    // which again means it must not have any signed max elements.
    if (!match(C2,
               m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE,
                                  APInt::getSignedMaxValue(
                                      C2->getType()->getScalarSizeInBits()))))
      return nullptr; // Can't do, have signed max element[s].
    C2 = InstCombiner::AddOne(C2);
    LLVM_FALLTHROUGH;
  case ICmpInst::Predicate::ICMP_SGE:
    // Also non-canonical, but here we don't need to change C2,
    // so we don't have any restrictions on C2, so we can just handle it.
    std::swap(ReplacementLow, ReplacementHigh);
    break;
  default:
    return nullptr; // Unknown predicate.
  }

  // The thresholds of this clamp-like pattern.
  auto *ThresholdLowIncl = ConstantExpr::getNeg(C1);
  auto *ThresholdHighExcl = ConstantExpr::getSub(C0, C1);

  // The fold has a precondition 1: C2 s>= ThresholdLow
  auto *Precond1 = ConstantExpr::getICmp(ICmpInst::Predicate::ICMP_SGE, C2,
                                         ThresholdLowIncl);
  if (!match(Precond1, m_One()))
    return nullptr;
  // The fold has a precondition 2: C2 s<= ThresholdHigh
  auto *Precond2 = ConstantExpr::getICmp(ICmpInst::Predicate::ICMP_SLE, C2,
                                         ThresholdHighExcl);
  if (!match(Precond2, m_One()))
    return nullptr;

  // All good, finally emit the new pattern.
  Value *ShouldReplaceLow = Builder.CreateICmpSLT(X, ThresholdLowIncl);
  Value *ShouldReplaceHigh = Builder.CreateICmpSGE(X, ThresholdHighExcl);
  Value *MaybeReplacedLow =
      Builder.CreateSelect(ShouldReplaceLow, ReplacementLow, X);
  Instruction *MaybeReplacedHigh =
      SelectInst::Create(ShouldReplaceHigh, ReplacementHigh, MaybeReplacedLow);

  return MaybeReplacedHigh;
}

// If we have
//  %cmp = icmp [canonical predicate] i32 %x, C0
//  %r = select i1 %cmp, i32 %y, i32 C1
// Where C0 != C1 and %x may be different from %y, see if the constant that we
// will have if we flip the strictness of the predicate (i.e. without changing
// the result) is identical to the C1 in select. If it matches we can change
// original comparison to one with swapped predicate, reuse the constant,
// and swap the hands of select.
static Instruction *
tryToReuseConstantFromSelectInComparison(SelectInst &Sel, ICmpInst &Cmp,
                                         InstCombinerImpl &IC) {
  ICmpInst::Predicate Pred;
  Value *X;
  Constant *C0;
  if (!match(&Cmp, m_OneUse(m_ICmp(
                       Pred, m_Value(X),
                       m_CombineAnd(m_AnyIntegralConstant(), m_Constant(C0))))))
    return nullptr;

  // If comparison predicate is non-relational, we won't be able to do anything.
  if (ICmpInst::isEquality(Pred))
    return nullptr;

  // If comparison predicate is non-canonical, then we certainly won't be able
  // to make it canonical; canonicalizeCmpWithConstant() already tried.
  if (!InstCombiner::isCanonicalPredicate(Pred))
    return nullptr;

  // If the [input] type of comparison and select type are different, lets abort
  // for now. We could try to compare constants with trunc/[zs]ext though.
  if (C0->getType() != Sel.getType())
    return nullptr;

  // FIXME: are there any magic icmp predicate+constant pairs we must not touch?

  Value *SelVal0, *SelVal1; // We do not care which one is from where.
  match(&Sel, m_Select(m_Value(), m_Value(SelVal0), m_Value(SelVal1)));
  // At least one of these values we are selecting between must be a constant
  // else we'll never succeed.
  if (!match(SelVal0, m_AnyIntegralConstant()) &&
      !match(SelVal1, m_AnyIntegralConstant()))
    return nullptr;

  // Does this constant C match any of the `select` values?
  auto MatchesSelectValue = [SelVal0, SelVal1](Constant *C) {
    return C->isElementWiseEqual(SelVal0) || C->isElementWiseEqual(SelVal1);
  };

  // If C0 *already* matches true/false value of select, we are done.
  if (MatchesSelectValue(C0))
    return nullptr;

  // Check the constant we'd have with flipped-strictness predicate.
  auto FlippedStrictness =
      InstCombiner::getFlippedStrictnessPredicateAndConstant(Pred, C0);
  if (!FlippedStrictness)
    return nullptr;

  // If said constant doesn't match either, then there is no hope,
  if (!MatchesSelectValue(FlippedStrictness->second))
    return nullptr;

  // It matched! Lets insert the new comparison just before select.
  InstCombiner::BuilderTy::InsertPointGuard Guard(IC.Builder);
  IC.Builder.SetInsertPoint(&Sel);

  Pred = ICmpInst::getSwappedPredicate(Pred); // Yes, swapped.
  Value *NewCmp = IC.Builder.CreateICmp(Pred, X, FlippedStrictness->second,
                                        Cmp.getName() + ".inv");
  IC.replaceOperand(Sel, 0, NewCmp);
  Sel.swapValues();
  Sel.swapProfMetadata();

  return &Sel;
}

/// Visit a SelectInst that has an ICmpInst as its first operand.
Instruction *InstCombinerImpl::foldSelectInstWithICmp(SelectInst &SI,
                                                      ICmpInst *ICI) {
  if (Instruction *NewSel = foldSelectValueEquivalence(SI, *ICI))
    return NewSel;

  if (Instruction *NewSel = canonicalizeMinMaxWithConstant(SI, *ICI, *this))
    return NewSel;

  if (Instruction *NewAbs = canonicalizeAbsNabs(SI, *ICI, *this))
    return NewAbs;

  if (Instruction *NewAbs = canonicalizeClampLike(SI, *ICI, Builder))
    return NewAbs;

  if (Instruction *NewSel =
          tryToReuseConstantFromSelectInComparison(SI, *ICI, *this))
    return NewSel;

  bool Changed = adjustMinMax(SI, *ICI);

  if (Value *V = foldSelectICmpAnd(SI, ICI, Builder))
    return replaceInstUsesWith(SI, V);

  // NOTE: if we wanted to, this is where to detect integer MIN/MAX
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  if (CmpRHS != CmpLHS && isa<Constant>(CmpRHS)) {
    if (CmpLHS == TrueVal && Pred == ICmpInst::ICMP_EQ) {
      // Transform (X == C) ? X : Y -> (X == C) ? C : Y
      SI.setOperand(1, CmpRHS);
      Changed = true;
    } else if (CmpLHS == FalseVal && Pred == ICmpInst::ICMP_NE) {
      // Transform (X != C) ? Y : X -> (X != C) ? Y : C
      SI.setOperand(2, CmpRHS);
      Changed = true;
    }
  }

  // FIXME: This code is nearly duplicated in InstSimplify. Using/refactoring
  // decomposeBitTestICmp() might help.
  {
    unsigned BitWidth =
        DL.getTypeSizeInBits(TrueVal->getType()->getScalarType());
    APInt MinSignedValue = APInt::getSignedMinValue(BitWidth);
    Value *X;
    const APInt *Y, *C;
    bool TrueWhenUnset;
    bool IsBitTest = false;
    if (ICmpInst::isEquality(Pred) &&
        match(CmpLHS, m_And(m_Value(X), m_Power2(Y))) &&
        match(CmpRHS, m_Zero())) {
      IsBitTest = true;
      TrueWhenUnset = Pred == ICmpInst::ICMP_EQ;
    } else if (Pred == ICmpInst::ICMP_SLT && match(CmpRHS, m_Zero())) {
      X = CmpLHS;
      Y = &MinSignedValue;
      IsBitTest = true;
      TrueWhenUnset = false;
    } else if (Pred == ICmpInst::ICMP_SGT && match(CmpRHS, m_AllOnes())) {
      X = CmpLHS;
      Y = &MinSignedValue;
      IsBitTest = true;
      TrueWhenUnset = true;
    }
    if (IsBitTest) {
      Value *V = nullptr;
      // (X & Y) == 0 ? X : X ^ Y  --> X & ~Y
      if (TrueWhenUnset && TrueVal == X &&
          match(FalseVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder.CreateAnd(X, ~(*Y));
      // (X & Y) != 0 ? X ^ Y : X  --> X & ~Y
      else if (!TrueWhenUnset && FalseVal == X &&
               match(TrueVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder.CreateAnd(X, ~(*Y));
      // (X & Y) == 0 ? X ^ Y : X  --> X | Y
      else if (TrueWhenUnset && FalseVal == X &&
               match(TrueVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder.CreateOr(X, *Y);
      // (X & Y) != 0 ? X : X ^ Y  --> X | Y
      else if (!TrueWhenUnset && TrueVal == X &&
               match(FalseVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder.CreateOr(X, *Y);

      if (V)
        return replaceInstUsesWith(SI, V);
    }
  }

  if (Instruction *V =
          foldSelectICmpAndAnd(SI.getType(), ICI, TrueVal, FalseVal, Builder))
    return V;

  if (Instruction *V = foldSelectCtlzToCttz(ICI, TrueVal, FalseVal, Builder))
    return V;

  if (Value *V = foldSelectICmpAndOr(ICI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  if (Value *V = foldSelectICmpLshrAshr(ICI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  if (Value *V = foldSelectCttzCtlz(ICI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  if (Value *V = canonicalizeSaturatedSubtract(ICI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  if (Value *V = canonicalizeSaturatedAdd(ICI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  return Changed ? &SI : nullptr;
}

/// SI is a select whose condition is a PHI node (but the two may be in
/// different blocks). See if the true/false values (V) are live in all of the
/// predecessor blocks of the PHI. For example, cases like this can't be mapped:
///
///   X = phi [ C1, BB1], [C2, BB2]
///   Y = add
///   Z = select X, Y, 0
///
/// because Y is not live in BB1/BB2.
static bool canSelectOperandBeMappingIntoPredBlock(const Value *V,
                                                   const SelectInst &SI) {
  // If the value is a non-instruction value like a constant or argument, it
  // can always be mapped.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return true;

  // If V is a PHI node defined in the same block as the condition PHI, we can
  // map the arguments.
  const PHINode *CondPHI = cast<PHINode>(SI.getCondition());

  if (const PHINode *VP = dyn_cast<PHINode>(I))
    if (VP->getParent() == CondPHI->getParent())
      return true;

  // Otherwise, if the PHI and select are defined in the same block and if V is
  // defined in a different block, then we can transform it.
  if (SI.getParent() == CondPHI->getParent() &&
      I->getParent() != CondPHI->getParent())
    return true;

  // Otherwise we have a 'hard' case and we can't tell without doing more
  // detailed dominator based analysis, punt.
  return false;
}

/// We have an SPF (e.g. a min or max) of an SPF of the form:
///   SPF2(SPF1(A, B), C)
Instruction *InstCombinerImpl::foldSPFofSPF(Instruction *Inner,
                                            SelectPatternFlavor SPF1, Value *A,
                                            Value *B, Instruction &Outer,
                                            SelectPatternFlavor SPF2,
                                            Value *C) {
  if (Outer.getType() != Inner->getType())
    return nullptr;

  if (C == A || C == B) {
    // MAX(MAX(A, B), B) -> MAX(A, B)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    // TODO: This could be done in instsimplify.
    if (SPF1 == SPF2 && SelectPatternResult::isMinOrMax(SPF1))
      return replaceInstUsesWith(Outer, Inner);

    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    // TODO: This could be done in instsimplify.
    if ((SPF1 == SPF_SMIN && SPF2 == SPF_SMAX) ||
        (SPF1 == SPF_SMAX && SPF2 == SPF_SMIN) ||
        (SPF1 == SPF_UMIN && SPF2 == SPF_UMAX) ||
        (SPF1 == SPF_UMAX && SPF2 == SPF_UMIN))
      return replaceInstUsesWith(Outer, C);
  }

  if (SPF1 == SPF2) {
    const APInt *CB, *CC;
    if (match(B, m_APInt(CB)) && match(C, m_APInt(CC))) {
      // MIN(MIN(A, 23), 97) -> MIN(A, 23)
      // MAX(MAX(A, 97), 23) -> MAX(A, 97)
      // TODO: This could be done in instsimplify.
      if ((SPF1 == SPF_UMIN && CB->ule(*CC)) ||
          (SPF1 == SPF_SMIN && CB->sle(*CC)) ||
          (SPF1 == SPF_UMAX && CB->uge(*CC)) ||
          (SPF1 == SPF_SMAX && CB->sge(*CC)))
        return replaceInstUsesWith(Outer, Inner);

      // MIN(MIN(A, 97), 23) -> MIN(A, 23)
      // MAX(MAX(A, 23), 97) -> MAX(A, 97)
      if ((SPF1 == SPF_UMIN && CB->ugt(*CC)) ||
          (SPF1 == SPF_SMIN && CB->sgt(*CC)) ||
          (SPF1 == SPF_UMAX && CB->ult(*CC)) ||
          (SPF1 == SPF_SMAX && CB->slt(*CC))) {
        Outer.replaceUsesOfWith(Inner, A);
        return &Outer;
      }
    }
  }

  // max(max(A, B), min(A, B)) --> max(A, B)
  // min(min(A, B), max(A, B)) --> min(A, B)
  // TODO: This could be done in instsimplify.
  if (SPF1 == SPF2 &&
      ((SPF1 == SPF_UMIN && match(C, m_c_UMax(m_Specific(A), m_Specific(B)))) ||
       (SPF1 == SPF_SMIN && match(C, m_c_SMax(m_Specific(A), m_Specific(B)))) ||
       (SPF1 == SPF_UMAX && match(C, m_c_UMin(m_Specific(A), m_Specific(B)))) ||
       (SPF1 == SPF_SMAX && match(C, m_c_SMin(m_Specific(A), m_Specific(B))))))
    return replaceInstUsesWith(Outer, Inner);

  // ABS(ABS(X)) -> ABS(X)
  // NABS(NABS(X)) -> NABS(X)
  // TODO: This could be done in instsimplify.
  if (SPF1 == SPF2 && (SPF1 == SPF_ABS || SPF1 == SPF_NABS)) {
    return replaceInstUsesWith(Outer, Inner);
  }

  // ABS(NABS(X)) -> ABS(X)
  // NABS(ABS(X)) -> NABS(X)
  if ((SPF1 == SPF_ABS && SPF2 == SPF_NABS) ||
      (SPF1 == SPF_NABS && SPF2 == SPF_ABS)) {
    SelectInst *SI = cast<SelectInst>(Inner);
    Value *NewSI =
        Builder.CreateSelect(SI->getCondition(), SI->getFalseValue(),
                             SI->getTrueValue(), SI->getName(), SI);
    return replaceInstUsesWith(Outer, NewSI);
  }

  auto IsFreeOrProfitableToInvert =
      [&](Value *V, Value *&NotV, bool &ElidesXor) {
    if (match(V, m_Not(m_Value(NotV)))) {
      // If V has at most 2 uses then we can get rid of the xor operation
      // entirely.
      ElidesXor |= !V->hasNUsesOrMore(3);
      return true;
    }

    if (isFreeToInvert(V, !V->hasNUsesOrMore(3))) {
      NotV = nullptr;
      return true;
    }

    return false;
  };

  Value *NotA, *NotB, *NotC;
  bool ElidesXor = false;

  // MIN(MIN(~A, ~B), ~C) == ~MAX(MAX(A, B), C)
  // MIN(MAX(~A, ~B), ~C) == ~MAX(MIN(A, B), C)
  // MAX(MIN(~A, ~B), ~C) == ~MIN(MAX(A, B), C)
  // MAX(MAX(~A, ~B), ~C) == ~MIN(MIN(A, B), C)
  //
  // This transform is performance neutral if we can elide at least one xor from
  // the set of three operands, since we'll be tacking on an xor at the very
  // end.
  if (SelectPatternResult::isMinOrMax(SPF1) &&
      SelectPatternResult::isMinOrMax(SPF2) &&
      IsFreeOrProfitableToInvert(A, NotA, ElidesXor) &&
      IsFreeOrProfitableToInvert(B, NotB, ElidesXor) &&
      IsFreeOrProfitableToInvert(C, NotC, ElidesXor) && ElidesXor) {
    if (!NotA)
      NotA = Builder.CreateNot(A);
    if (!NotB)
      NotB = Builder.CreateNot(B);
    if (!NotC)
      NotC = Builder.CreateNot(C);

    Value *NewInner = createMinMax(Builder, getInverseMinMaxFlavor(SPF1), NotA,
                                   NotB);
    Value *NewOuter = Builder.CreateNot(
        createMinMax(Builder, getInverseMinMaxFlavor(SPF2), NewInner, NotC));
    return replaceInstUsesWith(Outer, NewOuter);
  }

  return nullptr;
}

/// Turn select C, (X + Y), (X - Y) --> (X + (select C, Y, (-Y))).
/// This is even legal for FP.
static Instruction *foldAddSubSelect(SelectInst &SI,
                                     InstCombiner::BuilderTy &Builder) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  auto *TI = dyn_cast<Instruction>(TrueVal);
  auto *FI = dyn_cast<Instruction>(FalseVal);
  if (!TI || !FI || !TI->hasOneUse() || !FI->hasOneUse())
    return nullptr;

  Instruction *AddOp = nullptr, *SubOp = nullptr;
  if ((TI->getOpcode() == Instruction::Sub &&
       FI->getOpcode() == Instruction::Add) ||
      (TI->getOpcode() == Instruction::FSub &&
       FI->getOpcode() == Instruction::FAdd)) {
    AddOp = FI;
    SubOp = TI;
  } else if ((FI->getOpcode() == Instruction::Sub &&
              TI->getOpcode() == Instruction::Add) ||
             (FI->getOpcode() == Instruction::FSub &&
              TI->getOpcode() == Instruction::FAdd)) {
    AddOp = TI;
    SubOp = FI;
  }

  if (AddOp) {
    Value *OtherAddOp = nullptr;
    if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
      OtherAddOp = AddOp->getOperand(1);
    } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
      OtherAddOp = AddOp->getOperand(0);
    }

    if (OtherAddOp) {
      // So at this point we know we have (Y -> OtherAddOp):
      //        select C, (add X, Y), (sub X, Z)
      Value *NegVal; // Compute -Z
      if (SI.getType()->isFPOrFPVectorTy()) {
        NegVal = Builder.CreateFNeg(SubOp->getOperand(1));
        if (Instruction *NegInst = dyn_cast<Instruction>(NegVal)) {
          FastMathFlags Flags = AddOp->getFastMathFlags();
          Flags &= SubOp->getFastMathFlags();
          NegInst->setFastMathFlags(Flags);
        }
      } else {
        NegVal = Builder.CreateNeg(SubOp->getOperand(1));
      }

      Value *NewTrueOp = OtherAddOp;
      Value *NewFalseOp = NegVal;
      if (AddOp != TI)
        std::swap(NewTrueOp, NewFalseOp);
      Value *NewSel = Builder.CreateSelect(CondVal, NewTrueOp, NewFalseOp,
                                           SI.getName() + ".p", &SI);

      if (SI.getType()->isFPOrFPVectorTy()) {
        Instruction *RI =
            BinaryOperator::CreateFAdd(SubOp->getOperand(0), NewSel);

        FastMathFlags Flags = AddOp->getFastMathFlags();
        Flags &= SubOp->getFastMathFlags();
        RI->setFastMathFlags(Flags);
        return RI;
      } else
        return BinaryOperator::CreateAdd(SubOp->getOperand(0), NewSel);
    }
  }
  return nullptr;
}

/// Turn X + Y overflows ? -1 : X + Y -> uadd_sat X, Y
/// And X - Y overflows ? 0 : X - Y -> usub_sat X, Y
/// Along with a number of patterns similar to:
/// X + Y overflows ? (X < 0 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
/// X - Y overflows ? (X > 0 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
static Instruction *
foldOverflowingAddSubSelect(SelectInst &SI, InstCombiner::BuilderTy &Builder) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  WithOverflowInst *II;
  if (!match(CondVal, m_ExtractValue<1>(m_WithOverflowInst(II))) ||
      !match(FalseVal, m_ExtractValue<0>(m_Specific(II))))
    return nullptr;

  Value *X = II->getLHS();
  Value *Y = II->getRHS();

  auto IsSignedSaturateLimit = [&](Value *Limit, bool IsAdd) {
    Type *Ty = Limit->getType();

    ICmpInst::Predicate Pred;
    Value *TrueVal, *FalseVal, *Op;
    const APInt *C;
    if (!match(Limit, m_Select(m_ICmp(Pred, m_Value(Op), m_APInt(C)),
                               m_Value(TrueVal), m_Value(FalseVal))))
      return false;

    auto IsZeroOrOne = [](const APInt &C) {
      return C.isNullValue() || C.isOneValue();
    };
    auto IsMinMax = [&](Value *Min, Value *Max) {
      APInt MinVal = APInt::getSignedMinValue(Ty->getScalarSizeInBits());
      APInt MaxVal = APInt::getSignedMaxValue(Ty->getScalarSizeInBits());
      return match(Min, m_SpecificInt(MinVal)) &&
             match(Max, m_SpecificInt(MaxVal));
    };

    if (Op != X && Op != Y)
      return false;

    if (IsAdd) {
      // X + Y overflows ? (X <s 0 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (X <s 1 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (Y <s 0 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (Y <s 1 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
      if (Pred == ICmpInst::ICMP_SLT && IsZeroOrOne(*C) &&
          IsMinMax(TrueVal, FalseVal))
        return true;
      // X + Y overflows ? (X >s 0 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (X >s -1 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (Y >s 0 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
      // X + Y overflows ? (Y >s -1 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
      if (Pred == ICmpInst::ICMP_SGT && IsZeroOrOne(*C + 1) &&
          IsMinMax(FalseVal, TrueVal))
        return true;
    } else {
      // X - Y overflows ? (X <s 0 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
      // X - Y overflows ? (X <s -1 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
      if (Op == X && Pred == ICmpInst::ICMP_SLT && IsZeroOrOne(*C + 1) &&
          IsMinMax(TrueVal, FalseVal))
        return true;
      // X - Y overflows ? (X >s -1 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
      // X - Y overflows ? (X >s -2 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
      if (Op == X && Pred == ICmpInst::ICMP_SGT && IsZeroOrOne(*C + 2) &&
          IsMinMax(FalseVal, TrueVal))
        return true;
      // X - Y overflows ? (Y <s 0 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
      // X - Y overflows ? (Y <s 1 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
      if (Op == Y && Pred == ICmpInst::ICMP_SLT && IsZeroOrOne(*C) &&
          IsMinMax(FalseVal, TrueVal))
        return true;
      // X - Y overflows ? (Y >s 0 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
      // X - Y overflows ? (Y >s -1 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
      if (Op == Y && Pred == ICmpInst::ICMP_SGT && IsZeroOrOne(*C + 1) &&
          IsMinMax(TrueVal, FalseVal))
        return true;
    }

    return false;
  };

  Intrinsic::ID NewIntrinsicID;
  if (II->getIntrinsicID() == Intrinsic::uadd_with_overflow &&
      match(TrueVal, m_AllOnes()))
    // X + Y overflows ? -1 : X + Y -> uadd_sat X, Y
    NewIntrinsicID = Intrinsic::uadd_sat;
  else if (II->getIntrinsicID() == Intrinsic::usub_with_overflow &&
           match(TrueVal, m_Zero()))
    // X - Y overflows ? 0 : X - Y -> usub_sat X, Y
    NewIntrinsicID = Intrinsic::usub_sat;
  else if (II->getIntrinsicID() == Intrinsic::sadd_with_overflow &&
           IsSignedSaturateLimit(TrueVal, /*IsAdd=*/true))
    // X + Y overflows ? (X <s 0 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (X <s 1 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (X >s 0 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (X >s -1 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (Y <s 0 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (Y <s 1 ? INTMIN : INTMAX) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (Y >s 0 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
    // X + Y overflows ? (Y >s -1 ? INTMAX : INTMIN) : X + Y --> sadd_sat X, Y
    NewIntrinsicID = Intrinsic::sadd_sat;
  else if (II->getIntrinsicID() == Intrinsic::ssub_with_overflow &&
           IsSignedSaturateLimit(TrueVal, /*IsAdd=*/false))
    // X - Y overflows ? (X <s 0 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (X <s -1 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (X >s -1 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (X >s -2 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (Y <s 0 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (Y <s 1 ? INTMAX : INTMIN) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (Y >s 0 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
    // X - Y overflows ? (Y >s -1 ? INTMIN : INTMAX) : X - Y --> ssub_sat X, Y
    NewIntrinsicID = Intrinsic::ssub_sat;
  else
    return nullptr;

  Function *F =
      Intrinsic::getDeclaration(SI.getModule(), NewIntrinsicID, SI.getType());
  return CallInst::Create(F, {X, Y});
}

Instruction *InstCombinerImpl::foldSelectExtConst(SelectInst &Sel) {
  Constant *C;
  if (!match(Sel.getTrueValue(), m_Constant(C)) &&
      !match(Sel.getFalseValue(), m_Constant(C)))
    return nullptr;

  Instruction *ExtInst;
  if (!match(Sel.getTrueValue(), m_Instruction(ExtInst)) &&
      !match(Sel.getFalseValue(), m_Instruction(ExtInst)))
    return nullptr;

  auto ExtOpcode = ExtInst->getOpcode();
  if (ExtOpcode != Instruction::ZExt && ExtOpcode != Instruction::SExt)
    return nullptr;

  // If we are extending from a boolean type or if we can create a select that
  // has the same size operands as its condition, try to narrow the select.
  Value *X = ExtInst->getOperand(0);
  Type *SmallType = X->getType();
  Value *Cond = Sel.getCondition();
  auto *Cmp = dyn_cast<CmpInst>(Cond);
  if (!SmallType->isIntOrIntVectorTy(1) &&
      (!Cmp || Cmp->getOperand(0)->getType() != SmallType))
    return nullptr;

  // If the constant is the same after truncation to the smaller type and
  // extension to the original type, we can narrow the select.
  Type *SelType = Sel.getType();
  Constant *TruncC = ConstantExpr::getTrunc(C, SmallType);
  Constant *ExtC = ConstantExpr::getCast(ExtOpcode, TruncC, SelType);
  if (ExtC == C && ExtInst->hasOneUse()) {
    Value *TruncCVal = cast<Value>(TruncC);
    if (ExtInst == Sel.getFalseValue())
      std::swap(X, TruncCVal);

    // select Cond, (ext X), C --> ext(select Cond, X, C')
    // select Cond, C, (ext X) --> ext(select Cond, C', X)
    Value *NewSel = Builder.CreateSelect(Cond, X, TruncCVal, "narrow", &Sel);
    return CastInst::Create(Instruction::CastOps(ExtOpcode), NewSel, SelType);
  }

  // If one arm of the select is the extend of the condition, replace that arm
  // with the extension of the appropriate known bool value.
  if (Cond == X) {
    if (ExtInst == Sel.getTrueValue()) {
      // select X, (sext X), C --> select X, -1, C
      // select X, (zext X), C --> select X,  1, C
      Constant *One = ConstantInt::getTrue(SmallType);
      Constant *AllOnesOrOne = ConstantExpr::getCast(ExtOpcode, One, SelType);
      return SelectInst::Create(Cond, AllOnesOrOne, C, "", nullptr, &Sel);
    } else {
      // select X, C, (sext X) --> select X, C, 0
      // select X, C, (zext X) --> select X, C, 0
      Constant *Zero = ConstantInt::getNullValue(SelType);
      return SelectInst::Create(Cond, C, Zero, "", nullptr, &Sel);
    }
  }

  return nullptr;
}

/// Try to transform a vector select with a constant condition vector into a
/// shuffle for easier combining with other shuffles and insert/extract.
static Instruction *canonicalizeSelectToShuffle(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Constant *CondC;
  auto *CondValTy = dyn_cast<FixedVectorType>(CondVal->getType());
  if (!CondValTy || !match(CondVal, m_Constant(CondC)))
    return nullptr;

  unsigned NumElts = CondValTy->getNumElements();
  SmallVector<int, 16> Mask;
  Mask.reserve(NumElts);
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *Elt = CondC->getAggregateElement(i);
    if (!Elt)
      return nullptr;

    if (Elt->isOneValue()) {
      // If the select condition element is true, choose from the 1st vector.
      Mask.push_back(i);
    } else if (Elt->isNullValue()) {
      // If the select condition element is false, choose from the 2nd vector.
      Mask.push_back(i + NumElts);
    } else if (isa<UndefValue>(Elt)) {
      // Undef in a select condition (choose one of the operands) does not mean
      // the same thing as undef in a shuffle mask (any value is acceptable), so
      // give up.
      return nullptr;
    } else {
      // Bail out on a constant expression.
      return nullptr;
    }
  }

  return new ShuffleVectorInst(SI.getTrueValue(), SI.getFalseValue(), Mask);
}

/// If we have a select of vectors with a scalar condition, try to convert that
/// to a vector select by splatting the condition. A splat may get folded with
/// other operations in IR and having all operands of a select be vector types
/// is likely better for vector codegen.
static Instruction *canonicalizeScalarSelectOfVecs(SelectInst &Sel,
                                                   InstCombinerImpl &IC) {
  auto *Ty = dyn_cast<VectorType>(Sel.getType());
  if (!Ty)
    return nullptr;

  // We can replace a single-use extract with constant index.
  Value *Cond = Sel.getCondition();
  if (!match(Cond, m_OneUse(m_ExtractElt(m_Value(), m_ConstantInt()))))
    return nullptr;

  // select (extelt V, Index), T, F --> select (splat V, Index), T, F
  // Splatting the extracted condition reduces code (we could directly create a
  // splat shuffle of the source vector to eliminate the intermediate step).
  return IC.replaceOperand(
      Sel, 0, IC.Builder.CreateVectorSplat(Ty->getElementCount(), Cond));
}

/// Reuse bitcasted operands between a compare and select:
/// select (cmp (bitcast C), (bitcast D)), (bitcast' C), (bitcast' D) -->
/// bitcast (select (cmp (bitcast C), (bitcast D)), (bitcast C), (bitcast D))
static Instruction *foldSelectCmpBitcasts(SelectInst &Sel,
                                          InstCombiner::BuilderTy &Builder) {
  Value *Cond = Sel.getCondition();
  Value *TVal = Sel.getTrueValue();
  Value *FVal = Sel.getFalseValue();

  CmpInst::Predicate Pred;
  Value *A, *B;
  if (!match(Cond, m_Cmp(Pred, m_Value(A), m_Value(B))))
    return nullptr;

  // The select condition is a compare instruction. If the select's true/false
  // values are already the same as the compare operands, there's nothing to do.
  if (TVal == A || TVal == B || FVal == A || FVal == B)
    return nullptr;

  Value *C, *D;
  if (!match(A, m_BitCast(m_Value(C))) || !match(B, m_BitCast(m_Value(D))))
    return nullptr;

  // select (cmp (bitcast C), (bitcast D)), (bitcast TSrc), (bitcast FSrc)
  Value *TSrc, *FSrc;
  if (!match(TVal, m_BitCast(m_Value(TSrc))) ||
      !match(FVal, m_BitCast(m_Value(FSrc))))
    return nullptr;

  // If the select true/false values are *different bitcasts* of the same source
  // operands, make the select operands the same as the compare operands and
  // cast the result. This is the canonical select form for min/max.
  Value *NewSel;
  if (TSrc == C && FSrc == D) {
    // select (cmp (bitcast C), (bitcast D)), (bitcast' C), (bitcast' D) -->
    // bitcast (select (cmp A, B), A, B)
    NewSel = Builder.CreateSelect(Cond, A, B, "", &Sel);
  } else if (TSrc == D && FSrc == C) {
    // select (cmp (bitcast C), (bitcast D)), (bitcast' D), (bitcast' C) -->
    // bitcast (select (cmp A, B), B, A)
    NewSel = Builder.CreateSelect(Cond, B, A, "", &Sel);
  } else {
    return nullptr;
  }
  return CastInst::CreateBitOrPointerCast(NewSel, Sel.getType());
}

/// Try to eliminate select instructions that test the returned flag of cmpxchg
/// instructions.
///
/// If a select instruction tests the returned flag of a cmpxchg instruction and
/// selects between the returned value of the cmpxchg instruction its compare
/// operand, the result of the select will always be equal to its false value.
/// For example:
///
///   %0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
///   %1 = extractvalue { i64, i1 } %0, 1
///   %2 = extractvalue { i64, i1 } %0, 0
///   %3 = select i1 %1, i64 %compare, i64 %2
///   ret i64 %3
///
/// The returned value of the cmpxchg instruction (%2) is the original value
/// located at %ptr prior to any update. If the cmpxchg operation succeeds, %2
/// must have been equal to %compare. Thus, the result of the select is always
/// equal to %2, and the code can be simplified to:
///
///   %0 = cmpxchg i64* %ptr, i64 %compare, i64 %new_value seq_cst seq_cst
///   %1 = extractvalue { i64, i1 } %0, 0
///   ret i64 %1
///
static Value *foldSelectCmpXchg(SelectInst &SI) {
  // A helper that determines if V is an extractvalue instruction whose
  // aggregate operand is a cmpxchg instruction and whose single index is equal
  // to I. If such conditions are true, the helper returns the cmpxchg
  // instruction; otherwise, a nullptr is returned.
  auto isExtractFromCmpXchg = [](Value *V, unsigned I) -> AtomicCmpXchgInst * {
    auto *Extract = dyn_cast<ExtractValueInst>(V);
    if (!Extract)
      return nullptr;
    if (Extract->getIndices()[0] != I)
      return nullptr;
    return dyn_cast<AtomicCmpXchgInst>(Extract->getAggregateOperand());
  };

  // If the select has a single user, and this user is a select instruction that
  // we can simplify, skip the cmpxchg simplification for now.
  if (SI.hasOneUse())
    if (auto *Select = dyn_cast<SelectInst>(SI.user_back()))
      if (Select->getCondition() == SI.getCondition())
        if (Select->getFalseValue() == SI.getTrueValue() ||
            Select->getTrueValue() == SI.getFalseValue())
          return nullptr;

  // Ensure the select condition is the returned flag of a cmpxchg instruction.
  auto *CmpXchg = isExtractFromCmpXchg(SI.getCondition(), 1);
  if (!CmpXchg)
    return nullptr;

  // Check the true value case: The true value of the select is the returned
  // value of the same cmpxchg used by the condition, and the false value is the
  // cmpxchg instruction's compare operand.
  if (auto *X = isExtractFromCmpXchg(SI.getTrueValue(), 0))
    if (X == CmpXchg && X->getCompareOperand() == SI.getFalseValue())
      return SI.getFalseValue();

  // Check the false value case: The false value of the select is the returned
  // value of the same cmpxchg used by the condition, and the true value is the
  // cmpxchg instruction's compare operand.
  if (auto *X = isExtractFromCmpXchg(SI.getFalseValue(), 0))
    if (X == CmpXchg && X->getCompareOperand() == SI.getTrueValue())
      return SI.getFalseValue();

  return nullptr;
}

static Instruction *moveAddAfterMinMax(SelectPatternFlavor SPF, Value *X,
                                       Value *Y,
                                       InstCombiner::BuilderTy &Builder) {
  assert(SelectPatternResult::isMinOrMax(SPF) && "Expected min/max pattern");
  bool IsUnsigned = SPF == SelectPatternFlavor::SPF_UMIN ||
                    SPF == SelectPatternFlavor::SPF_UMAX;
  // TODO: If InstSimplify could fold all cases where C2 <= C1, we could change
  // the constant value check to an assert.
  Value *A;
  const APInt *C1, *C2;
  if (IsUnsigned && match(X, m_NUWAdd(m_Value(A), m_APInt(C1))) &&
      match(Y, m_APInt(C2)) && C2->uge(*C1) && X->hasNUses(2)) {
    // umin (add nuw A, C1), C2 --> add nuw (umin A, C2 - C1), C1
    // umax (add nuw A, C1), C2 --> add nuw (umax A, C2 - C1), C1
    Value *NewMinMax = createMinMax(Builder, SPF, A,
                                    ConstantInt::get(X->getType(), *C2 - *C1));
    return BinaryOperator::CreateNUW(BinaryOperator::Add, NewMinMax,
                                     ConstantInt::get(X->getType(), *C1));
  }

  if (!IsUnsigned && match(X, m_NSWAdd(m_Value(A), m_APInt(C1))) &&
      match(Y, m_APInt(C2)) && X->hasNUses(2)) {
    bool Overflow;
    APInt Diff = C2->ssub_ov(*C1, Overflow);
    if (!Overflow) {
      // smin (add nsw A, C1), C2 --> add nsw (smin A, C2 - C1), C1
      // smax (add nsw A, C1), C2 --> add nsw (smax A, C2 - C1), C1
      Value *NewMinMax = createMinMax(Builder, SPF, A,
                                      ConstantInt::get(X->getType(), Diff));
      return BinaryOperator::CreateNSW(BinaryOperator::Add, NewMinMax,
                                       ConstantInt::get(X->getType(), *C1));
    }
  }

  return nullptr;
}

/// Match a sadd_sat or ssub_sat which is using min/max to clamp the value.
Instruction *InstCombinerImpl::matchSAddSubSat(SelectInst &MinMax1) {
  Type *Ty = MinMax1.getType();

  // We are looking for a tree of:
  // max(INT_MIN, min(INT_MAX, add(sext(A), sext(B))))
  // Where the min and max could be reversed
  Instruction *MinMax2;
  BinaryOperator *AddSub;
  const APInt *MinValue, *MaxValue;
  if (match(&MinMax1, m_SMin(m_Instruction(MinMax2), m_APInt(MaxValue)))) {
    if (!match(MinMax2, m_SMax(m_BinOp(AddSub), m_APInt(MinValue))))
      return nullptr;
  } else if (match(&MinMax1,
                   m_SMax(m_Instruction(MinMax2), m_APInt(MinValue)))) {
    if (!match(MinMax2, m_SMin(m_BinOp(AddSub), m_APInt(MaxValue))))
      return nullptr;
  } else
    return nullptr;

  // Check that the constants clamp a saturate, and that the new type would be
  // sensible to convert to.
  if (!(*MaxValue + 1).isPowerOf2() || -*MinValue != *MaxValue + 1)
    return nullptr;
  // In what bitwidth can this be treated as saturating arithmetics?
  unsigned NewBitWidth = (*MaxValue + 1).logBase2() + 1;
  // FIXME: This isn't quite right for vectors, but using the scalar type is a
  // good first approximation for what should be done there.
  if (!shouldChangeType(Ty->getScalarType()->getIntegerBitWidth(), NewBitWidth))
    return nullptr;

  // Also make sure that the number of uses is as expected. The "3"s are for the
  // the two items of min/max (the compare and the select).
  if (MinMax2->hasNUsesOrMore(3) || AddSub->hasNUsesOrMore(3))
    return nullptr;

  // Create the new type (which can be a vector type)
  Type *NewTy = Ty->getWithNewBitWidth(NewBitWidth);
  // Match the two extends from the add/sub
  Value *A, *B;
  if(!match(AddSub, m_BinOp(m_SExt(m_Value(A)), m_SExt(m_Value(B)))))
    return nullptr;
  // And check the incoming values are of a type smaller than or equal to the
  // size of the saturation. Otherwise the higher bits can cause different
  // results.
  if (A->getType()->getScalarSizeInBits() > NewBitWidth ||
      B->getType()->getScalarSizeInBits() > NewBitWidth)
    return nullptr;

  Intrinsic::ID IntrinsicID;
  if (AddSub->getOpcode() == Instruction::Add)
    IntrinsicID = Intrinsic::sadd_sat;
  else if (AddSub->getOpcode() == Instruction::Sub)
    IntrinsicID = Intrinsic::ssub_sat;
  else
    return nullptr;

  // Finally create and return the sat intrinsic, truncated to the new type
  Function *F = Intrinsic::getDeclaration(MinMax1.getModule(), IntrinsicID, NewTy);
  Value *AT = Builder.CreateSExt(A, NewTy);
  Value *BT = Builder.CreateSExt(B, NewTy);
  Value *Sat = Builder.CreateCall(F, {AT, BT});
  return CastInst::Create(Instruction::SExt, Sat, Ty);
}

/// Reduce a sequence of min/max with a common operand.
static Instruction *factorizeMinMaxTree(SelectPatternFlavor SPF, Value *LHS,
                                        Value *RHS,
                                        InstCombiner::BuilderTy &Builder) {
  assert(SelectPatternResult::isMinOrMax(SPF) && "Expected a min/max");
  // TODO: Allow FP min/max with nnan/nsz.
  if (!LHS->getType()->isIntOrIntVectorTy())
    return nullptr;

  // Match 3 of the same min/max ops. Example: umin(umin(), umin()).
  Value *A, *B, *C, *D;
  SelectPatternResult L = matchSelectPattern(LHS, A, B);
  SelectPatternResult R = matchSelectPattern(RHS, C, D);
  if (SPF != L.Flavor || L.Flavor != R.Flavor)
    return nullptr;

  // Look for a common operand. The use checks are different than usual because
  // a min/max pattern typically has 2 uses of each op: 1 by the cmp and 1 by
  // the select.
  Value *MinMaxOp = nullptr;
  Value *ThirdOp = nullptr;
  if (!LHS->hasNUsesOrMore(3) && RHS->hasNUsesOrMore(3)) {
    // If the LHS is only used in this chain and the RHS is used outside of it,
    // reuse the RHS min/max because that will eliminate the LHS.
    if (D == A || C == A) {
      // min(min(a, b), min(c, a)) --> min(min(c, a), b)
      // min(min(a, b), min(a, d)) --> min(min(a, d), b)
      MinMaxOp = RHS;
      ThirdOp = B;
    } else if (D == B || C == B) {
      // min(min(a, b), min(c, b)) --> min(min(c, b), a)
      // min(min(a, b), min(b, d)) --> min(min(b, d), a)
      MinMaxOp = RHS;
      ThirdOp = A;
    }
  } else if (!RHS->hasNUsesOrMore(3)) {
    // Reuse the LHS. This will eliminate the RHS.
    if (D == A || D == B) {
      // min(min(a, b), min(c, a)) --> min(min(a, b), c)
      // min(min(a, b), min(c, b)) --> min(min(a, b), c)
      MinMaxOp = LHS;
      ThirdOp = C;
    } else if (C == A || C == B) {
      // min(min(a, b), min(b, d)) --> min(min(a, b), d)
      // min(min(a, b), min(c, b)) --> min(min(a, b), d)
      MinMaxOp = LHS;
      ThirdOp = D;
    }
  }
  if (!MinMaxOp || !ThirdOp)
    return nullptr;

  CmpInst::Predicate P = getMinMaxPred(SPF);
  Value *CmpABC = Builder.CreateICmp(P, MinMaxOp, ThirdOp);
  return SelectInst::Create(CmpABC, MinMaxOp, ThirdOp);
}

/// Try to reduce a funnel/rotate pattern that includes a compare and select
/// into a funnel shift intrinsic. Example:
/// rotl32(a, b) --> (b == 0 ? a : ((a >> (32 - b)) | (a << b)))
///              --> call llvm.fshl.i32(a, a, b)
/// fshl32(a, b, c) --> (c == 0 ? a : ((b >> (32 - c)) | (a << c)))
///                 --> call llvm.fshl.i32(a, b, c)
/// fshr32(a, b, c) --> (c == 0 ? b : ((a >> (32 - c)) | (b << c)))
///                 --> call llvm.fshr.i32(a, b, c)
static Instruction *foldSelectFunnelShift(SelectInst &Sel,
                                          InstCombiner::BuilderTy &Builder) {
  // This must be a power-of-2 type for a bitmasking transform to be valid.
  unsigned Width = Sel.getType()->getScalarSizeInBits();
  if (!isPowerOf2_32(Width))
    return nullptr;

  BinaryOperator *Or0, *Or1;
  if (!match(Sel.getFalseValue(), m_OneUse(m_Or(m_BinOp(Or0), m_BinOp(Or1)))))
    return nullptr;

  Value *SV0, *SV1, *SA0, *SA1;
  if (!match(Or0, m_OneUse(m_LogicalShift(m_Value(SV0),
                                          m_ZExtOrSelf(m_Value(SA0))))) ||
      !match(Or1, m_OneUse(m_LogicalShift(m_Value(SV1),
                                          m_ZExtOrSelf(m_Value(SA1))))) ||
      Or0->getOpcode() == Or1->getOpcode())
    return nullptr;

  // Canonicalize to or(shl(SV0, SA0), lshr(SV1, SA1)).
  if (Or0->getOpcode() == BinaryOperator::LShr) {
    std::swap(Or0, Or1);
    std::swap(SV0, SV1);
    std::swap(SA0, SA1);
  }
  assert(Or0->getOpcode() == BinaryOperator::Shl &&
         Or1->getOpcode() == BinaryOperator::LShr &&
         "Illegal or(shift,shift) pair");

  // Check the shift amounts to see if they are an opposite pair.
  Value *ShAmt;
  if (match(SA1, m_OneUse(m_Sub(m_SpecificInt(Width), m_Specific(SA0)))))
    ShAmt = SA0;
  else if (match(SA0, m_OneUse(m_Sub(m_SpecificInt(Width), m_Specific(SA1)))))
    ShAmt = SA1;
  else
    return nullptr;

  // We should now have this pattern:
  // select ?, TVal, (or (shl SV0, SA0), (lshr SV1, SA1))
  // The false value of the select must be a funnel-shift of the true value:
  // IsFShl -> TVal must be SV0 else TVal must be SV1.
  bool IsFshl = (ShAmt == SA0);
  Value *TVal = Sel.getTrueValue();
  if ((IsFshl && TVal != SV0) || (!IsFshl && TVal != SV1))
    return nullptr;

  // Finally, see if the select is filtering out a shift-by-zero.
  Value *Cond = Sel.getCondition();
  ICmpInst::Predicate Pred;
  if (!match(Cond, m_OneUse(m_ICmp(Pred, m_Specific(ShAmt), m_ZeroInt()))) ||
      Pred != ICmpInst::ICMP_EQ)
    return nullptr;

  // If this is not a rotate then the select was blocking poison from the
  // 'shift-by-zero' non-TVal, but a funnel shift won't - so freeze it.
  if (SV0 != SV1) {
    if (IsFshl && !llvm::isGuaranteedNotToBePoison(SV1))
      SV1 = Builder.CreateFreeze(SV1);
    else if (!IsFshl && !llvm::isGuaranteedNotToBePoison(SV0))
      SV0 = Builder.CreateFreeze(SV0);
  }

  // This is a funnel/rotate that avoids shift-by-bitwidth UB in a suboptimal way.
  // Convert to funnel shift intrinsic.
  Intrinsic::ID IID = IsFshl ? Intrinsic::fshl : Intrinsic::fshr;
  Function *F = Intrinsic::getDeclaration(Sel.getModule(), IID, Sel.getType());
  ShAmt = Builder.CreateZExt(ShAmt, Sel.getType());
  return CallInst::Create(F, { SV0, SV1, ShAmt });
}

static Instruction *foldSelectToCopysign(SelectInst &Sel,
                                         InstCombiner::BuilderTy &Builder) {
  Value *Cond = Sel.getCondition();
  Value *TVal = Sel.getTrueValue();
  Value *FVal = Sel.getFalseValue();
  Type *SelType = Sel.getType();

  // Match select ?, TC, FC where the constants are equal but negated.
  // TODO: Generalize to handle a negated variable operand?
  const APFloat *TC, *FC;
  if (!match(TVal, m_APFloat(TC)) || !match(FVal, m_APFloat(FC)) ||
      !abs(*TC).bitwiseIsEqual(abs(*FC)))
    return nullptr;

  assert(TC != FC && "Expected equal select arms to simplify");

  Value *X;
  const APInt *C;
  bool IsTrueIfSignSet;
  ICmpInst::Predicate Pred;
  if (!match(Cond, m_OneUse(m_ICmp(Pred, m_BitCast(m_Value(X)), m_APInt(C)))) ||
      !InstCombiner::isSignBitCheck(Pred, *C, IsTrueIfSignSet) ||
      X->getType() != SelType)
    return nullptr;

  // If needed, negate the value that will be the sign argument of the copysign:
  // (bitcast X) <  0 ? -TC :  TC --> copysign(TC,  X)
  // (bitcast X) <  0 ?  TC : -TC --> copysign(TC, -X)
  // (bitcast X) >= 0 ? -TC :  TC --> copysign(TC, -X)
  // (bitcast X) >= 0 ?  TC : -TC --> copysign(TC,  X)
  if (IsTrueIfSignSet ^ TC->isNegative())
    X = Builder.CreateFNegFMF(X, &Sel);

  // Canonicalize the magnitude argument as the positive constant since we do
  // not care about its sign.
  Value *MagArg = TC->isNegative() ? FVal : TVal;
  Function *F = Intrinsic::getDeclaration(Sel.getModule(), Intrinsic::copysign,
                                          Sel.getType());
  Instruction *CopySign = CallInst::Create(F, { MagArg, X });
  CopySign->setFastMathFlags(Sel.getFastMathFlags());
  return CopySign;
}

Instruction *InstCombinerImpl::foldVectorSelect(SelectInst &Sel) {
  auto *VecTy = dyn_cast<FixedVectorType>(Sel.getType());
  if (!VecTy)
    return nullptr;

  unsigned NumElts = VecTy->getNumElements();
  APInt UndefElts(NumElts, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(NumElts));
  if (Value *V = SimplifyDemandedVectorElts(&Sel, AllOnesEltMask, UndefElts)) {
    if (V != &Sel)
      return replaceInstUsesWith(Sel, V);
    return &Sel;
  }

  // A select of a "select shuffle" with a common operand can be rearranged
  // to select followed by "select shuffle". Because of poison, this only works
  // in the case of a shuffle with no undefined mask elements.
  Value *Cond = Sel.getCondition();
  Value *TVal = Sel.getTrueValue();
  Value *FVal = Sel.getFalseValue();
  Value *X, *Y;
  ArrayRef<int> Mask;
  if (match(TVal, m_OneUse(m_Shuffle(m_Value(X), m_Value(Y), m_Mask(Mask)))) &&
      !is_contained(Mask, UndefMaskElem) &&
      cast<ShuffleVectorInst>(TVal)->isSelect()) {
    if (X == FVal) {
      // select Cond, (shuf_sel X, Y), X --> shuf_sel X, (select Cond, Y, X)
      Value *NewSel = Builder.CreateSelect(Cond, Y, X, "sel", &Sel);
      return new ShuffleVectorInst(X, NewSel, Mask);
    }
    if (Y == FVal) {
      // select Cond, (shuf_sel X, Y), Y --> shuf_sel (select Cond, X, Y), Y
      Value *NewSel = Builder.CreateSelect(Cond, X, Y, "sel", &Sel);
      return new ShuffleVectorInst(NewSel, Y, Mask);
    }
  }
  if (match(FVal, m_OneUse(m_Shuffle(m_Value(X), m_Value(Y), m_Mask(Mask)))) &&
      !is_contained(Mask, UndefMaskElem) &&
      cast<ShuffleVectorInst>(FVal)->isSelect()) {
    if (X == TVal) {
      // select Cond, X, (shuf_sel X, Y) --> shuf_sel X, (select Cond, X, Y)
      Value *NewSel = Builder.CreateSelect(Cond, X, Y, "sel", &Sel);
      return new ShuffleVectorInst(X, NewSel, Mask);
    }
    if (Y == TVal) {
      // select Cond, Y, (shuf_sel X, Y) --> shuf_sel (select Cond, Y, X), Y
      Value *NewSel = Builder.CreateSelect(Cond, Y, X, "sel", &Sel);
      return new ShuffleVectorInst(NewSel, Y, Mask);
    }
  }

  return nullptr;
}

static Instruction *foldSelectToPhiImpl(SelectInst &Sel, BasicBlock *BB,
                                        const DominatorTree &DT,
                                        InstCombiner::BuilderTy &Builder) {
  // Find the block's immediate dominator that ends with a conditional branch
  // that matches select's condition (maybe inverted).
  auto *IDomNode = DT[BB]->getIDom();
  if (!IDomNode)
    return nullptr;
  BasicBlock *IDom = IDomNode->getBlock();

  Value *Cond = Sel.getCondition();
  Value *IfTrue, *IfFalse;
  BasicBlock *TrueSucc, *FalseSucc;
  if (match(IDom->getTerminator(),
            m_Br(m_Specific(Cond), m_BasicBlock(TrueSucc),
                 m_BasicBlock(FalseSucc)))) {
    IfTrue = Sel.getTrueValue();
    IfFalse = Sel.getFalseValue();
  } else if (match(IDom->getTerminator(),
                   m_Br(m_Not(m_Specific(Cond)), m_BasicBlock(TrueSucc),
                        m_BasicBlock(FalseSucc)))) {
    IfTrue = Sel.getFalseValue();
    IfFalse = Sel.getTrueValue();
  } else
    return nullptr;

  // Make sure the branches are actually different.
  if (TrueSucc == FalseSucc)
    return nullptr;

  // We want to replace select %cond, %a, %b with a phi that takes value %a
  // for all incoming edges that are dominated by condition `%cond == true`,
  // and value %b for edges dominated by condition `%cond == false`. If %a
  // or %b are also phis from the same basic block, we can go further and take
  // their incoming values from the corresponding blocks.
  BasicBlockEdge TrueEdge(IDom, TrueSucc);
  BasicBlockEdge FalseEdge(IDom, FalseSucc);
  DenseMap<BasicBlock *, Value *> Inputs;
  for (auto *Pred : predecessors(BB)) {
    // Check implication.
    BasicBlockEdge Incoming(Pred, BB);
    if (DT.dominates(TrueEdge, Incoming))
      Inputs[Pred] = IfTrue->DoPHITranslation(BB, Pred);
    else if (DT.dominates(FalseEdge, Incoming))
      Inputs[Pred] = IfFalse->DoPHITranslation(BB, Pred);
    else
      return nullptr;
    // Check availability.
    if (auto *Insn = dyn_cast<Instruction>(Inputs[Pred]))
      if (!DT.dominates(Insn, Pred->getTerminator()))
        return nullptr;
  }

  Builder.SetInsertPoint(&*BB->begin());
  auto *PN = Builder.CreatePHI(Sel.getType(), Inputs.size());
  for (auto *Pred : predecessors(BB))
    PN->addIncoming(Inputs[Pred], Pred);
  PN->takeName(&Sel);
  return PN;
}

static Instruction *foldSelectToPhi(SelectInst &Sel, const DominatorTree &DT,
                                    InstCombiner::BuilderTy &Builder) {
  // Try to replace this select with Phi in one of these blocks.
  SmallSetVector<BasicBlock *, 4> CandidateBlocks;
  CandidateBlocks.insert(Sel.getParent());
  for (Value *V : Sel.operands())
    if (auto *I = dyn_cast<Instruction>(V))
      CandidateBlocks.insert(I->getParent());

  for (BasicBlock *BB : CandidateBlocks)
    if (auto *PN = foldSelectToPhiImpl(Sel, BB, DT, Builder))
      return PN;
  return nullptr;
}

static Value *foldSelectWithFrozenICmp(SelectInst &Sel, InstCombiner::BuilderTy &Builder) {
  FreezeInst *FI = dyn_cast<FreezeInst>(Sel.getCondition());
  if (!FI)
    return nullptr;

  Value *Cond = FI->getOperand(0);
  Value *TrueVal = Sel.getTrueValue(), *FalseVal = Sel.getFalseValue();

  //   select (freeze(x == y)), x, y --> y
  //   select (freeze(x != y)), x, y --> x
  // The freeze should be only used by this select. Otherwise, remaining uses of
  // the freeze can observe a contradictory value.
  //   c = freeze(x == y)   ; Let's assume that y = poison & x = 42; c is 0 or 1
  //   a = select c, x, y   ;
  //   f(a, c)              ; f(poison, 1) cannot happen, but if a is folded
  //                        ; to y, this can happen.
  CmpInst::Predicate Pred;
  if (FI->hasOneUse() &&
      match(Cond, m_c_ICmp(Pred, m_Specific(TrueVal), m_Specific(FalseVal))) &&
      (Pred == ICmpInst::ICMP_EQ || Pred == ICmpInst::ICMP_NE)) {
    return Pred == ICmpInst::ICMP_EQ ? FalseVal : TrueVal;
  }

  return nullptr;
}

Instruction *InstCombinerImpl::foldAndOrOfSelectUsingImpliedCond(Value *Op,
                                                                 SelectInst &SI,
                                                                 bool IsAnd) {
  Value *CondVal = SI.getCondition();
  Value *A = SI.getTrueValue();
  Value *B = SI.getFalseValue();

  assert(Op->getType()->isIntOrIntVectorTy(1) &&
         "Op must be either i1 or vector of i1.");

  Optional<bool> Res = isImpliedCondition(Op, CondVal, DL, IsAnd);
  if (!Res)
    return nullptr;

  Value *Zero = Constant::getNullValue(A->getType());
  Value *One = Constant::getAllOnesValue(A->getType());

  if (*Res == true) {
    if (IsAnd)
      // select op, (select cond, A, B), false => select op, A, false
      // and    op, (select cond, A, B)        => select op, A, false
      //   if op = true implies condval = true.
      return SelectInst::Create(Op, A, Zero);
    else
      // select op, true, (select cond, A, B) => select op, true, A
      // or     op, (select cond, A, B)       => select op, true, A
      //   if op = false implies condval = true.
      return SelectInst::Create(Op, One, A);
  } else {
    if (IsAnd)
      // select op, (select cond, A, B), false => select op, B, false
      // and    op, (select cond, A, B)        => select op, B, false
      //   if op = true implies condval = false.
      return SelectInst::Create(Op, B, Zero);
    else
      // select op, true, (select cond, A, B) => select op, true, B
      // or     op, (select cond, A, B)       => select op, true, B
      //   if op = false implies condval = false.
      return SelectInst::Create(Op, One, B);
  }
}

Instruction *InstCombinerImpl::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  Type *SelType = SI.getType();

  // FIXME: Remove this workaround when freeze related patches are done.
  // For select with undef operand which feeds into an equality comparison,
  // don't simplify it so loop unswitch can know the equality comparison
  // may have an undef operand. This is a workaround for PR31652 caused by
  // descrepancy about branch on undef between LoopUnswitch and GVN.
  if (match(TrueVal, m_Undef()) || match(FalseVal, m_Undef())) {
    if (llvm::any_of(SI.users(), [&](User *U) {
          ICmpInst *CI = dyn_cast<ICmpInst>(U);
          if (CI && CI->isEquality())
            return true;
          return false;
        })) {
      return nullptr;
    }
  }

  if (Value *V = SimplifySelectInst(CondVal, TrueVal, FalseVal,
                                    SQ.getWithInstruction(&SI)))
    return replaceInstUsesWith(SI, V);

  if (Instruction *I = canonicalizeSelectToShuffle(SI))
    return I;

  if (Instruction *I = canonicalizeScalarSelectOfVecs(SI, *this))
    return I;

  CmpInst::Predicate Pred;

  // Avoid potential infinite loops by checking for non-constant condition.
  // TODO: Can we assert instead by improving canonicalizeSelectToShuffle()?
  //       Scalar select must have simplified?
  if (SelType->isIntOrIntVectorTy(1) && !isa<Constant>(CondVal) &&
      TrueVal->getType() == CondVal->getType()) {
    // Folding select to and/or i1 isn't poison safe in general. impliesPoison
    // checks whether folding it does not convert a well-defined value into
    // poison.
    if (match(TrueVal, m_One()) && impliesPoison(FalseVal, CondVal)) {
      // Change: A = select B, true, C --> A = or B, C
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    }
    if (match(FalseVal, m_Zero()) && impliesPoison(TrueVal, CondVal)) {
      // Change: A = select B, C, false --> A = and B, C
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
    }

    auto *One = ConstantInt::getTrue(SelType);
    auto *Zero = ConstantInt::getFalse(SelType);

    // We match the "full" 0 or 1 constant here to avoid a potential infinite
    // loop with vectors that may have undefined/poison elements.
    // select a, false, b -> select !a, b, false
    if (match(TrueVal, m_Specific(Zero))) {
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
      return SelectInst::Create(NotCond, FalseVal, Zero);
    }
    // select a, b, true -> select !a, true, b
    if (match(FalseVal, m_Specific(One))) {
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
      return SelectInst::Create(NotCond, One, TrueVal);
    }

    // select a, a, b -> select a, true, b
    if (CondVal == TrueVal)
      return replaceOperand(SI, 1, One);
    // select a, b, a -> select a, b, false
    if (CondVal == FalseVal)
      return replaceOperand(SI, 2, Zero);

    // select a, !a, b -> select !a, b, false
    if (match(TrueVal, m_Not(m_Specific(CondVal))))
      return SelectInst::Create(TrueVal, FalseVal, Zero);
    // select a, b, !a -> select !a, true, b
    if (match(FalseVal, m_Not(m_Specific(CondVal))))
      return SelectInst::Create(FalseVal, One, TrueVal);

    Value *A, *B;

    // DeMorgan in select form: !a && !b --> !(a || b)
    // select !a, !b, false --> not (select a, true, b)
    if (match(&SI, m_LogicalAnd(m_Not(m_Value(A)), m_Not(m_Value(B)))) &&
        (CondVal->hasOneUse() || TrueVal->hasOneUse()) &&
        !match(A, m_ConstantExpr()) && !match(B, m_ConstantExpr()))
      return BinaryOperator::CreateNot(Builder.CreateSelect(A, One, B));

    // DeMorgan in select form: !a || !b --> !(a && b)
    // select !a, true, !b --> not (select a, b, false)
    if (match(&SI, m_LogicalOr(m_Not(m_Value(A)), m_Not(m_Value(B)))) &&
        (CondVal->hasOneUse() || FalseVal->hasOneUse()) &&
        !match(A, m_ConstantExpr()) && !match(B, m_ConstantExpr()))
      return BinaryOperator::CreateNot(Builder.CreateSelect(A, B, Zero));

    // select (select a, true, b), true, b -> select a, true, b
    if (match(CondVal, m_Select(m_Value(A), m_One(), m_Value(B))) &&
        match(TrueVal, m_One()) && match(FalseVal, m_Specific(B)))
      return replaceOperand(SI, 0, A);
    // select (select a, b, false), b, false -> select a, b, false
    if (match(CondVal, m_Select(m_Value(A), m_Value(B), m_Zero())) &&
        match(TrueVal, m_Specific(B)) && match(FalseVal, m_Zero()))
      return replaceOperand(SI, 0, A);

    if (!SelType->isVectorTy()) {
      if (Value *S = simplifyWithOpReplaced(TrueVal, CondVal, One, SQ,
                                            /* AllowRefinement */ true))
        return replaceOperand(SI, 1, S);
      if (Value *S = simplifyWithOpReplaced(FalseVal, CondVal, Zero, SQ,
                                            /* AllowRefinement */ true))
        return replaceOperand(SI, 2, S);
    }

    if (match(FalseVal, m_Zero()) || match(TrueVal, m_One())) {
      Use *Y = nullptr;
      bool IsAnd = match(FalseVal, m_Zero()) ? true : false;
      Value *Op1 = IsAnd ? TrueVal : FalseVal;
      if (isCheckForZeroAndMulWithOverflow(CondVal, Op1, IsAnd, Y)) {
        auto *FI = new FreezeInst(*Y, (*Y)->getName() + ".fr");
        InsertNewInstBefore(FI, *cast<Instruction>(Y->getUser()));
        replaceUse(*Y, FI);
        return replaceInstUsesWith(SI, Op1);
      }

      if (auto *Op1SI = dyn_cast<SelectInst>(Op1))
        if (auto *I = foldAndOrOfSelectUsingImpliedCond(CondVal, *Op1SI,
                                                        /* IsAnd */ IsAnd))
          return I;

      if (auto *ICmp0 = dyn_cast<ICmpInst>(CondVal))
        if (auto *ICmp1 = dyn_cast<ICmpInst>(Op1))
          if (auto *V = foldAndOrOfICmpsOfAndWithPow2(ICmp0, ICmp1, &SI, IsAnd,
                                                      /* IsLogical */ true))
            return replaceInstUsesWith(SI, V);
    }

    // select (select a, true, b), c, false -> select a, c, false
    // select c, (select a, true, b), false -> select c, a, false
    //   if c implies that b is false.
    if (match(CondVal, m_Select(m_Value(A), m_One(), m_Value(B))) &&
        match(FalseVal, m_Zero())) {
      Optional<bool> Res = isImpliedCondition(TrueVal, B, DL);
      if (Res && *Res == false)
        return replaceOperand(SI, 0, A);
    }
    if (match(TrueVal, m_Select(m_Value(A), m_One(), m_Value(B))) &&
        match(FalseVal, m_Zero())) {
      Optional<bool> Res = isImpliedCondition(CondVal, B, DL);
      if (Res && *Res == false)
        return replaceOperand(SI, 1, A);
    }
    // select c, true, (select a, b, false)  -> select c, true, a
    // select (select a, b, false), true, c  -> select a, true, c
    //   if c = false implies that b = true
    if (match(TrueVal, m_One()) &&
        match(FalseVal, m_Select(m_Value(A), m_Value(B), m_Zero()))) {
      Optional<bool> Res = isImpliedCondition(CondVal, B, DL, false);
      if (Res && *Res == true)
        return replaceOperand(SI, 2, A);
    }
    if (match(CondVal, m_Select(m_Value(A), m_Value(B), m_Zero())) &&
        match(TrueVal, m_One())) {
      Optional<bool> Res = isImpliedCondition(FalseVal, B, DL, false);
      if (Res && *Res == true)
        return replaceOperand(SI, 0, A);
    }

    // sel (sel c, a, false), true, (sel !c, b, false) -> sel c, a, b
    // sel (sel !c, a, false), true, (sel c, b, false) -> sel c, b, a
    Value *C1, *C2;
    if (match(CondVal, m_Select(m_Value(C1), m_Value(A), m_Zero())) &&
        match(TrueVal, m_One()) &&
        match(FalseVal, m_Select(m_Value(C2), m_Value(B), m_Zero()))) {
      if (match(C2, m_Not(m_Specific(C1)))) // first case
        return SelectInst::Create(C1, A, B);
      else if (match(C1, m_Not(m_Specific(C2)))) // second case
        return SelectInst::Create(C2, B, A);
    }
  }

  // Selecting between two integer or vector splat integer constants?
  //
  // Note that we don't handle a scalar select of vectors:
  // select i1 %c, <2 x i8> <1, 1>, <2 x i8> <0, 0>
  // because that may need 3 instructions to splat the condition value:
  // extend, insertelement, shufflevector.
  //
  // Do not handle i1 TrueVal and FalseVal otherwise would result in
  // zext/sext i1 to i1.
  if (SelType->isIntOrIntVectorTy() && !SelType->isIntOrIntVectorTy(1) &&
      CondVal->getType()->isVectorTy() == SelType->isVectorTy()) {
    // select C, 1, 0 -> zext C to int
    if (match(TrueVal, m_One()) && match(FalseVal, m_Zero()))
      return new ZExtInst(CondVal, SelType);

    // select C, -1, 0 -> sext C to int
    if (match(TrueVal, m_AllOnes()) && match(FalseVal, m_Zero()))
      return new SExtInst(CondVal, SelType);

    // select C, 0, 1 -> zext !C to int
    if (match(TrueVal, m_Zero()) && match(FalseVal, m_One())) {
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
      return new ZExtInst(NotCond, SelType);
    }

    // select C, 0, -1 -> sext !C to int
    if (match(TrueVal, m_Zero()) && match(FalseVal, m_AllOnes())) {
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
      return new SExtInst(NotCond, SelType);
    }
  }

  if (auto *FCmp = dyn_cast<FCmpInst>(CondVal)) {
    Value *Cmp0 = FCmp->getOperand(0), *Cmp1 = FCmp->getOperand(1);
    // Are we selecting a value based on a comparison of the two values?
    if ((Cmp0 == TrueVal && Cmp1 == FalseVal) ||
        (Cmp0 == FalseVal && Cmp1 == TrueVal)) {
      // Canonicalize to use ordered comparisons by swapping the select
      // operands.
      //
      // e.g.
      // (X ugt Y) ? X : Y -> (X ole Y) ? Y : X
      if (FCmp->hasOneUse() && FCmpInst::isUnordered(FCmp->getPredicate())) {
        FCmpInst::Predicate InvPred = FCmp->getInversePredicate();
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        // FIXME: The FMF should propagate from the select, not the fcmp.
        Builder.setFastMathFlags(FCmp->getFastMathFlags());
        Value *NewCond = Builder.CreateFCmp(InvPred, Cmp0, Cmp1,
                                            FCmp->getName() + ".inv");
        Value *NewSel = Builder.CreateSelect(NewCond, FalseVal, TrueVal);
        return replaceInstUsesWith(SI, NewSel);
      }

      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
  }

  // Canonicalize select with fcmp to fabs(). -0.0 makes this tricky. We need
  // fast-math-flags (nsz) or fsub with +0.0 (not fneg) for this to work. We
  // also require nnan because we do not want to unintentionally change the
  // sign of a NaN value.
  // FIXME: These folds should test/propagate FMF from the select, not the
  //        fsub or fneg.
  // (X <= +/-0.0) ? (0.0 - X) : X --> fabs(X)
  Instruction *FSub;
  if (match(CondVal, m_FCmp(Pred, m_Specific(FalseVal), m_AnyZeroFP())) &&
      match(TrueVal, m_FSub(m_PosZeroFP(), m_Specific(FalseVal))) &&
      match(TrueVal, m_Instruction(FSub)) && FSub->hasNoNaNs() &&
      (Pred == FCmpInst::FCMP_OLE || Pred == FCmpInst::FCMP_ULE)) {
    Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FalseVal, FSub);
    return replaceInstUsesWith(SI, Fabs);
  }
  // (X >  +/-0.0) ? X : (0.0 - X) --> fabs(X)
  if (match(CondVal, m_FCmp(Pred, m_Specific(TrueVal), m_AnyZeroFP())) &&
      match(FalseVal, m_FSub(m_PosZeroFP(), m_Specific(TrueVal))) &&
      match(FalseVal, m_Instruction(FSub)) && FSub->hasNoNaNs() &&
      (Pred == FCmpInst::FCMP_OGT || Pred == FCmpInst::FCMP_UGT)) {
    Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, TrueVal, FSub);
    return replaceInstUsesWith(SI, Fabs);
  }
  // With nnan and nsz:
  // (X <  +/-0.0) ? -X : X --> fabs(X)
  // (X <= +/-0.0) ? -X : X --> fabs(X)
  Instruction *FNeg;
  if (match(CondVal, m_FCmp(Pred, m_Specific(FalseVal), m_AnyZeroFP())) &&
      match(TrueVal, m_FNeg(m_Specific(FalseVal))) &&
      match(TrueVal, m_Instruction(FNeg)) &&
      FNeg->hasNoNaNs() && FNeg->hasNoSignedZeros() &&
      (Pred == FCmpInst::FCMP_OLT || Pred == FCmpInst::FCMP_OLE ||
       Pred == FCmpInst::FCMP_ULT || Pred == FCmpInst::FCMP_ULE)) {
    Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FalseVal, FNeg);
    return replaceInstUsesWith(SI, Fabs);
  }
  // With nnan and nsz:
  // (X >  +/-0.0) ? X : -X --> fabs(X)
  // (X >= +/-0.0) ? X : -X --> fabs(X)
  if (match(CondVal, m_FCmp(Pred, m_Specific(TrueVal), m_AnyZeroFP())) &&
      match(FalseVal, m_FNeg(m_Specific(TrueVal))) &&
      match(FalseVal, m_Instruction(FNeg)) &&
      FNeg->hasNoNaNs() && FNeg->hasNoSignedZeros() &&
      (Pred == FCmpInst::FCMP_OGT || Pred == FCmpInst::FCMP_OGE ||
       Pred == FCmpInst::FCMP_UGT || Pred == FCmpInst::FCMP_UGE)) {
    Value *Fabs = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, TrueVal, FNeg);
    return replaceInstUsesWith(SI, Fabs);
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = foldSelectInstWithICmp(SI, ICI))
      return Result;

  if (Instruction *Add = foldAddSubSelect(SI, Builder))
    return Add;
  if (Instruction *Add = foldOverflowingAddSubSelect(SI, Builder))
    return Add;
  if (Instruction *Or = foldSetClearBits(SI, Builder))
    return Or;

  // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
  auto *TI = dyn_cast<Instruction>(TrueVal);
  auto *FI = dyn_cast<Instruction>(FalseVal);
  if (TI && FI && TI->getOpcode() == FI->getOpcode())
    if (Instruction *IV = foldSelectOpOp(SI, TI, FI))
      return IV;

  if (Instruction *I = foldSelectExtConst(SI))
    return I;

  // See if we can fold the select into one of our operands.
  if (SelType->isIntOrIntVectorTy() || SelType->isFPOrFPVectorTy()) {
    if (Instruction *FoldI = foldSelectIntoOp(SI, TrueVal, FalseVal))
      return FoldI;

    Value *LHS, *RHS;
    Instruction::CastOps CastOp;
    SelectPatternResult SPR = matchSelectPattern(&SI, LHS, RHS, &CastOp);
    auto SPF = SPR.Flavor;
    if (SPF) {
      Value *LHS2, *RHS2;
      if (SelectPatternFlavor SPF2 = matchSelectPattern(LHS, LHS2, RHS2).Flavor)
        if (Instruction *R = foldSPFofSPF(cast<Instruction>(LHS), SPF2, LHS2,
                                          RHS2, SI, SPF, RHS))
          return R;
      if (SelectPatternFlavor SPF2 = matchSelectPattern(RHS, LHS2, RHS2).Flavor)
        if (Instruction *R = foldSPFofSPF(cast<Instruction>(RHS), SPF2, LHS2,
                                          RHS2, SI, SPF, LHS))
          return R;
      // TODO.
      // ABS(-X) -> ABS(X)
    }

    if (SelectPatternResult::isMinOrMax(SPF)) {
      // Canonicalize so that
      // - type casts are outside select patterns.
      // - float clamp is transformed to min/max pattern

      bool IsCastNeeded = LHS->getType() != SelType;
      Value *CmpLHS = cast<CmpInst>(CondVal)->getOperand(0);
      Value *CmpRHS = cast<CmpInst>(CondVal)->getOperand(1);
      if (IsCastNeeded ||
          (LHS->getType()->isFPOrFPVectorTy() &&
           ((CmpLHS != LHS && CmpLHS != RHS) ||
            (CmpRHS != LHS && CmpRHS != RHS)))) {
        CmpInst::Predicate MinMaxPred = getMinMaxPred(SPF, SPR.Ordered);

        Value *Cmp;
        if (CmpInst::isIntPredicate(MinMaxPred)) {
          Cmp = Builder.CreateICmp(MinMaxPred, LHS, RHS);
        } else {
          IRBuilder<>::FastMathFlagGuard FMFG(Builder);
          auto FMF =
              cast<FPMathOperator>(SI.getCondition())->getFastMathFlags();
          Builder.setFastMathFlags(FMF);
          Cmp = Builder.CreateFCmp(MinMaxPred, LHS, RHS);
        }

        Value *NewSI = Builder.CreateSelect(Cmp, LHS, RHS, SI.getName(), &SI);
        if (!IsCastNeeded)
          return replaceInstUsesWith(SI, NewSI);

        Value *NewCast = Builder.CreateCast(CastOp, NewSI, SelType);
        return replaceInstUsesWith(SI, NewCast);
      }

      // MAX(~a, ~b) -> ~MIN(a, b)
      // MAX(~a, C)  -> ~MIN(a, ~C)
      // MIN(~a, ~b) -> ~MAX(a, b)
      // MIN(~a, C)  -> ~MAX(a, ~C)
      auto moveNotAfterMinMax = [&](Value *X, Value *Y) -> Instruction * {
        Value *A;
        if (match(X, m_Not(m_Value(A))) && !X->hasNUsesOrMore(3) &&
            !isFreeToInvert(A, A->hasOneUse()) &&
            // Passing false to only consider m_Not and constants.
            isFreeToInvert(Y, false)) {
          Value *B = Builder.CreateNot(Y);
          Value *NewMinMax = createMinMax(Builder, getInverseMinMaxFlavor(SPF),
                                          A, B);
          // Copy the profile metadata.
          if (MDNode *MD = SI.getMetadata(LLVMContext::MD_prof)) {
            cast<SelectInst>(NewMinMax)->setMetadata(LLVMContext::MD_prof, MD);
            // Swap the metadata if the operands are swapped.
            if (X == SI.getFalseValue() && Y == SI.getTrueValue())
              cast<SelectInst>(NewMinMax)->swapProfMetadata();
          }

          return BinaryOperator::CreateNot(NewMinMax);
        }

        return nullptr;
      };

      if (Instruction *I = moveNotAfterMinMax(LHS, RHS))
        return I;
      if (Instruction *I = moveNotAfterMinMax(RHS, LHS))
        return I;

      if (Instruction *I = moveAddAfterMinMax(SPF, LHS, RHS, Builder))
        return I;

      if (Instruction *I = factorizeMinMaxTree(SPF, LHS, RHS, Builder))
        return I;
      if (Instruction *I = matchSAddSubSat(SI))
        return I;
    }
  }

  // Canonicalize select of FP values where NaN and -0.0 are not valid as
  // minnum/maxnum intrinsics.
  if (isa<FPMathOperator>(SI) && SI.hasNoNaNs() && SI.hasNoSignedZeros()) {
    Value *X, *Y;
    if (match(&SI, m_OrdFMax(m_Value(X), m_Value(Y))))
      return replaceInstUsesWith(
          SI, Builder.CreateBinaryIntrinsic(Intrinsic::maxnum, X, Y, &SI));

    if (match(&SI, m_OrdFMin(m_Value(X), m_Value(Y))))
      return replaceInstUsesWith(
          SI, Builder.CreateBinaryIntrinsic(Intrinsic::minnum, X, Y, &SI));
  }

  // See if we can fold the select into a phi node if the condition is a select.
  if (auto *PN = dyn_cast<PHINode>(SI.getCondition()))
    // The true/false values have to be live in the PHI predecessor's blocks.
    if (canSelectOperandBeMappingIntoPredBlock(TrueVal, SI) &&
        canSelectOperandBeMappingIntoPredBlock(FalseVal, SI))
      if (Instruction *NV = foldOpIntoPhi(SI, PN))
        return NV;

  if (SelectInst *TrueSI = dyn_cast<SelectInst>(TrueVal)) {
    if (TrueSI->getCondition()->getType() == CondVal->getType()) {
      // select(C, select(C, a, b), c) -> select(C, a, c)
      if (TrueSI->getCondition() == CondVal) {
        if (SI.getTrueValue() == TrueSI->getTrueValue())
          return nullptr;
        return replaceOperand(SI, 1, TrueSI->getTrueValue());
      }
      // select(C0, select(C1, a, b), b) -> select(C0&C1, a, b)
      // We choose this as normal form to enable folding on the And and
      // shortening paths for the values (this helps getUnderlyingObjects() for
      // example).
      if (TrueSI->getFalseValue() == FalseVal && TrueSI->hasOneUse()) {
        Value *And = Builder.CreateLogicalAnd(CondVal, TrueSI->getCondition());
        replaceOperand(SI, 0, And);
        replaceOperand(SI, 1, TrueSI->getTrueValue());
        return &SI;
      }
    }
  }
  if (SelectInst *FalseSI = dyn_cast<SelectInst>(FalseVal)) {
    if (FalseSI->getCondition()->getType() == CondVal->getType()) {
      // select(C, a, select(C, b, c)) -> select(C, a, c)
      if (FalseSI->getCondition() == CondVal) {
        if (SI.getFalseValue() == FalseSI->getFalseValue())
          return nullptr;
        return replaceOperand(SI, 2, FalseSI->getFalseValue());
      }
      // select(C0, a, select(C1, a, b)) -> select(C0|C1, a, b)
      if (FalseSI->getTrueValue() == TrueVal && FalseSI->hasOneUse()) {
        Value *Or = Builder.CreateLogicalOr(CondVal, FalseSI->getCondition());
        replaceOperand(SI, 0, Or);
        replaceOperand(SI, 2, FalseSI->getFalseValue());
        return &SI;
      }
    }
  }

  auto canMergeSelectThroughBinop = [](BinaryOperator *BO) {
    // The select might be preventing a division by 0.
    switch (BO->getOpcode()) {
    default:
      return true;
    case Instruction::SRem:
    case Instruction::URem:
    case Instruction::SDiv:
    case Instruction::UDiv:
      return false;
    }
  };

  // Try to simplify a binop sandwiched between 2 selects with the same
  // condition.
  // select(C, binop(select(C, X, Y), W), Z) -> select(C, binop(X, W), Z)
  BinaryOperator *TrueBO;
  if (match(TrueVal, m_OneUse(m_BinOp(TrueBO))) &&
      canMergeSelectThroughBinop(TrueBO)) {
    if (auto *TrueBOSI = dyn_cast<SelectInst>(TrueBO->getOperand(0))) {
      if (TrueBOSI->getCondition() == CondVal) {
        replaceOperand(*TrueBO, 0, TrueBOSI->getTrueValue());
        Worklist.push(TrueBO);
        return &SI;
      }
    }
    if (auto *TrueBOSI = dyn_cast<SelectInst>(TrueBO->getOperand(1))) {
      if (TrueBOSI->getCondition() == CondVal) {
        replaceOperand(*TrueBO, 1, TrueBOSI->getTrueValue());
        Worklist.push(TrueBO);
        return &SI;
      }
    }
  }

  // select(C, Z, binop(select(C, X, Y), W)) -> select(C, Z, binop(Y, W))
  BinaryOperator *FalseBO;
  if (match(FalseVal, m_OneUse(m_BinOp(FalseBO))) &&
      canMergeSelectThroughBinop(FalseBO)) {
    if (auto *FalseBOSI = dyn_cast<SelectInst>(FalseBO->getOperand(0))) {
      if (FalseBOSI->getCondition() == CondVal) {
        replaceOperand(*FalseBO, 0, FalseBOSI->getFalseValue());
        Worklist.push(FalseBO);
        return &SI;
      }
    }
    if (auto *FalseBOSI = dyn_cast<SelectInst>(FalseBO->getOperand(1))) {
      if (FalseBOSI->getCondition() == CondVal) {
        replaceOperand(*FalseBO, 1, FalseBOSI->getFalseValue());
        Worklist.push(FalseBO);
        return &SI;
      }
    }
  }

  Value *NotCond;
  if (match(CondVal, m_Not(m_Value(NotCond))) &&
      !InstCombiner::shouldAvoidAbsorbingNotIntoSelect(SI)) {
    replaceOperand(SI, 0, NotCond);
    SI.swapValues();
    SI.swapProfMetadata();
    return &SI;
  }

  if (Instruction *I = foldVectorSelect(SI))
    return I;

  // If we can compute the condition, there's no need for a select.
  // Like the above fold, we are attempting to reduce compile-time cost by
  // putting this fold here with limitations rather than in InstSimplify.
  // The motivation for this call into value tracking is to take advantage of
  // the assumption cache, so make sure that is populated.
  if (!CondVal->getType()->isVectorTy() && !AC.assumptions().empty()) {
    KnownBits Known(1);
    computeKnownBits(CondVal, Known, 0, &SI);
    if (Known.One.isOneValue())
      return replaceInstUsesWith(SI, TrueVal);
    if (Known.Zero.isOneValue())
      return replaceInstUsesWith(SI, FalseVal);
  }

  if (Instruction *BitCastSel = foldSelectCmpBitcasts(SI, Builder))
    return BitCastSel;

  // Simplify selects that test the returned flag of cmpxchg instructions.
  if (Value *V = foldSelectCmpXchg(SI))
    return replaceInstUsesWith(SI, V);

  if (Instruction *Select = foldSelectBinOpIdentity(SI, TLI, *this))
    return Select;

  if (Instruction *Funnel = foldSelectFunnelShift(SI, Builder))
    return Funnel;

  if (Instruction *Copysign = foldSelectToCopysign(SI, Builder))
    return Copysign;

  if (Instruction *PN = foldSelectToPhi(SI, DT, Builder))
    return replaceInstUsesWith(SI, PN);

  if (Value *Fr = foldSelectWithFrozenICmp(SI, Builder))
    return replaceInstUsesWith(SI, Fr);

  return nullptr;
}
