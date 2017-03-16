//===- InstCombineSelect.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitSelect function.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

static SelectPatternFlavor
getInverseMinMaxSelectPattern(SelectPatternFlavor SPF) {
  switch (SPF) {
  default:
    llvm_unreachable("unhandled!");

  case SPF_SMIN:
    return SPF_SMAX;
  case SPF_UMIN:
    return SPF_UMAX;
  case SPF_SMAX:
    return SPF_SMIN;
  case SPF_UMAX:
    return SPF_UMIN;
  }
}

static CmpInst::Predicate getCmpPredicateForMinMax(SelectPatternFlavor SPF,
                                                   bool Ordered=false) {
  switch (SPF) {
  default:
    llvm_unreachable("unhandled!");

  case SPF_SMIN:
    return ICmpInst::ICMP_SLT;
  case SPF_UMIN:
    return ICmpInst::ICMP_ULT;
  case SPF_SMAX:
    return ICmpInst::ICMP_SGT;
  case SPF_UMAX:
    return ICmpInst::ICMP_UGT;
  case SPF_FMINNUM:
    return Ordered ? FCmpInst::FCMP_OLT : FCmpInst::FCMP_ULT;
  case SPF_FMAXNUM:
    return Ordered ? FCmpInst::FCMP_OGT : FCmpInst::FCMP_UGT;
  }
}

static Value *generateMinMaxSelectPattern(InstCombiner::BuilderTy *Builder,
                                          SelectPatternFlavor SPF, Value *A,
                                          Value *B) {
  CmpInst::Predicate Pred = getCmpPredicateForMinMax(SPF);
  assert(CmpInst::isIntPredicate(Pred));
  return Builder->CreateSelect(Builder->CreateICmp(Pred, A, B), A, B);
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
///
static unsigned getSelectFoldableOperands(Instruction *I) {
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

/// For the same transformation as the previous function, return the identity
/// constant that goes into the select.
static Constant *getSelectFoldableConstant(Instruction *I) {
  switch (I->getOpcode()) {
  default: llvm_unreachable("This cannot happen!");
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    return Constant::getNullValue(I->getType());
  case Instruction::And:
    return Constant::getAllOnesValue(I->getType());
  case Instruction::Mul:
    return ConstantInt::get(I->getType(), 1);
  }
}

/// We have (select c, TI, FI), and we know that TI and FI have the same opcode.
Instruction *InstCombiner::foldSelectOpOp(SelectInst &SI, Instruction *TI,
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
  if (TI->getNumOperands() == 1 && TI->isCast()) {
    Type *FIOpndTy = FI->getOperand(0)->getType();
    if (TI->getOperand(0)->getType() != FIOpndTy)
      return nullptr;

    // The select condition may be a vector. We may only change the operand
    // type if the vector width remains the same (and matches the condition).
    Type *CondTy = SI.getCondition()->getType();
    if (CondTy->isVectorTy()) {
      if (!FIOpndTy->isVectorTy())
        return nullptr;
      if (CondTy->getVectorNumElements() != FIOpndTy->getVectorNumElements())
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
        Builder->CreateSelect(SI.getCondition(), TI->getOperand(0),
                              FI->getOperand(0), SI.getName() + ".v", &SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI,
                            TI->getType());
  }

  // Only handle binary operators with one-use here. As with the cast case
  // above, it may be possible to relax the one-use constraint, but that needs
  // be examined carefully since it may not reduce the total number of
  // instructions.
  BinaryOperator *BO = dyn_cast<BinaryOperator>(TI);
  if (!BO || !TI->hasOneUse() || !FI->hasOneUse())
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

  // If we reach here, they do have operations in common.
  Value *NewSI = Builder->CreateSelect(SI.getCondition(), OtherOpT, OtherOpF,
                                       SI.getName() + ".v", &SI);
  Value *Op0 = MatchIsOpZero ? MatchOp : NewSI;
  Value *Op1 = MatchIsOpZero ? NewSI : MatchOp;
  return BinaryOperator::Create(BO->getOpcode(), Op0, Op1);
}

static bool isSelect01(Constant *C1, Constant *C2) {
  ConstantInt *C1I = dyn_cast<ConstantInt>(C1);
  if (!C1I)
    return false;
  ConstantInt *C2I = dyn_cast<ConstantInt>(C2);
  if (!C2I)
    return false;
  if (!C1I->isZero() && !C2I->isZero()) // One side must be zero.
    return false;
  return C1I->isOne() || C1I->isAllOnesValue() ||
         C2I->isOne() || C2I->isAllOnesValue();
}

/// Try to fold the select into one of the operands to allow further
/// optimization.
Instruction *InstCombiner::foldSelectIntoOp(SelectInst &SI, Value *TrueVal,
                                            Value *FalseVal) {
  // See the comment above GetSelectFoldableOperands for a description of the
  // transformation we are doing here.
  if (Instruction *TVI = dyn_cast<Instruction>(TrueVal)) {
    if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
        !isa<Constant>(FalseVal)) {
      if (unsigned SFO = getSelectFoldableOperands(TVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = getSelectFoldableConstant(TVI);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Value *NewSel = Builder->CreateSelect(SI.getCondition(), OOp, C);
            NewSel->takeName(TVI);
            BinaryOperator *TVI_BO = cast<BinaryOperator>(TVI);
            BinaryOperator *BO = BinaryOperator::Create(TVI_BO->getOpcode(),
                                                        FalseVal, NewSel);
            BO->copyIRFlags(TVI_BO);
            return BO;
          }
        }
      }
    }
  }

  if (Instruction *FVI = dyn_cast<Instruction>(FalseVal)) {
    if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
        !isa<Constant>(TrueVal)) {
      if (unsigned SFO = getSelectFoldableOperands(FVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = getSelectFoldableConstant(FVI);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Value *NewSel = Builder->CreateSelect(SI.getCondition(), C, OOp);
            NewSel->takeName(FVI);
            BinaryOperator *FVI_BO = cast<BinaryOperator>(FVI);
            BinaryOperator *BO = BinaryOperator::Create(FVI_BO->getOpcode(),
                                                        TrueVal, NewSel);
            BO->copyIRFlags(FVI_BO);
            return BO;
          }
        }
      }
    }
  }

  return nullptr;
}

/// We want to turn:
///   (select (icmp eq (and X, C1), 0), Y, (or Y, C2))
/// into:
///   (or (shl (and X, C1), C3), y)
/// iff:
///   C1 and C2 are both powers of 2
/// where:
///   C3 = Log(C2) - Log(C1)
///
/// This transform handles cases where:
/// 1. The icmp predicate is inverted
/// 2. The select operands are reversed
/// 3. The magnitude of C2 and C1 are flipped
static Value *foldSelectICmpAndOr(const SelectInst &SI, Value *TrueVal,
                                  Value *FalseVal,
                                  InstCombiner::BuilderTy *Builder) {
  const ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition());
  if (!IC || !IC->isEquality() || !SI.getType()->isIntegerTy())
    return nullptr;

  Value *CmpLHS = IC->getOperand(0);
  Value *CmpRHS = IC->getOperand(1);

  if (!match(CmpRHS, m_Zero()))
    return nullptr;

  Value *X;
  const APInt *C1;
  if (!match(CmpLHS, m_And(m_Value(X), m_Power2(C1))))
    return nullptr;

  const APInt *C2;
  bool OrOnTrueVal = false;
  bool OrOnFalseVal = match(FalseVal, m_Or(m_Specific(TrueVal), m_Power2(C2)));
  if (!OrOnFalseVal)
    OrOnTrueVal = match(TrueVal, m_Or(m_Specific(FalseVal), m_Power2(C2)));

  if (!OrOnFalseVal && !OrOnTrueVal)
    return nullptr;

  Value *V = CmpLHS;
  Value *Y = OrOnFalseVal ? TrueVal : FalseVal;

  unsigned C1Log = C1->logBase2();
  unsigned C2Log = C2->logBase2();
  if (C2Log > C1Log) {
    V = Builder->CreateZExtOrTrunc(V, Y->getType());
    V = Builder->CreateShl(V, C2Log - C1Log);
  } else if (C1Log > C2Log) {
    V = Builder->CreateLShr(V, C1Log - C2Log);
    V = Builder->CreateZExtOrTrunc(V, Y->getType());
  } else
    V = Builder->CreateZExtOrTrunc(V, Y->getType());

  ICmpInst::Predicate Pred = IC->getPredicate();
  if ((Pred == ICmpInst::ICMP_NE && OrOnFalseVal) ||
      (Pred == ICmpInst::ICMP_EQ && OrOnTrueVal))
    V = Builder->CreateXor(V, *C2);

  return Builder->CreateOr(V, Y);
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
                                 InstCombiner::BuilderTy *Builder) {
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);

  // Check if the condition value compares a value for equality against zero.
  if (!ICI->isEquality() || !match(CmpRHS, m_Zero()))
    return nullptr;

  Value *Count = FalseVal;
  Value *ValueOnZero = TrueVal;
  if (Pred == ICmpInst::ICMP_NE)
    std::swap(Count, ValueOnZero);

  // Skip zero extend/truncate.
  Value *V = nullptr;
  if (match(Count, m_ZExt(m_Value(V))) ||
      match(Count, m_Trunc(m_Value(V))))
    Count = V;

  // Check if the value propagated on zero is a constant number equal to the
  // sizeof in bits of 'Count'.
  unsigned SizeOfInBits = Count->getType()->getScalarSizeInBits();
  if (!match(ValueOnZero, m_SpecificInt(SizeOfInBits)))
    return nullptr;

  // Check that 'Count' is a call to intrinsic cttz/ctlz. Also check that the
  // input to the cttz/ctlz is used as LHS for the compare instruction.
  if (match(Count, m_Intrinsic<Intrinsic::cttz>(m_Specific(CmpLHS))) ||
      match(Count, m_Intrinsic<Intrinsic::ctlz>(m_Specific(CmpLHS)))) {
    IntrinsicInst *II = cast<IntrinsicInst>(Count);
    // Explicitly clear the 'undef_on_zero' flag.
    IntrinsicInst *NewI = cast<IntrinsicInst>(II->clone());
    Type *Ty = NewI->getArgOperand(1)->getType();
    NewI->setArgOperand(1, Constant::getNullValue(Ty));
    Builder->Insert(NewI);
    return Builder->CreateZExtOrTrunc(NewI, ValueOnZero->getType());
  }

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
static Instruction *
canonicalizeMinMaxWithConstant(SelectInst &Sel, ICmpInst &Cmp,
                               InstCombiner::BuilderTy &Builder) {
  if (!Cmp.hasOneUse() || !isa<Constant>(Cmp.getOperand(1)))
    return nullptr;

  // Canonicalize the compare predicate based on whether we have min or max.
  Value *LHS, *RHS;
  ICmpInst::Predicate NewPred;
  SelectPatternResult SPR = matchSelectPattern(&Sel, LHS, RHS);
  switch (SPR.Flavor) {
  case SPF_SMIN: NewPred = ICmpInst::ICMP_SLT; break;
  case SPF_UMIN: NewPred = ICmpInst::ICMP_ULT; break;
  case SPF_SMAX: NewPred = ICmpInst::ICMP_SGT; break;
  case SPF_UMAX: NewPred = ICmpInst::ICMP_UGT; break;
  default: return nullptr;
  }

  // Is this already canonical?
  if (Cmp.getOperand(0) == LHS && Cmp.getOperand(1) == RHS &&
      Cmp.getPredicate() == NewPred)
    return nullptr;

  // Create the canonical compare and plug it into the select.
  Sel.setCondition(Builder.CreateICmp(NewPred, LHS, RHS));

  // If the select operands did not change, we're done.
  if (Sel.getTrueValue() == LHS && Sel.getFalseValue() == RHS)
    return &Sel;

  // If we are swapping the select operands, swap the metadata too.
  assert(Sel.getTrueValue() == RHS && Sel.getFalseValue() == LHS &&
         "Unexpected results from matchSelectPattern");
  Sel.setTrueValue(LHS);
  Sel.setFalseValue(RHS);
  Sel.swapProfMetadata();
  return &Sel;
}

/// Visit a SelectInst that has an ICmpInst as its first operand.
Instruction *InstCombiner::foldSelectInstWithICmp(SelectInst &SI,
                                                  ICmpInst *ICI) {
  if (Instruction *NewSel = canonicalizeMinMaxWithConstant(SI, *ICI, *Builder))
    return NewSel;

  bool Changed = adjustMinMax(SI, *ICI);

  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // Transform (X >s -1) ? C1 : C2 --> ((X >>s 31) & (C2 - C1)) + C1
  // and       (X <s  0) ? C2 : C1 --> ((X >>s 31) & (C2 - C1)) + C1
  // FIXME: Type and constness constraints could be lifted, but we have to
  //        watch code size carefully. We should consider xor instead of
  //        sub/add when we decide to do that.
  if (IntegerType *Ty = dyn_cast<IntegerType>(CmpLHS->getType())) {
    if (TrueVal->getType() == Ty) {
      if (ConstantInt *Cmp = dyn_cast<ConstantInt>(CmpRHS)) {
        ConstantInt *C1 = nullptr, *C2 = nullptr;
        if (Pred == ICmpInst::ICMP_SGT && Cmp->isAllOnesValue()) {
          C1 = dyn_cast<ConstantInt>(TrueVal);
          C2 = dyn_cast<ConstantInt>(FalseVal);
        } else if (Pred == ICmpInst::ICMP_SLT && Cmp->isNullValue()) {
          C1 = dyn_cast<ConstantInt>(FalseVal);
          C2 = dyn_cast<ConstantInt>(TrueVal);
        }
        if (C1 && C2) {
          // This shift results in either -1 or 0.
          Value *AShr = Builder->CreateAShr(CmpLHS, Ty->getBitWidth()-1);

          // Check if we can express the operation with a single or.
          if (C2->isAllOnesValue())
            return replaceInstUsesWith(SI, Builder->CreateOr(AShr, C1));

          Value *And = Builder->CreateAnd(AShr, C2->getValue()-C1->getValue());
          return replaceInstUsesWith(SI, Builder->CreateAdd(And, C1));
        }
      }
    }
  }

  // NOTE: if we wanted to, this is where to detect integer MIN/MAX

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
    APInt MinSignedValue = APInt::getSignBit(BitWidth);
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
        V = Builder->CreateAnd(X, ~(*Y));
      // (X & Y) != 0 ? X ^ Y : X  --> X & ~Y
      else if (!TrueWhenUnset && FalseVal == X &&
               match(TrueVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder->CreateAnd(X, ~(*Y));
      // (X & Y) == 0 ? X ^ Y : X  --> X | Y
      else if (TrueWhenUnset && FalseVal == X &&
               match(TrueVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder->CreateOr(X, *Y);
      // (X & Y) != 0 ? X : X ^ Y  --> X | Y
      else if (!TrueWhenUnset && TrueVal == X &&
               match(FalseVal, m_Xor(m_Specific(X), m_APInt(C))) && *Y == *C)
        V = Builder->CreateOr(X, *Y);

      if (V)
        return replaceInstUsesWith(SI, V);
    }
  }

  if (Value *V = foldSelectICmpAndOr(SI, TrueVal, FalseVal, Builder))
    return replaceInstUsesWith(SI, V);

  if (Value *V = foldSelectCttzCtlz(ICI, TrueVal, FalseVal, Builder))
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
///
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
Instruction *InstCombiner::foldSPFofSPF(Instruction *Inner,
                                        SelectPatternFlavor SPF1,
                                        Value *A, Value *B,
                                        Instruction &Outer,
                                        SelectPatternFlavor SPF2, Value *C) {
  if (Outer.getType() != Inner->getType())
    return nullptr;

  if (C == A || C == B) {
    // MAX(MAX(A, B), B) -> MAX(A, B)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    if (SPF1 == SPF2)
      return replaceInstUsesWith(Outer, Inner);

    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
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

  // ABS(ABS(X)) -> ABS(X)
  // NABS(NABS(X)) -> NABS(X)
  if (SPF1 == SPF2 && (SPF1 == SPF_ABS || SPF1 == SPF_NABS)) {
    return replaceInstUsesWith(Outer, Inner);
  }

  // ABS(NABS(X)) -> ABS(X)
  // NABS(ABS(X)) -> NABS(X)
  if ((SPF1 == SPF_ABS && SPF2 == SPF_NABS) ||
      (SPF1 == SPF_NABS && SPF2 == SPF_ABS)) {
    SelectInst *SI = cast<SelectInst>(Inner);
    Value *NewSI =
        Builder->CreateSelect(SI->getCondition(), SI->getFalseValue(),
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

    if (IsFreeToInvert(V, !V->hasNUsesOrMore(3))) {
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
      NotA = Builder->CreateNot(A);
    if (!NotB)
      NotB = Builder->CreateNot(B);
    if (!NotC)
      NotC = Builder->CreateNot(C);

    Value *NewInner = generateMinMaxSelectPattern(
        Builder, getInverseMinMaxSelectPattern(SPF1), NotA, NotB);
    Value *NewOuter = Builder->CreateNot(generateMinMaxSelectPattern(
        Builder, getInverseMinMaxSelectPattern(SPF2), NewInner, NotC));
    return replaceInstUsesWith(Outer, NewOuter);
  }

  return nullptr;
}

/// If one of the constants is zero (we know they can't both be) and we have an
/// icmp instruction with zero, and we have an 'and' with the non-constant value
/// and a power of two we can turn the select into a shift on the result of the
/// 'and'.
static Value *foldSelectICmpAnd(const SelectInst &SI, ConstantInt *TrueVal,
                                ConstantInt *FalseVal,
                                InstCombiner::BuilderTy *Builder) {
  const ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition());
  if (!IC || !IC->isEquality() || !SI.getType()->isIntegerTy())
    return nullptr;

  if (!match(IC->getOperand(1), m_Zero()))
    return nullptr;

  ConstantInt *AndRHS;
  Value *LHS = IC->getOperand(0);
  if (!match(LHS, m_And(m_Value(), m_ConstantInt(AndRHS))))
    return nullptr;

  // If both select arms are non-zero see if we have a select of the form
  // 'x ? 2^n + C : C'. Then we can offset both arms by C, use the logic
  // for 'x ? 2^n : 0' and fix the thing up at the end.
  ConstantInt *Offset = nullptr;
  if (!TrueVal->isZero() && !FalseVal->isZero()) {
    if ((TrueVal->getValue() - FalseVal->getValue()).isPowerOf2())
      Offset = FalseVal;
    else if ((FalseVal->getValue() - TrueVal->getValue()).isPowerOf2())
      Offset = TrueVal;
    else
      return nullptr;

    // Adjust TrueVal and FalseVal to the offset.
    TrueVal = ConstantInt::get(Builder->getContext(),
                               TrueVal->getValue() - Offset->getValue());
    FalseVal = ConstantInt::get(Builder->getContext(),
                                FalseVal->getValue() - Offset->getValue());
  }

  // Make sure the mask in the 'and' and one of the select arms is a power of 2.
  if (!AndRHS->getValue().isPowerOf2() ||
      (!TrueVal->getValue().isPowerOf2() &&
       !FalseVal->getValue().isPowerOf2()))
    return nullptr;

  // Determine which shift is needed to transform result of the 'and' into the
  // desired result.
  ConstantInt *ValC = !TrueVal->isZero() ? TrueVal : FalseVal;
  unsigned ValZeros = ValC->getValue().logBase2();
  unsigned AndZeros = AndRHS->getValue().logBase2();

  // If types don't match we can still convert the select by introducing a zext
  // or a trunc of the 'and'. The trunc case requires that all of the truncated
  // bits are zero, we can figure that out by looking at the 'and' mask.
  if (AndZeros >= ValC->getBitWidth())
    return nullptr;

  Value *V = Builder->CreateZExtOrTrunc(LHS, SI.getType());
  if (ValZeros > AndZeros)
    V = Builder->CreateShl(V, ValZeros - AndZeros);
  else if (ValZeros < AndZeros)
    V = Builder->CreateLShr(V, AndZeros - ValZeros);

  // Okay, now we know that everything is set up, we just don't know whether we
  // have a icmp_ne or icmp_eq and whether the true or false val is the zero.
  bool ShouldNotVal = !TrueVal->isZero();
  ShouldNotVal ^= IC->getPredicate() == ICmpInst::ICMP_NE;
  if (ShouldNotVal)
    V = Builder->CreateXor(V, ValC);

  // Apply an offset if needed.
  if (Offset)
    V = Builder->CreateAdd(V, Offset);
  return V;
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

Instruction *InstCombiner::foldSelectExtConst(SelectInst &Sel) {
  Instruction *ExtInst;
  if (!match(Sel.getTrueValue(), m_Instruction(ExtInst)) &&
      !match(Sel.getFalseValue(), m_Instruction(ExtInst)))
    return nullptr;

  auto ExtOpcode = ExtInst->getOpcode();
  if (ExtOpcode != Instruction::ZExt && ExtOpcode != Instruction::SExt)
    return nullptr;

  // TODO: Handle larger types? That requires adjusting FoldOpIntoSelect too.
  Value *X = ExtInst->getOperand(0);
  Type *SmallType = X->getType();
  if (!SmallType->getScalarType()->isIntegerTy(1))
    return nullptr;

  Constant *C;
  if (!match(Sel.getTrueValue(), m_Constant(C)) &&
      !match(Sel.getFalseValue(), m_Constant(C)))
    return nullptr;

  // If the constant is the same after truncation to the smaller type and
  // extension to the original type, we can narrow the select.
  Value *Cond = Sel.getCondition();
  Type *SelType = Sel.getType();
  Constant *TruncC = ConstantExpr::getTrunc(C, SmallType);
  Constant *ExtC = ConstantExpr::getCast(ExtOpcode, TruncC, SelType);
  if (ExtC == C) {
    Value *TruncCVal = cast<Value>(TruncC);
    if (ExtInst == Sel.getFalseValue())
      std::swap(X, TruncCVal);

    // select Cond, (ext X), C --> ext(select Cond, X, C')
    // select Cond, C, (ext X) --> ext(select Cond, C', X)
    Value *NewSel = Builder->CreateSelect(Cond, X, TruncCVal, "narrow", &Sel);
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
  if (!CondVal->getType()->isVectorTy() || !match(CondVal, m_Constant(CondC)))
    return nullptr;

  unsigned NumElts = CondVal->getType()->getVectorNumElements();
  SmallVector<Constant *, 16> Mask;
  Mask.reserve(NumElts);
  Type *Int32Ty = Type::getInt32Ty(CondVal->getContext());
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *Elt = CondC->getAggregateElement(i);
    if (!Elt)
      return nullptr;

    if (Elt->isOneValue()) {
      // If the select condition element is true, choose from the 1st vector.
      Mask.push_back(ConstantInt::get(Int32Ty, i));
    } else if (Elt->isNullValue()) {
      // If the select condition element is false, choose from the 2nd vector.
      Mask.push_back(ConstantInt::get(Int32Ty, i + NumElts));
    } else if (isa<UndefValue>(Elt)) {
      // If the select condition element is undef, the shuffle mask is undef.
      Mask.push_back(UndefValue::get(Int32Ty));
    } else {
      // Bail out on a constant expression.
      return nullptr;
    }
  }

  return new ShuffleVectorInst(SI.getTrueValue(), SI.getFalseValue(),
                               ConstantVector::get(Mask));
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

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  Type *SelType = SI.getType();

  if (Value *V =
          SimplifySelectInst(CondVal, TrueVal, FalseVal, DL, &TLI, &DT, &AC))
    return replaceInstUsesWith(SI, V);

  if (Instruction *I = canonicalizeSelectToShuffle(SI))
    return I;

  if (SelType->getScalarType()->isIntegerTy(1) &&
      TrueVal->getType() == CondVal->getType()) {
    if (match(TrueVal, m_One())) {
      // Change: A = select B, true, C --> A = or B, C
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    }
    if (match(TrueVal, m_Zero())) {
      // Change: A = select B, false, C --> A = and !B, C
      Value *NotCond = Builder->CreateNot(CondVal, "not." + CondVal->getName());
      return BinaryOperator::CreateAnd(NotCond, FalseVal);
    }
    if (match(FalseVal, m_Zero())) {
      // Change: A = select B, C, false --> A = and B, C
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
    }
    if (match(FalseVal, m_One())) {
      // Change: A = select B, C, true --> A = or !B, C
      Value *NotCond = Builder->CreateNot(CondVal, "not." + CondVal->getName());
      return BinaryOperator::CreateOr(NotCond, TrueVal);
    }

    // select a, a, b  -> a | b
    // select a, b, a  -> a & b
    if (CondVal == TrueVal)
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    if (CondVal == FalseVal)
      return BinaryOperator::CreateAnd(CondVal, TrueVal);

    // select a, ~a, b -> (~a) & b
    // select a, b, ~a -> (~a) | b
    if (match(TrueVal, m_Not(m_Specific(CondVal))))
      return BinaryOperator::CreateAnd(TrueVal, FalseVal);
    if (match(FalseVal, m_Not(m_Specific(CondVal))))
      return BinaryOperator::CreateOr(TrueVal, FalseVal);
  }

  // Selecting between two integer or vector splat integer constants?
  //
  // Note that we don't handle a scalar select of vectors:
  // select i1 %c, <2 x i8> <1, 1>, <2 x i8> <0, 0>
  // because that may need 3 instructions to splat the condition value:
  // extend, insertelement, shufflevector.
  if (CondVal->getType()->isVectorTy() == SelType->isVectorTy()) {
    // select C, 1, 0 -> zext C to int
    if (match(TrueVal, m_One()) && match(FalseVal, m_Zero()))
      return new ZExtInst(CondVal, SelType);

    // select C, -1, 0 -> sext C to int
    if (match(TrueVal, m_AllOnes()) && match(FalseVal, m_Zero()))
      return new SExtInst(CondVal, SelType);

    // select C, 0, 1 -> zext !C to int
    if (match(TrueVal, m_Zero()) && match(FalseVal, m_One())) {
      Value *NotCond = Builder->CreateNot(CondVal, "not." + CondVal->getName());
      return new ZExtInst(NotCond, SelType);
    }

    // select C, 0, -1 -> sext !C to int
    if (match(TrueVal, m_Zero()) && match(FalseVal, m_AllOnes())) {
      Value *NotCond = Builder->CreateNot(CondVal, "not." + CondVal->getName());
      return new SExtInst(NotCond, SelType);
    }
  }

  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal))
      if (Value *V = foldSelectICmpAnd(SI, TrueValC, FalseValC, Builder))
        return replaceInstUsesWith(SI, V);

  // See if we are selecting two values based on a comparison of the two values.
  if (FCmpInst *FCI = dyn_cast<FCmpInst>(CondVal)) {
    if (FCI->getOperand(0) == TrueVal && FCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return replaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X une Y) ? X : Y  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_UNE) {
        // This is not safe in general for floating point:
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return replaceInstUsesWith(SI, TrueVal);
      }

      // Canonicalize to use ordered comparisons by swapping the select
      // operands.
      //
      // e.g.
      // (X ugt Y) ? X : Y -> (X ole Y) ? Y : X
      if (FCI->hasOneUse() && FCmpInst::isUnordered(FCI->getPredicate())) {
        FCmpInst::Predicate InvPred = FCI->getInversePredicate();
        IRBuilder<>::FastMathFlagGuard FMFG(*Builder);
        Builder->setFastMathFlags(FCI->getFastMathFlags());
        Value *NewCond = Builder->CreateFCmp(InvPred, TrueVal, FalseVal,
                                             FCI->getName() + ".inv");

        return SelectInst::Create(NewCond, FalseVal, TrueVal,
                                  SI.getName() + ".p");
      }

      // NOTE: if we wanted to, this is where to detect MIN/MAX
    } else if (FCI->getOperand(0) == FalseVal && FCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return replaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X une Y) ? Y : X  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_UNE) {
        // This is not safe in general for floating point:
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return replaceInstUsesWith(SI, TrueVal);
      }

      // Canonicalize to use ordered comparisons by swapping the select
      // operands.
      //
      // e.g.
      // (X ugt Y) ? X : Y -> (X ole Y) ? X : Y
      if (FCI->hasOneUse() && FCmpInst::isUnordered(FCI->getPredicate())) {
        FCmpInst::Predicate InvPred = FCI->getInversePredicate();
        IRBuilder<>::FastMathFlagGuard FMFG(*Builder);
        Builder->setFastMathFlags(FCI->getFastMathFlags());
        Value *NewCond = Builder->CreateFCmp(InvPred, FalseVal, TrueVal,
                                             FCI->getName() + ".inv");

        return SelectInst::Create(NewCond, FalseVal, TrueVal,
                                  SI.getName() + ".p");
      }

      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
    // NOTE: if we wanted to, this is where to detect ABS
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = foldSelectInstWithICmp(SI, ICI))
      return Result;

  if (Instruction *Add = foldAddSubSelect(SI, *Builder))
    return Add;

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

    Value *LHS, *RHS, *LHS2, *RHS2;
    Instruction::CastOps CastOp;
    SelectPatternResult SPR = matchSelectPattern(&SI, LHS, RHS, &CastOp);
    auto SPF = SPR.Flavor;

    if (SelectPatternResult::isMinOrMax(SPF)) {
      // Canonicalize so that type casts are outside select patterns.
      if (LHS->getType()->getPrimitiveSizeInBits() !=
          SelType->getPrimitiveSizeInBits()) {
        CmpInst::Predicate Pred = getCmpPredicateForMinMax(SPF, SPR.Ordered);

        Value *Cmp;
        if (CmpInst::isIntPredicate(Pred)) {
          Cmp = Builder->CreateICmp(Pred, LHS, RHS);
        } else {
          IRBuilder<>::FastMathFlagGuard FMFG(*Builder);
          auto FMF = cast<FPMathOperator>(SI.getCondition())->getFastMathFlags();
          Builder->setFastMathFlags(FMF);
          Cmp = Builder->CreateFCmp(Pred, LHS, RHS);
        }

        Value *NewSI = Builder->CreateCast(
            CastOp, Builder->CreateSelect(Cmp, LHS, RHS, SI.getName(), &SI),
            SelType);
        return replaceInstUsesWith(SI, NewSI);
      }
    }

    if (SPF) {
      // MAX(MAX(a, b), a) -> MAX(a, b)
      // MIN(MIN(a, b), a) -> MIN(a, b)
      // MAX(MIN(a, b), a) -> a
      // MIN(MAX(a, b), a) -> a
      // ABS(ABS(a)) -> ABS(a)
      // NABS(NABS(a)) -> NABS(a)
      if (SelectPatternFlavor SPF2 = matchSelectPattern(LHS, LHS2, RHS2).Flavor)
        if (Instruction *R = foldSPFofSPF(cast<Instruction>(LHS),SPF2,LHS2,RHS2,
                                          SI, SPF, RHS))
          return R;
      if (SelectPatternFlavor SPF2 = matchSelectPattern(RHS, LHS2, RHS2).Flavor)
        if (Instruction *R = foldSPFofSPF(cast<Instruction>(RHS),SPF2,LHS2,RHS2,
                                          SI, SPF, LHS))
          return R;
    }

    // MAX(~a, ~b) -> ~MIN(a, b)
    if ((SPF == SPF_SMAX || SPF == SPF_UMAX) &&
        IsFreeToInvert(LHS, LHS->hasNUses(2)) &&
        IsFreeToInvert(RHS, RHS->hasNUses(2))) {
      // For this transform to be profitable, we need to eliminate at least two
      // 'not' instructions if we're going to add one 'not' instruction.
      int NumberOfNots =
          (LHS->hasNUses(2) && match(LHS, m_Not(m_Value()))) +
          (RHS->hasNUses(2) && match(RHS, m_Not(m_Value()))) +
          (SI.hasOneUse() && match(*SI.user_begin(), m_Not(m_Value())));

      if (NumberOfNots >= 2) {
        Value *NewLHS = Builder->CreateNot(LHS);
        Value *NewRHS = Builder->CreateNot(RHS);
        Value *NewCmp = SPF == SPF_SMAX
                            ? Builder->CreateICmpSLT(NewLHS, NewRHS)
                            : Builder->CreateICmpULT(NewLHS, NewRHS);
        Value *NewSI =
            Builder->CreateNot(Builder->CreateSelect(NewCmp, NewLHS, NewRHS));
        return replaceInstUsesWith(SI, NewSI);
      }
    }

    // TODO.
    // ABS(-X) -> ABS(X)
  }

  // See if we can fold the select into a phi node if the condition is a select.
  if (isa<PHINode>(SI.getCondition()))
    // The true/false values have to be live in the PHI predecessor's blocks.
    if (canSelectOperandBeMappingIntoPredBlock(TrueVal, SI) &&
        canSelectOperandBeMappingIntoPredBlock(FalseVal, SI))
      if (Instruction *NV = FoldOpIntoPhi(SI))
        return NV;

  if (SelectInst *TrueSI = dyn_cast<SelectInst>(TrueVal)) {
    if (TrueSI->getCondition()->getType() == CondVal->getType()) {
      // select(C, select(C, a, b), c) -> select(C, a, c)
      if (TrueSI->getCondition() == CondVal) {
        if (SI.getTrueValue() == TrueSI->getTrueValue())
          return nullptr;
        SI.setOperand(1, TrueSI->getTrueValue());
        return &SI;
      }
      // select(C0, select(C1, a, b), b) -> select(C0&C1, a, b)
      // We choose this as normal form to enable folding on the And and shortening
      // paths for the values (this helps GetUnderlyingObjects() for example).
      if (TrueSI->getFalseValue() == FalseVal && TrueSI->hasOneUse()) {
        Value *And = Builder->CreateAnd(CondVal, TrueSI->getCondition());
        SI.setOperand(0, And);
        SI.setOperand(1, TrueSI->getTrueValue());
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
        SI.setOperand(2, FalseSI->getFalseValue());
        return &SI;
      }
      // select(C0, a, select(C1, a, b)) -> select(C0|C1, a, b)
      if (FalseSI->getTrueValue() == TrueVal && FalseSI->hasOneUse()) {
        Value *Or = Builder->CreateOr(CondVal, FalseSI->getCondition());
        SI.setOperand(0, Or);
        SI.setOperand(2, FalseSI->getFalseValue());
        return &SI;
      }
    }
  }

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  if (VectorType *VecTy = dyn_cast<VectorType>(SelType)) {
    unsigned VWidth = VecTy->getNumElements();
    APInt UndefElts(VWidth, 0);
    APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
    if (Value *V = SimplifyDemandedVectorElts(&SI, AllOnesEltMask, UndefElts)) {
      if (V != &SI)
        return replaceInstUsesWith(SI, V);
      return &SI;
    }

    if (isa<ConstantAggregateZero>(CondVal)) {
      return replaceInstUsesWith(SI, FalseVal);
    }
  }

  // See if we can determine the result of this select based on a dominating
  // condition.
  BasicBlock *Parent = SI.getParent();
  if (BasicBlock *Dom = Parent->getSinglePredecessor()) {
    auto *PBI = dyn_cast_or_null<BranchInst>(Dom->getTerminator());
    if (PBI && PBI->isConditional() &&
        PBI->getSuccessor(0) != PBI->getSuccessor(1) &&
        (PBI->getSuccessor(0) == Parent || PBI->getSuccessor(1) == Parent)) {
      bool CondIsFalse = PBI->getSuccessor(1) == Parent;
      Optional<bool> Implication = isImpliedCondition(
        PBI->getCondition(), SI.getCondition(), DL, CondIsFalse);
      if (Implication) {
        Value *V = *Implication ? TrueVal : FalseVal;
        return replaceInstUsesWith(SI, V);
      }
    }
  }

  // If we can compute the condition, there's no need for a select.
  // Like the above fold, we are attempting to reduce compile-time cost by
  // putting this fold here with limitations rather than in InstSimplify.
  // The motivation for this call into value tracking is to take advantage of
  // the assumption cache, so make sure that is populated.
  if (!CondVal->getType()->isVectorTy() && !AC.assumptions().empty()) {
    APInt KnownOne(1, 0), KnownZero(1, 0);
    computeKnownBits(CondVal, KnownZero, KnownOne, 0, &SI);
    if (KnownOne == 1)
      return replaceInstUsesWith(SI, TrueVal);
    if (KnownZero == 1)
      return replaceInstUsesWith(SI, FalseVal);
  }

  if (Instruction *BitCastSel = foldSelectCmpBitcasts(SI, *Builder))
    return BitCastSel;

  return nullptr;
}
