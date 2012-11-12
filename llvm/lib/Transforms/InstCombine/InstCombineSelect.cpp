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

#include "InstCombine.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
using namespace llvm;
using namespace PatternMatch;

/// MatchSelectPattern - Pattern match integer [SU]MIN, [SU]MAX, and ABS idioms,
/// returning the kind and providing the out parameter results if we
/// successfully match.
static SelectPatternFlavor
MatchSelectPattern(Value *V, Value *&LHS, Value *&RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (SI == 0) return SPF_UNKNOWN;

  ICmpInst *ICI = dyn_cast<ICmpInst>(SI->getCondition());
  if (ICI == 0) return SPF_UNKNOWN;

  LHS = ICI->getOperand(0);
  RHS = ICI->getOperand(1);

  // (icmp X, Y) ? X : Y
  if (SI->getTrueValue() == ICI->getOperand(0) &&
      SI->getFalseValue() == ICI->getOperand(1)) {
    switch (ICI->getPredicate()) {
    default: return SPF_UNKNOWN; // Equality.
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return SPF_UMAX;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return SPF_SMAX;
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE: return SPF_UMIN;
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: return SPF_SMIN;
    }
  }

  // (icmp X, Y) ? Y : X
  if (SI->getTrueValue() == ICI->getOperand(1) &&
      SI->getFalseValue() == ICI->getOperand(0)) {
    switch (ICI->getPredicate()) {
      default: return SPF_UNKNOWN; // Equality.
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE: return SPF_UMIN;
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE: return SPF_SMIN;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE: return SPF_UMAX;
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE: return SPF_SMAX;
    }
  }

  // TODO: (X > 4) ? X : 5   -->  (X >= 5) ? X : 5  -->  MAX(X, 5)

  return SPF_UNKNOWN;
}


/// GetSelectFoldableOperands - We want to turn code that looks like this:
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
static unsigned GetSelectFoldableOperands(Instruction *I) {
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

/// GetSelectFoldableConstant - For the same transformation as the previous
/// function, return the identity constant that goes into the select.
static Constant *GetSelectFoldableConstant(Instruction *I) {
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

/// FoldSelectOpOp - Here we have (select c, TI, FI), and we know that TI and FI
/// have the same opcode and only one use each.  Try to simplify this.
Instruction *InstCombiner::FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                          Instruction *FI) {
  if (TI->getNumOperands() == 1) {
    // If this is a non-volatile load or a cast from the same type,
    // merge.
    if (TI->isCast()) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
      // The select condition may be a vector. We may only change the operand
      // type if the vector width remains the same (and matches the condition).
      Type *CondTy = SI.getCondition()->getType();
      if (CondTy->isVectorTy() && CondTy->getVectorNumElements() !=
          FI->getOperand(0)->getType()->getVectorNumElements())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    Value *NewSI = Builder->CreateSelect(SI.getCondition(), TI->getOperand(0),
                                         FI->getOperand(0), SI.getName()+".v");
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI,
                            TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<BinaryOperator>(TI))
    return 0;

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
    return 0;
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
    return 0;
  }

  // If we reach here, they do have operations in common.
  Value *NewSI = Builder->CreateSelect(SI.getCondition(), OtherOpT,
                                       OtherOpF, SI.getName()+".v");

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::Create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::Create(BO->getOpcode(), NewSI, MatchOp);
  }
  llvm_unreachable("Shouldn't get here");
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

/// FoldSelectIntoOp - Try fold the select into one of the operands to
/// facilitate further optimization.
Instruction *InstCombiner::FoldSelectIntoOp(SelectInst &SI, Value *TrueVal,
                                            Value *FalseVal) {
  // See the comment above GetSelectFoldableOperands for a description of the
  // transformation we are doing here.
  if (Instruction *TVI = dyn_cast<Instruction>(TrueVal)) {
    if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
        !isa<Constant>(FalseVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(TVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(TVI);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Value *NewSel = Builder->CreateSelect(SI.getCondition(), OOp, C);
            NewSel->takeName(TVI);
            BinaryOperator *TVI_BO = cast<BinaryOperator>(TVI);
            BinaryOperator *BO = BinaryOperator::Create(TVI_BO->getOpcode(),
                                                        FalseVal, NewSel);
            if (isa<PossiblyExactOperator>(BO))
              BO->setIsExact(TVI_BO->isExact());
            if (isa<OverflowingBinaryOperator>(BO)) {
              BO->setHasNoUnsignedWrap(TVI_BO->hasNoUnsignedWrap());
              BO->setHasNoSignedWrap(TVI_BO->hasNoSignedWrap());
            }
            return BO;
          }
        }
      }
    }
  }

  if (Instruction *FVI = dyn_cast<Instruction>(FalseVal)) {
    if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
        !isa<Constant>(TrueVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(FVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
          OpToFold = 1;
        } else if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(FVI);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Value *NewSel = Builder->CreateSelect(SI.getCondition(), C, OOp);
            NewSel->takeName(FVI);
            BinaryOperator *FVI_BO = cast<BinaryOperator>(FVI);
            BinaryOperator *BO = BinaryOperator::Create(FVI_BO->getOpcode(),
                                                        TrueVal, NewSel);
            if (isa<PossiblyExactOperator>(BO))
              BO->setIsExact(FVI_BO->isExact());
            if (isa<OverflowingBinaryOperator>(BO)) {
              BO->setHasNoUnsignedWrap(FVI_BO->hasNoUnsignedWrap());
              BO->setHasNoSignedWrap(FVI_BO->hasNoSignedWrap());
            }
            return BO;
          }
        }
      }
    }
  }

  return 0;
}

/// SimplifyWithOpReplaced - See if V simplifies when its operand Op is
/// replaced with RepOp.
static Value *SimplifyWithOpReplaced(Value *V, Value *Op, Value *RepOp,
                                     const DataLayout *TD,
                                     const TargetLibraryInfo *TLI) {
  // Trivial replacement.
  if (V == Op)
    return RepOp;

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return 0;

  // If this is a binary operator, try to simplify it with the replaced op.
  if (BinaryOperator *B = dyn_cast<BinaryOperator>(I)) {
    if (B->getOperand(0) == Op)
      return SimplifyBinOp(B->getOpcode(), RepOp, B->getOperand(1), TD, TLI);
    if (B->getOperand(1) == Op)
      return SimplifyBinOp(B->getOpcode(), B->getOperand(0), RepOp, TD, TLI);
  }

  // Same for CmpInsts.
  if (CmpInst *C = dyn_cast<CmpInst>(I)) {
    if (C->getOperand(0) == Op)
      return SimplifyCmpInst(C->getPredicate(), RepOp, C->getOperand(1), TD,
                             TLI);
    if (C->getOperand(1) == Op)
      return SimplifyCmpInst(C->getPredicate(), C->getOperand(0), RepOp, TD,
                             TLI);
  }

  // TODO: We could hand off more cases to instsimplify here.

  // If all operands are constant after substituting Op for RepOp then we can
  // constant fold the instruction.
  if (Constant *CRepOp = dyn_cast<Constant>(RepOp)) {
    // Build a list of all constant operands.
    SmallVector<Constant*, 8> ConstOps;
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      if (I->getOperand(i) == Op)
        ConstOps.push_back(CRepOp);
      else if (Constant *COp = dyn_cast<Constant>(I->getOperand(i)))
        ConstOps.push_back(COp);
      else
        break;
    }

    // All operands were constants, fold it.
    if (ConstOps.size() == I->getNumOperands()) {
      if (CmpInst *C = dyn_cast<CmpInst>(I))
        return ConstantFoldCompareInstOperands(C->getPredicate(), ConstOps[0],
                                               ConstOps[1], TD, TLI);

      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        if (!LI->isVolatile())
          return ConstantFoldLoadFromConstPtr(ConstOps[0], TD);

      return ConstantFoldInstOperands(I->getOpcode(), I->getType(),
                                      ConstOps, TD, TLI);
    }
  }

  return 0;
}

/// visitSelectInstWithICmp - Visit a SelectInst that has an
/// ICmpInst as its first operand.
///
Instruction *InstCombiner::visitSelectInstWithICmp(SelectInst &SI,
                                                   ICmpInst *ICI) {
  bool Changed = false;
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // Check cases where the comparison is with a constant that
  // can be adjusted to fit the min/max idiom. We may move or edit ICI
  // here, so make sure the select is the only user.
  if (ICI->hasOneUse())
    if (ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS)) {
      // X < MIN ? T : F  -->  F
      if ((Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_ULT)
          && CI->isMinValue(Pred == ICmpInst::ICMP_SLT))
        return ReplaceInstUsesWith(SI, FalseVal);
      // X > MAX ? T : F  -->  F
      else if ((Pred == ICmpInst::ICMP_SGT || Pred == ICmpInst::ICMP_UGT)
               && CI->isMaxValue(Pred == ICmpInst::ICMP_SGT))
        return ReplaceInstUsesWith(SI, FalseVal);
      switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT: {
        // These transformations only work for selects over integers.
        IntegerType *SelectTy = dyn_cast<IntegerType>(SI.getType());
        if (!SelectTy)
          break;

        Constant *AdjustedRHS;
        if (Pred == ICmpInst::ICMP_UGT || Pred == ICmpInst::ICMP_SGT)
          AdjustedRHS = ConstantInt::get(CI->getContext(), CI->getValue() + 1);
        else // (Pred == ICmpInst::ICMP_ULT || Pred == ICmpInst::ICMP_SLT)
          AdjustedRHS = ConstantInt::get(CI->getContext(), CI->getValue() - 1);

        // X > C ? X : C+1  -->  X < C+1 ? C+1 : X
        // X < C ? X : C-1  -->  X > C-1 ? C-1 : X
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal))
          ; // Nothing to do here. Values match without any sign/zero extension.

        // Types do not match. Instead of calculating this with mixed types
        // promote all to the larger type. This enables scalar evolution to
        // analyze this expression.
        else if (CmpRHS->getType()->getScalarSizeInBits()
                 < SelectTy->getBitWidth()) {
          Constant *sextRHS = ConstantExpr::getSExt(AdjustedRHS, SelectTy);

          // X = sext x; x >s c ? X : C+1 --> X = sext x; X <s C+1 ? C+1 : X
          // X = sext x; x <s c ? X : C-1 --> X = sext x; X >s C-1 ? C-1 : X
          // X = sext x; x >u c ? X : C+1 --> X = sext x; X <u C+1 ? C+1 : X
          // X = sext x; x <u c ? X : C-1 --> X = sext x; X >u C-1 ? C-1 : X
          if (match(TrueVal, m_SExt(m_Specific(CmpLHS))) &&
                sextRHS == FalseVal) {
            CmpLHS = TrueVal;
            AdjustedRHS = sextRHS;
          } else if (match(FalseVal, m_SExt(m_Specific(CmpLHS))) &&
                     sextRHS == TrueVal) {
            CmpLHS = FalseVal;
            AdjustedRHS = sextRHS;
          } else if (ICI->isUnsigned()) {
            Constant *zextRHS = ConstantExpr::getZExt(AdjustedRHS, SelectTy);
            // X = zext x; x >u c ? X : C+1 --> X = zext x; X <u C+1 ? C+1 : X
            // X = zext x; x <u c ? X : C-1 --> X = zext x; X >u C-1 ? C-1 : X
            // zext + signed compare cannot be changed:
            //    0xff <s 0x00, but 0x00ff >s 0x0000
            if (match(TrueVal, m_ZExt(m_Specific(CmpLHS))) &&
                zextRHS == FalseVal) {
              CmpLHS = TrueVal;
              AdjustedRHS = zextRHS;
            } else if (match(FalseVal, m_ZExt(m_Specific(CmpLHS))) &&
                       zextRHS == TrueVal) {
              CmpLHS = FalseVal;
              AdjustedRHS = zextRHS;
            } else
              break;
          } else
            break;
        } else
          break;

        Pred = ICmpInst::getSwappedPredicate(Pred);
        CmpRHS = AdjustedRHS;
        std::swap(FalseVal, TrueVal);
        ICI->setPredicate(Pred);
        ICI->setOperand(0, CmpLHS);
        ICI->setOperand(1, CmpRHS);
        SI.setOperand(1, TrueVal);
        SI.setOperand(2, FalseVal);

        // Move ICI instruction right before the select instruction. Otherwise
        // the sext/zext value may be defined after the ICI instruction uses it.
        ICI->moveBefore(&SI);

        Changed = true;
        break;
      }
      }
    }

  // Transform (X >s -1) ? C1 : C2 --> ((X >>s 31) & (C2 - C1)) + C1
  // and       (X <s  0) ? C2 : C1 --> ((X >>s 31) & (C2 - C1)) + C1
  // FIXME: Type and constness constraints could be lifted, but we have to
  //        watch code size carefully. We should consider xor instead of
  //        sub/add when we decide to do that.
  if (IntegerType *Ty = dyn_cast<IntegerType>(CmpLHS->getType())) {
    if (TrueVal->getType() == Ty) {
      if (ConstantInt *Cmp = dyn_cast<ConstantInt>(CmpRHS)) {
        ConstantInt *C1 = NULL, *C2 = NULL;
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
            return ReplaceInstUsesWith(SI, Builder->CreateOr(AShr, C1));

          Value *And = Builder->CreateAnd(AShr, C2->getValue()-C1->getValue());
          return ReplaceInstUsesWith(SI, Builder->CreateAdd(And, C1));
        }
      }
    }
  }

  // If we have an equality comparison then we know the value in one of the
  // arms of the select. See if substituting this value into the arm and
  // simplifying the result yields the same value as the other arm.
  if (Pred == ICmpInst::ICMP_EQ) {
    if (SimplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, TD, TLI) == TrueVal ||
        SimplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, TD, TLI) == TrueVal)
      return ReplaceInstUsesWith(SI, FalseVal);
    if (SimplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, TD, TLI) == FalseVal ||
        SimplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, TD, TLI) == FalseVal)
      return ReplaceInstUsesWith(SI, FalseVal);
  } else if (Pred == ICmpInst::ICMP_NE) {
    if (SimplifyWithOpReplaced(TrueVal, CmpLHS, CmpRHS, TD, TLI) == FalseVal ||
        SimplifyWithOpReplaced(TrueVal, CmpRHS, CmpLHS, TD, TLI) == FalseVal)
      return ReplaceInstUsesWith(SI, TrueVal);
    if (SimplifyWithOpReplaced(FalseVal, CmpLHS, CmpRHS, TD, TLI) == TrueVal ||
        SimplifyWithOpReplaced(FalseVal, CmpRHS, CmpLHS, TD, TLI) == TrueVal)
      return ReplaceInstUsesWith(SI, TrueVal);
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

  return Changed ? &SI : 0;
}


/// CanSelectOperandBeMappingIntoPredBlock - SI is a select whose condition is a
/// PHI node (but the two may be in different blocks).  See if the true/false
/// values (V) are live in all of the predecessor blocks of the PHI.  For
/// example, cases like this cannot be mapped:
///
///   X = phi [ C1, BB1], [C2, BB2]
///   Y = add
///   Z = select X, Y, 0
///
/// because Y is not live in BB1/BB2.
///
static bool CanSelectOperandBeMappingIntoPredBlock(const Value *V,
                                                   const SelectInst &SI) {
  // If the value is a non-instruction value like a constant or argument, it
  // can always be mapped.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return true;

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

/// FoldSPFofSPF - We have an SPF (e.g. a min or max) of an SPF of the form:
///   SPF2(SPF1(A, B), C)
Instruction *InstCombiner::FoldSPFofSPF(Instruction *Inner,
                                        SelectPatternFlavor SPF1,
                                        Value *A, Value *B,
                                        Instruction &Outer,
                                        SelectPatternFlavor SPF2, Value *C) {
  if (C == A || C == B) {
    // MAX(MAX(A, B), B) -> MAX(A, B)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    if (SPF1 == SPF2)
      return ReplaceInstUsesWith(Outer, Inner);

    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    if ((SPF1 == SPF_SMIN && SPF2 == SPF_SMAX) ||
        (SPF1 == SPF_SMAX && SPF2 == SPF_SMIN) ||
        (SPF1 == SPF_UMIN && SPF2 == SPF_UMAX) ||
        (SPF1 == SPF_UMAX && SPF2 == SPF_UMIN))
      return ReplaceInstUsesWith(Outer, C);
  }

  // TODO: MIN(MIN(A, 23), 97)
  return 0;
}


/// foldSelectICmpAnd - If one of the constants is zero (we know they can't
/// both be) and we have an icmp instruction with zero, and we have an 'and'
/// with the non-constant value and a power of two we can turn the select
/// into a shift on the result of the 'and'.
static Value *foldSelectICmpAnd(const SelectInst &SI, ConstantInt *TrueVal,
                                ConstantInt *FalseVal,
                                InstCombiner::BuilderTy *Builder) {
  const ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition());
  if (!IC || !IC->isEquality())
    return 0;

  if (!match(IC->getOperand(1), m_Zero()))
    return 0;

  ConstantInt *AndRHS;
  Value *LHS = IC->getOperand(0);
  if (LHS->getType() != SI.getType() ||
      !match(LHS, m_And(m_Value(), m_ConstantInt(AndRHS))))
    return 0;

  // If both select arms are non-zero see if we have a select of the form
  // 'x ? 2^n + C : C'. Then we can offset both arms by C, use the logic
  // for 'x ? 2^n : 0' and fix the thing up at the end.
  ConstantInt *Offset = 0;
  if (!TrueVal->isZero() && !FalseVal->isZero()) {
    if ((TrueVal->getValue() - FalseVal->getValue()).isPowerOf2())
      Offset = FalseVal;
    else if ((FalseVal->getValue() - TrueVal->getValue()).isPowerOf2())
      Offset = TrueVal;
    else
      return 0;

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
    return 0;

  // Determine which shift is needed to transform result of the 'and' into the
  // desired result.
  ConstantInt *ValC = !TrueVal->isZero() ? TrueVal : FalseVal;
  unsigned ValZeros = ValC->getValue().logBase2();
  unsigned AndZeros = AndRHS->getValue().logBase2();

  Value *V = LHS;
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

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  if (Value *V = SimplifySelectInst(CondVal, TrueVal, FalseVal, TD))
    return ReplaceInstUsesWith(SI, V);

  if (SI.getType()->isIntegerTy(1)) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(TrueVal)) {
      if (C->getZExtValue()) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::CreateOr(CondVal, FalseVal);
      }
      // Change: A = select B, false, C --> A = and !B, C
      Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
      return BinaryOperator::CreateAnd(NotCond, FalseVal);
    } else if (ConstantInt *C = dyn_cast<ConstantInt>(FalseVal)) {
      if (C->getZExtValue() == false) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::CreateAnd(CondVal, TrueVal);
      }
      // Change: A = select B, C, true --> A = or !B, C
      Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
      return BinaryOperator::CreateOr(NotCond, TrueVal);
    }

    // select a, b, a  -> a&b
    // select a, a, b  -> a|b
    if (CondVal == TrueVal)
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    else if (CondVal == FalseVal)
      return BinaryOperator::CreateAnd(CondVal, TrueVal);

    // select a, ~a, b -> (~a)&b
    // select a, b, ~a -> (~a)|b
    if (match(TrueVal, m_Not(m_Specific(CondVal))))
      return BinaryOperator::CreateAnd(TrueVal, FalseVal);
    else if (match(FalseVal, m_Not(m_Specific(CondVal))))
      return BinaryOperator::CreateOr(TrueVal, FalseVal);
  }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> zext C to int
      if (FalseValC->isZero() && TrueValC->getValue() == 1)
        return new ZExtInst(CondVal, SI.getType());

      // select C, -1, 0 -> sext C to int
      if (FalseValC->isZero() && TrueValC->isAllOnesValue())
        return new SExtInst(CondVal, SI.getType());

      // select C, 0, 1 -> zext !C to int
      if (TrueValC->isZero() && FalseValC->getValue() == 1) {
        Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
        return new ZExtInst(NotCond, SI.getType());
      }

      // select C, 0, -1 -> sext !C to int
      if (TrueValC->isZero() && FalseValC->isAllOnesValue()) {
        Value *NotCond = Builder->CreateNot(CondVal, "not."+CondVal->getName());
        return new SExtInst(NotCond, SI.getType());
      }

      if (Value *V = foldSelectICmpAnd(SI, TrueValC, FalseValC, Builder))
        return ReplaceInstUsesWith(SI, V);
    }

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
        return ReplaceInstUsesWith(SI, FalseVal);
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
        return ReplaceInstUsesWith(SI, TrueVal);
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
          return ReplaceInstUsesWith(SI, FalseVal);
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
          return ReplaceInstUsesWith(SI, TrueVal);
      }
      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
    // NOTE: if we wanted to, this is where to detect ABS
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = visitSelectInstWithICmp(SI, ICI))
      return Result;

  if (Instruction *TI = dyn_cast<Instruction>(TrueVal))
    if (Instruction *FI = dyn_cast<Instruction>(FalseVal))
      if (TI->hasOneUse() && FI->hasOneUse()) {
        Instruction *AddOp = 0, *SubOp = 0;

        // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
        if (TI->getOpcode() == FI->getOpcode())
          if (Instruction *IV = FoldSelectOpOp(SI, TI, FI))
            return IV;

        // Turn select C, (X+Y), (X-Y) --> (X+(select C, Y, (-Y))).  This is
        // even legal for FP.
        if ((TI->getOpcode() == Instruction::Sub &&
             FI->getOpcode() == Instruction::Add) ||
            (TI->getOpcode() == Instruction::FSub &&
             FI->getOpcode() == Instruction::FAdd)) {
          AddOp = FI; SubOp = TI;
        } else if ((FI->getOpcode() == Instruction::Sub &&
                    TI->getOpcode() == Instruction::Add) ||
                   (FI->getOpcode() == Instruction::FSub &&
                    TI->getOpcode() == Instruction::FAdd)) {
          AddOp = TI; SubOp = FI;
        }

        if (AddOp) {
          Value *OtherAddOp = 0;
          if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
            OtherAddOp = AddOp->getOperand(1);
          } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
            OtherAddOp = AddOp->getOperand(0);
          }

          if (OtherAddOp) {
            // So at this point we know we have (Y -> OtherAddOp):
            //        select C, (add X, Y), (sub X, Z)
            Value *NegVal;  // Compute -Z
            if (SI.getType()->isFPOrFPVectorTy()) {
              NegVal = Builder->CreateFNeg(SubOp->getOperand(1));
            } else {
              NegVal = Builder->CreateNeg(SubOp->getOperand(1));
            }

            Value *NewTrueOp = OtherAddOp;
            Value *NewFalseOp = NegVal;
            if (AddOp != TI)
              std::swap(NewTrueOp, NewFalseOp);
            Value *NewSel = 
              Builder->CreateSelect(CondVal, NewTrueOp,
                                    NewFalseOp, SI.getName() + ".p");

            if (SI.getType()->isFPOrFPVectorTy())
              return BinaryOperator::CreateFAdd(SubOp->getOperand(0), NewSel);
            else
              return BinaryOperator::CreateAdd(SubOp->getOperand(0), NewSel);
          }
        }
      }

  // See if we can fold the select into one of our operands.
  if (SI.getType()->isIntegerTy()) {
    if (Instruction *FoldI = FoldSelectIntoOp(SI, TrueVal, FalseVal))
      return FoldI;

    // MAX(MAX(a, b), a) -> MAX(a, b)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    Value *LHS, *RHS, *LHS2, *RHS2;
    if (SelectPatternFlavor SPF = MatchSelectPattern(&SI, LHS, RHS)) {
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(LHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(LHS),SPF2,LHS2,RHS2, 
                                          SI, SPF, RHS))
          return R;
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(RHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(RHS),SPF2,LHS2,RHS2,
                                          SI, SPF, LHS))
          return R;
    }

    // TODO.
    // ABS(-X) -> ABS(X)
    // ABS(ABS(X)) -> ABS(X)
  }

  // See if we can fold the select into a phi node if the condition is a select.
  if (isa<PHINode>(SI.getCondition()))
    // The true/false values have to be live in the PHI predecessor's blocks.
    if (CanSelectOperandBeMappingIntoPredBlock(TrueVal, SI) &&
        CanSelectOperandBeMappingIntoPredBlock(FalseVal, SI))
      if (Instruction *NV = FoldOpIntoPhi(SI))
        return NV;

  if (SelectInst *TrueSI = dyn_cast<SelectInst>(TrueVal)) {
    if (TrueSI->getCondition() == CondVal) {
      if (SI.getTrueValue() == TrueSI->getTrueValue())
        return 0;
      SI.setOperand(1, TrueSI->getTrueValue());
      return &SI;
    }
  }
  if (SelectInst *FalseSI = dyn_cast<SelectInst>(FalseVal)) {
    if (FalseSI->getCondition() == CondVal) {
      if (SI.getFalseValue() == FalseSI->getFalseValue())
        return 0;
      SI.setOperand(2, FalseSI->getFalseValue());
      return &SI;
    }
  }

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  if (VectorType *VecTy = dyn_cast<VectorType>(SI.getType())) {
    unsigned VWidth = VecTy->getNumElements();
    APInt UndefElts(VWidth, 0);
    APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
    if (Value *V = SimplifyDemandedVectorElts(&SI, AllOnesEltMask, UndefElts)) {
      if (V != &SI)
        return ReplaceInstUsesWith(SI, V);
      return &SI;
    }

    if (ConstantVector *CV = dyn_cast<ConstantVector>(CondVal)) {
      // Form a shufflevector instruction.
      SmallVector<Constant *, 8> Mask(VWidth);
      Type *Int32Ty = Type::getInt32Ty(CV->getContext());
      for (unsigned i = 0; i != VWidth; ++i) {
        Constant *Elem = cast<Constant>(CV->getOperand(i));
        if (ConstantInt *E = dyn_cast<ConstantInt>(Elem))
          Mask[i] = ConstantInt::get(Int32Ty, i + (E->isZero() ? VWidth : 0));
        else if (isa<UndefValue>(Elem))
          Mask[i] = UndefValue::get(Int32Ty);
        else
          return 0;
      }
      Constant *MaskVal = ConstantVector::get(Mask);
      Value *V = Builder->CreateShuffleVector(TrueVal, FalseVal, MaskVal);
      return ReplaceInstUsesWith(SI, V);
    }

    if (isa<ConstantAggregateZero>(CondVal)) {
      return ReplaceInstUsesWith(SI, FalseVal);
    }
  }

  return 0;
}
