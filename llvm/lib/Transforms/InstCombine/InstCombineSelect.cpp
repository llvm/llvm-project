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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CmpInstAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
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
#include <cassert>
#include <utility>

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

static Value *generateMinMaxSelectPattern(InstCombiner::BuilderTy &Builder,
                                          SelectPatternFlavor SPF, Value *A,
                                          Value *B) {
  CmpInst::Predicate Pred = getCmpPredicateForMinMax(SPF);
  assert(CmpInst::isIntPredicate(Pred));
  return Builder.CreateSelect(Builder.CreateICmp(Pred, A, B), A, B);
}

/// If one of the constants is zero (we know they can't both be) and we have an
/// icmp instruction with zero, and we have an 'and' with the non-constant value
/// and a power of two we can turn the select into a shift on the result of the
/// 'and'.
/// This folds:
///  select (icmp eq (and X, C1)), C2, C3
///    iff C1 is a power 2 and the difference between C2 and C3 is a power of 2.
/// To something like:
///  (shr (and (X, C1)), (log2(C1) - log2(C2-C3))) + C3
/// Or:
///  (shl (and (X, C1)), (log2(C2-C3) - log2(C1))) + C3
/// With some variations depending if C3 is larger than C2, or the shift
/// isn't needed, or the bit widths don't match.
static Value *foldSelectICmpAnd(Type *SelType, const ICmpInst *IC,
                                APInt TrueVal, APInt FalseVal,
                                InstCombiner::BuilderTy &Builder) {
  assert(SelType->isIntOrIntVectorTy() && "Not an integer select?");

  // If this is a vector select, we need a vector compare.
  if (SelType->isVectorTy() != IC->getType()->isVectorTy())
    return nullptr;

  Value *V;
  APInt AndMask;
  bool CreateAnd = false;
  ICmpInst::Predicate Pred = IC->getPredicate();
  if (ICmpInst::isEquality(Pred)) {
    if (!match(IC->getOperand(1), m_Zero()))
      return nullptr;

    V = IC->getOperand(0);

    const APInt *AndRHS;
    if (!match(V, m_And(m_Value(), m_Power2(AndRHS))))
      return nullptr;

    AndMask = *AndRHS;
  } else if (decomposeBitTestICmp(IC->getOperand(0), IC->getOperand(1),
                                  Pred, V, AndMask)) {
    assert(ICmpInst::isEquality(Pred) && "Not equality test?");

    if (!AndMask.isPowerOf2())
      return nullptr;

    CreateAnd = true;
  } else {
    return nullptr;
  }

  // If both select arms are non-zero see if we have a select of the form
  // 'x ? 2^n + C : C'. Then we can offset both arms by C, use the logic
  // for 'x ? 2^n : 0' and fix the thing up at the end.
  APInt Offset(TrueVal.getBitWidth(), 0);
  if (!TrueVal.isNullValue() && !FalseVal.isNullValue()) {
    if ((TrueVal - FalseVal).isPowerOf2())
      Offset = FalseVal;
    else if ((FalseVal - TrueVal).isPowerOf2())
      Offset = TrueVal;
    else
      return nullptr;

    // Adjust TrueVal and FalseVal to the offset.
    TrueVal -= Offset;
    FalseVal -= Offset;
  }

  // Make sure one of the select arms is a power of 2.
  if (!TrueVal.isPowerOf2() && !FalseVal.isPowerOf2())
    return nullptr;

  // Determine which shift is needed to transform result of the 'and' into the
  // desired result.
  const APInt &ValC = !TrueVal.isNullValue() ? TrueVal : FalseVal;
  unsigned ValZeros = ValC.logBase2();
  unsigned AndZeros = AndMask.logBase2();

  if (CreateAnd) {
    // Insert the AND instruction on the input to the truncate.
    V = Builder.CreateAnd(V, ConstantInt::get(V->getType(), AndMask));
  }

  // If types don't match we can still convert the select by introducing a zext
  // or a trunc of the 'and'.
  if (ValZeros > AndZeros) {
    V = Builder.CreateZExtOrTrunc(V, SelType);
    V = Builder.CreateShl(V, ValZeros - AndZeros);
  } else if (ValZeros < AndZeros) {
    V = Builder.CreateLShr(V, AndZeros - ValZeros);
    V = Builder.CreateZExtOrTrunc(V, SelType);
  } else
    V = Builder.CreateZExtOrTrunc(V, SelType);

  // Okay, now we know that everything is set up, we just don't know whether we
  // have a icmp_ne or icmp_eq and whether the true or false val is the zero.
  bool ShouldNotVal = !TrueVal.isNullValue();
  ShouldNotVal ^= Pred == ICmpInst::ICMP_NE;
  if (ShouldNotVal)
    V = Builder.CreateXor(V, ValC);

  // Apply an offset if needed.
  if (!Offset.isNullValue())
    V = Builder.CreateAdd(V, ConstantInt::get(V->getType(), Offset));
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

/// For the same transformation as the previous function, return the identity
/// constant that goes into the select.
static APInt getSelectFoldableConstant(BinaryOperator *I) {
  switch (I->getOpcode()) {
  default: llvm_unreachable("This cannot happen!");
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    return APInt::getNullValue(I->getType()->getScalarSizeInBits());
  case Instruction::And:
    return APInt::getAllOnesValue(I->getType()->getScalarSizeInBits());
  case Instruction::Mul:
    return APInt(I->getType()->getScalarSizeInBits(), 1);
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
        Builder.CreateSelect(SI.getCondition(), TI->getOperand(0),
                             FI->getOperand(0), SI.getName() + ".v", &SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI,
                            TI->getType());
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

  // If we reach here, they do have operations in common.
  Value *NewSI = Builder.CreateSelect(SI.getCondition(), OtherOpT, OtherOpF,
                                      SI.getName() + ".v", &SI);
  Value *Op0 = MatchIsOpZero ? MatchOp : NewSI;
  Value *Op1 = MatchIsOpZero ? NewSI : MatchOp;
  if (auto *BO = dyn_cast<BinaryOperator>(TI)) {
    return BinaryOperator::Create(BO->getOpcode(), Op0, Op1);
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
Instruction *InstCombiner::foldSelectIntoOp(SelectInst &SI, Value *TrueVal,
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
          APInt CI = getSelectFoldableConstant(TVI);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          const APInt *OOpC;
          bool OOpIsAPInt = match(OOp, m_APInt(OOpC));
          if (!isa<Constant>(OOp) || (OOpIsAPInt && isSelect01(CI, *OOpC))) {
            Value *C = ConstantInt::get(OOp->getType(), CI);
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
          APInt CI = getSelectFoldableConstant(FVI);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0, 1 and -1.
          const APInt *OOpC;
          bool OOpIsAPInt = match(OOp, m_APInt(OOpC));
          if (!isa<Constant>(OOp) || (OOpIsAPInt && isSelect01(CI, *OOpC))) {
            Value *C = ConstantInt::get(OOp->getType(), CI);
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
    NewI->setArgOperand(1, ConstantInt::getFalse(NewI->getContext()));
    Builder.Insert(NewI);
    return Builder.CreateZExtOrTrunc(NewI, ValueOnZero->getType());
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
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  if (Instruction *NewSel = canonicalizeMinMaxWithConstant(SI, *ICI, Builder))
    return NewSel;

  bool Changed = adjustMinMax(SI, *ICI);

  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);

  // Transform (X >s -1) ? C1 : C2 --> ((X >>s 31) & (C2 - C1)) + C1
  // and       (X <s  0) ? C2 : C1 --> ((X >>s 31) & (C2 - C1)) + C1
  // FIXME: Type and constness constraints could be lifted, but we have to
  //        watch code size carefully. We should consider xor instead of
  //        sub/add when we decide to do that.
  // TODO: Merge this with foldSelectICmpAnd somehow.
  if (CmpLHS->getType()->isIntOrIntVectorTy() &&
      CmpLHS->getType() == TrueVal->getType()) {
    const APInt *C1, *C2;
    if (match(TrueVal, m_APInt(C1)) && match(FalseVal, m_APInt(C2))) {
      ICmpInst::Predicate Pred = ICI->getPredicate();
      Value *X;
      APInt Mask;
      if (decomposeBitTestICmp(CmpLHS, CmpRHS, Pred, X, Mask, false)) {
        if (Mask.isSignMask()) {
          assert(X == CmpLHS && "Expected to use the compare input directly");
          assert(ICmpInst::isEquality(Pred) && "Expected equality predicate");

          if (Pred == ICmpInst::ICMP_NE)
            std::swap(C1, C2);

          // This shift results in either -1 or 0.
          Value *AShr = Builder.CreateAShr(X, Mask.getBitWidth() - 1);

          // Check if we can express the operation with a single or.
          if (C2->isAllOnesValue())
            return replaceInstUsesWith(SI, Builder.CreateOr(AShr, *C1));

          Value *And = Builder.CreateAnd(AShr, *C2 - *C1);
          return replaceInstUsesWith(SI, Builder.CreateAdd(And,
                                        ConstantInt::get(And->getType(), *C1)));
        }
      }
    }
  }

  {
    const APInt *TrueValC, *FalseValC;
    if (match(TrueVal, m_APInt(TrueValC)) &&
        match(FalseVal, m_APInt(FalseValC)))
      if (Value *V = foldSelectICmpAnd(SI.getType(), ICI, *TrueValC,
                                       *FalseValC, Builder))
        return replaceInstUsesWith(SI, V);
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

  if (Value *V = foldSelectICmpAndOr(ICI, TrueVal, FalseVal, Builder))
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
      NotA = Builder.CreateNot(A);
    if (!NotB)
      NotB = Builder.CreateNot(B);
    if (!NotC)
      NotC = Builder.CreateNot(C);

    Value *NewInner = generateMinMaxSelectPattern(
        Builder, getInverseMinMaxSelectPattern(SPF1), NotA, NotB);
    Value *NewOuter = Builder.CreateNot(generateMinMaxSelectPattern(
        Builder, getInverseMinMaxSelectPattern(SPF2), NewInner, NotC));
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
  if (!SmallType->isIntOrIntVectorTy(1))
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
      // Undef in a select condition (choose one of the operands) does not mean
      // the same thing as undef in a shuffle mask (any value is acceptable), so
      // give up.
      return nullptr;
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
static Instruction *foldSelectCmpXchg(SelectInst &SI) {
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
    if (X == CmpXchg && X->getCompareOperand() == SI.getFalseValue()) {
      SI.setTrueValue(SI.getFalseValue());
      return &SI;
    }

  // Check the false value case: The false value of the select is the returned
  // value of the same cmpxchg used by the condition, and the true value is the
  // cmpxchg instruction's compare operand.
  if (auto *X = isExtractFromCmpXchg(SI.getFalseValue(), 0))
    if (X == CmpXchg && X->getCompareOperand() == SI.getTrueValue()) {
      SI.setTrueValue(SI.getFalseValue());
      return &SI;
    }

  return nullptr;
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
  if (LHS->getNumUses() <= 2 && RHS->getNumUses() > 2) {
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
  } else if (RHS->getNumUses() <= 2) {
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

  CmpInst::Predicate P = getCmpPredicateForMinMax(SPF);
  Value *CmpABC = Builder.CreateICmp(P, MinMaxOp, ThirdOp);
  return SelectInst::Create(CmpABC, MinMaxOp, ThirdOp);
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();
  Type *SelType = SI.getType();

  // FIXME: Remove this workaround when freeze related patches are done.
  // For select with undef operand which feeds into an equality comparison,
  // don't simplify it so loop unswitch can know the equality comparison
  // may have an undef operand. This is a workaround for PR31652 caused by
  // descrepancy about branch on undef between LoopUnswitch and GVN.
  if (isa<UndefValue>(TrueVal) || isa<UndefValue>(FalseVal)) {
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

  // Canonicalize a one-use integer compare with a non-canonical predicate by
  // inverting the predicate and swapping the select operands. This matches a
  // compare canonicalization for conditional branches.
  // TODO: Should we do the same for FP compares?
  CmpInst::Predicate Pred;
  if (match(CondVal, m_OneUse(m_ICmp(Pred, m_Value(), m_Value()))) &&
      !isCanonicalPredicate(Pred)) {
    // Swap true/false values and condition.
    CmpInst *Cond = cast<CmpInst>(CondVal);
    Cond->setPredicate(CmpInst::getInversePredicate(Pred));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    SI.swapProfMetadata();
    Worklist.Add(Cond);
    return &SI;
  }

  if (SelType->isIntOrIntVectorTy(1) &&
      TrueVal->getType() == CondVal->getType()) {
    if (match(TrueVal, m_One())) {
      // Change: A = select B, true, C --> A = or B, C
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    }
    if (match(TrueVal, m_Zero())) {
      // Change: A = select B, false, C --> A = and !B, C
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
      return BinaryOperator::CreateAnd(NotCond, FalseVal);
    }
    if (match(FalseVal, m_Zero())) {
      // Change: A = select B, C, false --> A = and B, C
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
    }
    if (match(FalseVal, m_One())) {
      // Change: A = select B, C, true --> A = or !B, C
      Value *NotCond = Builder.CreateNot(CondVal, "not." + CondVal->getName());
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
  if (SelType->isIntOrIntVectorTy() &&
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
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        Builder.setFastMathFlags(FCI->getFastMathFlags());
        Value *NewCond = Builder.CreateFCmp(InvPred, TrueVal, FalseVal,
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
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        Builder.setFastMathFlags(FCI->getFastMathFlags());
        Value *NewCond = Builder.CreateFCmp(InvPred, FalseVal, TrueVal,
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

  if (Instruction *Add = foldAddSubSelect(SI, Builder))
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
        CmpInst::Predicate Pred = getCmpPredicateForMinMax(SPF, SPR.Ordered);

        Value *Cmp;
        if (CmpInst::isIntPredicate(Pred)) {
          Cmp = Builder.CreateICmp(Pred, LHS, RHS);
        } else {
          IRBuilder<>::FastMathFlagGuard FMFG(Builder);
          auto FMF = cast<FPMathOperator>(SI.getCondition())->getFastMathFlags();
          Builder.setFastMathFlags(FMF);
          Cmp = Builder.CreateFCmp(Pred, LHS, RHS);
        }

        Value *NewSI = Builder.CreateSelect(Cmp, LHS, RHS, SI.getName(), &SI);
        if (!IsCastNeeded)
          return replaceInstUsesWith(SI, NewSI);

        Value *NewCast = Builder.CreateCast(CastOp, NewSI, SelType);
        return replaceInstUsesWith(SI, NewCast);
      }

      // MAX(~a, ~b) -> ~MIN(a, b)
      // MIN(~a, ~b) -> ~MAX(a, b)
      Value *A, *B;
      if (match(LHS, m_Not(m_Value(A))) && match(RHS, m_Not(m_Value(B))) &&
          (LHS->getNumUses() <= 2 || RHS->getNumUses() <= 2)) {
        CmpInst::Predicate InvertedPred =
            getCmpPredicateForMinMax(getInverseMinMaxSelectPattern(SPF));
        Value *InvertedCmp = Builder.CreateICmp(InvertedPred, A, B);
        Value *NewSel = Builder.CreateSelect(InvertedCmp, A, B);
        return BinaryOperator::CreateNot(NewSel);
      }

      if (Instruction *I = factorizeMinMaxTree(SPF, LHS, RHS, Builder))
        return I;
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

    // TODO.
    // ABS(-X) -> ABS(X)
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
        SI.setOperand(1, TrueSI->getTrueValue());
        return &SI;
      }
      // select(C0, select(C1, a, b), b) -> select(C0&C1, a, b)
      // We choose this as normal form to enable folding on the And and shortening
      // paths for the values (this helps GetUnderlyingObjects() for example).
      if (TrueSI->getFalseValue() == FalseVal && TrueSI->hasOneUse()) {
        Value *And = Builder.CreateAnd(CondVal, TrueSI->getCondition());
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
        Value *Or = Builder.CreateOr(CondVal, FalseSI->getCondition());
        SI.setOperand(0, Or);
        SI.setOperand(2, FalseSI->getFalseValue());
        return &SI;
      }
    }
  }

  // Try to simplify a binop sandwiched between 2 selects with the same
  // condition.
  // select(C, binop(select(C, X, Y), W), Z) -> select(C, binop(X, W), Z)
  BinaryOperator *TrueBO;
  if (match(TrueVal, m_OneUse(m_BinOp(TrueBO)))) {
    if (auto *TrueBOSI = dyn_cast<SelectInst>(TrueBO->getOperand(0))) {
      if (TrueBOSI->getCondition() == CondVal) {
        TrueBO->setOperand(0, TrueBOSI->getTrueValue());
        Worklist.Add(TrueBO);
        return &SI;
      }
    }
    if (auto *TrueBOSI = dyn_cast<SelectInst>(TrueBO->getOperand(1))) {
      if (TrueBOSI->getCondition() == CondVal) {
        TrueBO->setOperand(1, TrueBOSI->getTrueValue());
        Worklist.Add(TrueBO);
        return &SI;
      }
    }
  }

  // select(C, Z, binop(select(C, X, Y), W)) -> select(C, Z, binop(Y, W))
  BinaryOperator *FalseBO;
  if (match(FalseVal, m_OneUse(m_BinOp(FalseBO)))) {
    if (auto *FalseBOSI = dyn_cast<SelectInst>(FalseBO->getOperand(0))) {
      if (FalseBOSI->getCondition() == CondVal) {
        FalseBO->setOperand(0, FalseBOSI->getFalseValue());
        Worklist.Add(FalseBO);
        return &SI;
      }
    }
    if (auto *FalseBOSI = dyn_cast<SelectInst>(FalseBO->getOperand(1))) {
      if (FalseBOSI->getCondition() == CondVal) {
        FalseBO->setOperand(1, FalseBOSI->getFalseValue());
        Worklist.Add(FalseBO);
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
  }

  // See if we can determine the result of this select based on a dominating
  // condition.
  BasicBlock *Parent = SI.getParent();
  if (BasicBlock *Dom = Parent->getSinglePredecessor()) {
    auto *PBI = dyn_cast_or_null<BranchInst>(Dom->getTerminator());
    if (PBI && PBI->isConditional() &&
        PBI->getSuccessor(0) != PBI->getSuccessor(1) &&
        (PBI->getSuccessor(0) == Parent || PBI->getSuccessor(1) == Parent)) {
      bool CondIsTrue = PBI->getSuccessor(0) == Parent;
      Optional<bool> Implication = isImpliedCondition(
          PBI->getCondition(), SI.getCondition(), DL, CondIsTrue);
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
  if (Instruction *Select = foldSelectCmpXchg(SI))
    return Select;

  return nullptr;
}
