//===- InstCombineMulDivRem.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for mul, fmul, sdiv, udiv, fdiv,
// srem, urem, frem.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/InstCombine/InstCombineWorklist.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

/// The specific integer value is used in a context where it is known to be
/// non-zero.  If this allows us to simplify the computation, do so and return
/// the new operand, otherwise return null.
static Value *simplifyValueKnownNonZero(Value *V, InstCombiner &IC,
                                        Instruction &CxtI) {
  // If V has multiple uses, then we would have to do more analysis to determine
  // if this is safe.  For example, the use could be in dynamically unreached
  // code.
  if (!V->hasOneUse()) return nullptr;

  bool MadeChange = false;

  // ((1 << A) >>u B) --> (1 << (A-B))
  // Because V cannot be zero, we know that B is less than A.
  Value *A = nullptr, *B = nullptr, *One = nullptr;
  if (match(V, m_LShr(m_OneUse(m_Shl(m_Value(One), m_Value(A))), m_Value(B))) &&
      match(One, m_One())) {
    A = IC.Builder.CreateSub(A, B);
    return IC.Builder.CreateShl(One, A);
  }

  // (PowerOfTwo >>u B) --> isExact since shifting out the result would make it
  // inexact.  Similarly for <<.
  BinaryOperator *I = dyn_cast<BinaryOperator>(V);
  if (I && I->isLogicalShift() &&
      IC.isKnownToBeAPowerOfTwo(I->getOperand(0), false, 0, &CxtI)) {
    // We know that this is an exact/nuw shift and that the input is a
    // non-zero context as well.
    if (Value *V2 = simplifyValueKnownNonZero(I->getOperand(0), IC, CxtI)) {
      I->setOperand(0, V2);
      MadeChange = true;
    }

    if (I->getOpcode() == Instruction::LShr && !I->isExact()) {
      I->setIsExact();
      MadeChange = true;
    }

    if (I->getOpcode() == Instruction::Shl && !I->hasNoUnsignedWrap()) {
      I->setHasNoUnsignedWrap();
      MadeChange = true;
    }
  }

  // TODO: Lots more we could do here:
  //    If V is a phi node, we can call this on each of its operands.
  //    "select cond, X, 0" can simplify to "X".

  return MadeChange ? V : nullptr;
}

/// \brief A helper routine of InstCombiner::visitMul().
///
/// If C is a scalar/vector of known powers of 2, then this function returns
/// a new scalar/vector obtained from logBase2 of C.
/// Return a null pointer otherwise.
static Constant *getLogBase2(Type *Ty, Constant *C) {
  const APInt *IVal;
  if (match(C, m_APInt(IVal)) && IVal->isPowerOf2())
    return ConstantInt::get(Ty, IVal->logBase2());

  if (!Ty->isVectorTy())
    return nullptr;

  SmallVector<Constant *, 4> Elts;
  for (unsigned I = 0, E = Ty->getVectorNumElements(); I != E; ++I) {
    Constant *Elt = C->getAggregateElement(I);
    if (!Elt)
      return nullptr;
    if (isa<UndefValue>(Elt)) {
      Elts.push_back(UndefValue::get(Ty->getScalarType()));
      continue;
    }
    if (!match(Elt, m_APInt(IVal)) || !IVal->isPowerOf2())
      return nullptr;
    Elts.push_back(ConstantInt::get(Ty->getScalarType(), IVal->logBase2()));
  }

  return ConstantVector::get(Elts);
}

/// \brief Return true if we can prove that:
///    (mul LHS, RHS)  === (mul nsw LHS, RHS)
bool InstCombiner::willNotOverflowSignedMul(const Value *LHS,
                                            const Value *RHS,
                                            const Instruction &CxtI) const {
  // Multiplying n * m significant bits yields a result of n + m significant
  // bits. If the total number of significant bits does not exceed the
  // result bit width (minus 1), there is no overflow.
  // This means if we have enough leading sign bits in the operands
  // we can guarantee that the result does not overflow.
  // Ref: "Hacker's Delight" by Henry Warren
  unsigned BitWidth = LHS->getType()->getScalarSizeInBits();

  // Note that underestimating the number of sign bits gives a more
  // conservative answer.
  unsigned SignBits =
      ComputeNumSignBits(LHS, 0, &CxtI) + ComputeNumSignBits(RHS, 0, &CxtI);

  // First handle the easy case: if we have enough sign bits there's
  // definitely no overflow.
  if (SignBits > BitWidth + 1)
    return true;

  // There are two ambiguous cases where there can be no overflow:
  //   SignBits == BitWidth + 1    and
  //   SignBits == BitWidth
  // The second case is difficult to check, therefore we only handle the
  // first case.
  if (SignBits == BitWidth + 1) {
    // It overflows only when both arguments are negative and the true
    // product is exactly the minimum negative number.
    // E.g. mul i16 with 17 sign bits: 0xff00 * 0xff80 = 0x8000
    // For simplicity we just check if at least one side is not negative.
    KnownBits LHSKnown = computeKnownBits(LHS, /*Depth=*/0, &CxtI);
    KnownBits RHSKnown = computeKnownBits(RHS, /*Depth=*/0, &CxtI);
    if (LHSKnown.isNonNegative() || RHSKnown.isNonNegative())
      return true;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyMulInst(Op0, Op1, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  // X * -1 == 0 - X
  if (match(Op1, m_AllOnes())) {
    BinaryOperator *BO = BinaryOperator::CreateNeg(Op0, I.getName());
    if (I.hasNoSignedWrap())
      BO->setHasNoSignedWrap();
    return BO;
  }

  // Also allow combining multiply instructions on vectors.
  {
    Value *NewOp;
    Constant *C1, *C2;
    const APInt *IVal;
    if (match(&I, m_Mul(m_Shl(m_Value(NewOp), m_Constant(C2)),
                        m_Constant(C1))) &&
        match(C1, m_APInt(IVal))) {
      // ((X << C2)*C1) == (X * (C1 << C2))
      Constant *Shl = ConstantExpr::getShl(C1, C2);
      BinaryOperator *Mul = cast<BinaryOperator>(I.getOperand(0));
      BinaryOperator *BO = BinaryOperator::CreateMul(NewOp, Shl);
      if (I.hasNoUnsignedWrap() && Mul->hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && Mul->hasNoSignedWrap() &&
          Shl->isNotMinSignedValue())
        BO->setHasNoSignedWrap();
      return BO;
    }

    if (match(&I, m_Mul(m_Value(NewOp), m_Constant(C1)))) {
      // Replace X*(2^C) with X << C, where C is either a scalar or a vector.
      if (Constant *NewCst = getLogBase2(NewOp->getType(), C1)) {
        unsigned Width = NewCst->getType()->getPrimitiveSizeInBits();
        BinaryOperator *Shl = BinaryOperator::CreateShl(NewOp, NewCst);

        if (I.hasNoUnsignedWrap())
          Shl->setHasNoUnsignedWrap();
        if (I.hasNoSignedWrap()) {
          const APInt *V;
          if (match(NewCst, m_APInt(V)) && *V != Width - 1)
            Shl->setHasNoSignedWrap();
        }

        return Shl;
      }
    }
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // (Y - X) * (-(2**n)) -> (X - Y) * (2**n), for positive nonzero n
    // (Y + const) * (-(2**n)) -> (-constY) * (2**n), for positive nonzero n
    // The "* (2**n)" thus becomes a potential shifting opportunity.
    {
      const APInt &   Val = CI->getValue();
      const APInt &PosVal = Val.abs();
      if (Val.isNegative() && PosVal.isPowerOf2()) {
        Value *X = nullptr, *Y = nullptr;
        if (Op0->hasOneUse()) {
          ConstantInt *C1;
          Value *Sub = nullptr;
          if (match(Op0, m_Sub(m_Value(Y), m_Value(X))))
            Sub = Builder.CreateSub(X, Y, "suba");
          else if (match(Op0, m_Add(m_Value(Y), m_ConstantInt(C1))))
            Sub = Builder.CreateSub(Builder.CreateNeg(C1), Y, "subc");
          if (Sub)
            return
              BinaryOperator::CreateMul(Sub,
                                        ConstantInt::get(Y->getType(), PosVal));
        }
      }
    }
  }

  if (Instruction *FoldedMul = foldBinOpIntoSelectOrPhi(I))
    return FoldedMul;

  // Simplify mul instructions with a constant RHS.
  if (isa<Constant>(Op1)) {
    // Canonicalize (X+C1)*CI -> X*CI+C1*CI.
    Value *X;
    Constant *C1;
    if (match(Op0, m_OneUse(m_Add(m_Value(X), m_Constant(C1))))) {
      Value *Mul = Builder.CreateMul(C1, Op1);
      // Only go forward with the transform if C1*CI simplifies to a tidier
      // constant.
      if (!match(Mul, m_Mul(m_Value(), m_Value())))
        return BinaryOperator::CreateAdd(Builder.CreateMul(X, Op1), Mul);
    }
  }

  // -X * C --> X * -C
  Value *X, *Y;
  Constant *Op1C;
  if (match(Op0, m_Neg(m_Value(X))) && match(Op1, m_Constant(Op1C)))
    return BinaryOperator::CreateMul(X, ConstantExpr::getNeg(Op1C));

  // -X * -Y --> X * Y
  if (match(Op0, m_Neg(m_Value(X))) && match(Op1, m_Neg(m_Value(Y)))) {
    auto *NewMul = BinaryOperator::CreateMul(X, Y);
    if (I.hasNoSignedWrap() &&
        cast<OverflowingBinaryOperator>(Op0)->hasNoSignedWrap() &&
        cast<OverflowingBinaryOperator>(Op1)->hasNoSignedWrap())
      NewMul->setHasNoSignedWrap();
    return NewMul;
  }

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Y = Op1;
    BinaryOperator *Div = dyn_cast<BinaryOperator>(Op0);
    if (!Div || (Div->getOpcode() != Instruction::UDiv &&
                 Div->getOpcode() != Instruction::SDiv)) {
      Y = Op0;
      Div = dyn_cast<BinaryOperator>(Op1);
    }
    Value *Neg = dyn_castNegVal(Y);
    if (Div && Div->hasOneUse() &&
        (Div->getOperand(1) == Y || Div->getOperand(1) == Neg) &&
        (Div->getOpcode() == Instruction::UDiv ||
         Div->getOpcode() == Instruction::SDiv)) {
      Value *X = Div->getOperand(0), *DivOp1 = Div->getOperand(1);

      // If the division is exact, X % Y is zero, so we end up with X or -X.
      if (Div->isExact()) {
        if (DivOp1 == Y)
          return replaceInstUsesWith(I, X);
        return BinaryOperator::CreateNeg(X);
      }

      auto RemOpc = Div->getOpcode() == Instruction::UDiv ? Instruction::URem
                                                          : Instruction::SRem;
      Value *Rem = Builder.CreateBinOp(RemOpc, X, DivOp1);
      if (DivOp1 == Y)
        return BinaryOperator::CreateSub(X, Rem);
      return BinaryOperator::CreateSub(Rem, X);
    }
  }

  /// i1 mul -> i1 and.
  if (I.getType()->isIntOrIntVectorTy(1))
    return BinaryOperator::CreateAnd(Op0, Op1);

  // X*(1 << Y) --> X << Y
  // (1 << Y)*X --> X << Y
  {
    Value *Y;
    BinaryOperator *BO = nullptr;
    bool ShlNSW = false;
    if (match(Op0, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op1, Y);
      ShlNSW = cast<ShlOperator>(Op0)->hasNoSignedWrap();
    } else if (match(Op1, m_Shl(m_One(), m_Value(Y)))) {
      BO = BinaryOperator::CreateShl(Op0, Y);
      ShlNSW = cast<ShlOperator>(Op1)->hasNoSignedWrap();
    }
    if (BO) {
      if (I.hasNoUnsignedWrap())
        BO->setHasNoUnsignedWrap();
      if (I.hasNoSignedWrap() && ShlNSW)
        BO->setHasNoSignedWrap();
      return BO;
    }
  }

  // (bool X) * Y --> X ? Y : 0
  // Y * (bool X) --> X ? Y : 0
  if (match(Op0, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(X, Op1, ConstantInt::get(I.getType(), 0));
  if (match(Op1, m_ZExt(m_Value(X))) && X->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(X, Op0, ConstantInt::get(I.getType(), 0));

  // (lshr X, 31) * Y --> (ashr X, 31) & Y
  // Y * (lshr X, 31) --> (ashr X, 31) & Y
  // TODO: We are not checking one-use because the elimination of the multiply
  //       is better for analysis?
  // TODO: Should we canonicalize to '(X < 0) ? Y : 0' instead? That would be
  //       more similar to what we're doing above.
  const APInt *C;
  if (match(Op0, m_LShr(m_Value(X), m_APInt(C))) && *C == C->getBitWidth() - 1)
    return BinaryOperator::CreateAnd(Builder.CreateAShr(X, *C), Op1);
  if (match(Op1, m_LShr(m_Value(X), m_APInt(C))) && *C == C->getBitWidth() - 1)
    return BinaryOperator::CreateAnd(Builder.CreateAShr(X, *C), Op0);

  // Check for (mul (sext x), y), see if we can merge this into an
  // integer mul followed by a sext.
  if (SExtInst *Op0Conv = dyn_cast<SExtInst>(Op0)) {
    // (mul (sext x), cst) --> (sext (mul x, cst'))
    if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
      if (Op0Conv->hasOneUse()) {
        Constant *CI =
            ConstantExpr::getTrunc(Op1C, Op0Conv->getOperand(0)->getType());
        if (ConstantExpr::getSExt(CI, I.getType()) == Op1C &&
            willNotOverflowSignedMul(Op0Conv->getOperand(0), CI, I)) {
          // Insert the new, smaller mul.
          Value *NewMul =
              Builder.CreateNSWMul(Op0Conv->getOperand(0), CI, "mulconv");
          return new SExtInst(NewMul, I.getType());
        }
      }
    }

    // (mul (sext x), (sext y)) --> (sext (mul int x, y))
    if (SExtInst *Op1Conv = dyn_cast<SExtInst>(Op1)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of sexts), and if the
      // integer mul will not overflow.
      if (Op0Conv->getOperand(0)->getType() ==
              Op1Conv->getOperand(0)->getType() &&
          (Op0Conv->hasOneUse() || Op1Conv->hasOneUse()) &&
          willNotOverflowSignedMul(Op0Conv->getOperand(0),
                                   Op1Conv->getOperand(0), I)) {
        // Insert the new integer mul.
        Value *NewMul = Builder.CreateNSWMul(
            Op0Conv->getOperand(0), Op1Conv->getOperand(0), "mulconv");
        return new SExtInst(NewMul, I.getType());
      }
    }
  }

  // Check for (mul (zext x), y), see if we can merge this into an
  // integer mul followed by a zext.
  if (auto *Op0Conv = dyn_cast<ZExtInst>(Op0)) {
    // (mul (zext x), cst) --> (zext (mul x, cst'))
    if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
      if (Op0Conv->hasOneUse()) {
        Constant *CI =
            ConstantExpr::getTrunc(Op1C, Op0Conv->getOperand(0)->getType());
        if (ConstantExpr::getZExt(CI, I.getType()) == Op1C &&
            willNotOverflowUnsignedMul(Op0Conv->getOperand(0), CI, I)) {
          // Insert the new, smaller mul.
          Value *NewMul =
              Builder.CreateNUWMul(Op0Conv->getOperand(0), CI, "mulconv");
          return new ZExtInst(NewMul, I.getType());
        }
      }
    }

    // (mul (zext x), (zext y)) --> (zext (mul int x, y))
    if (auto *Op1Conv = dyn_cast<ZExtInst>(Op1)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of zexts), and if the
      // integer mul will not overflow.
      if (Op0Conv->getOperand(0)->getType() ==
              Op1Conv->getOperand(0)->getType() &&
          (Op0Conv->hasOneUse() || Op1Conv->hasOneUse()) &&
          willNotOverflowUnsignedMul(Op0Conv->getOperand(0),
                                     Op1Conv->getOperand(0), I)) {
        // Insert the new integer mul.
        Value *NewMul = Builder.CreateNUWMul(
            Op0Conv->getOperand(0), Op1Conv->getOperand(0), "mulconv");
        return new ZExtInst(NewMul, I.getType());
      }
    }
  }

  if (!I.hasNoSignedWrap() && willNotOverflowSignedMul(Op0, Op1, I)) {
    Changed = true;
    I.setHasNoSignedWrap(true);
  }

  if (!I.hasNoUnsignedWrap() && willNotOverflowUnsignedMul(Op0, Op1, I)) {
    Changed = true;
    I.setHasNoUnsignedWrap(true);
  }

  return Changed ? &I : nullptr;
}

/// Helper function of InstCombiner::visitFMul(). Return true iff the given
/// value is FMul or FDiv with one and only one operand being a finite-non-zero
/// constant (i.e. not Zero/NaN/Infinity).
static bool isFMulOrFDivWithConstant(Value *V) {
  Constant *C;
  return (match(V, m_FMul(m_Value(), m_Constant(C))) ||
          match(V, m_FDiv(m_Value(), m_Constant(C))) ||
          match(V, m_FDiv(m_Constant(C), m_Value()))) && C->isFiniteNonZeroFP();
}

/// foldFMulConst() is a helper routine of InstCombiner::visitFMul().
/// The input \p FMulOrDiv is a FMul/FDiv with one and only one operand
/// being a constant (i.e. isFMulOrFDivWithConstant(FMulOrDiv) == true).
/// This function is to simplify "FMulOrDiv * C" and returns the
/// resulting expression. Note that this function could return NULL in
/// case the constants cannot be folded into a normal floating-point.
Value *InstCombiner::foldFMulConst(Instruction *FMulOrDiv, Constant *C,
                                   Instruction *InsertBefore) {
  assert(isFMulOrFDivWithConstant(FMulOrDiv) && "V is invalid");

  Value *Opnd0 = FMulOrDiv->getOperand(0);
  Value *Opnd1 = FMulOrDiv->getOperand(1);

  Constant *C0 = dyn_cast<Constant>(Opnd0);
  Constant *C1 = dyn_cast<Constant>(Opnd1);

  BinaryOperator *R = nullptr;

  // (X * C0) * C => X * (C0*C)
  if (FMulOrDiv->getOpcode() == Instruction::FMul) {
    Constant *F = ConstantExpr::getFMul(C1 ? C1 : C0, C);
    if (F->isNormalFP())
      R = BinaryOperator::CreateFMul(C1 ? Opnd0 : Opnd1, F);
  } else {
    if (C0) {
      // (C0 / X) * C => (C0 * C) / X
      if (FMulOrDiv->hasOneUse()) {
        // It would otherwise introduce another div.
        Constant *F = ConstantExpr::getFMul(C0, C);
        if (F->isNormalFP())
          R = BinaryOperator::CreateFDiv(F, Opnd1);
      }
    } else {
      // (X / C1) * C => X * (C/C1) if C/C1 is not a denormal
      Constant *F = ConstantExpr::getFDiv(C, C1);
      if (F->isNormalFP()) {
        R = BinaryOperator::CreateFMul(Opnd0, F);
      } else {
        // (X / C1) * C => X / (C1/C)
        Constant *F = ConstantExpr::getFDiv(C1, C);
        if (F->isNormalFP())
          R = BinaryOperator::CreateFDiv(Opnd0, F);
      }
    }
  }

  if (R) {
    R->setFast(true);
    InsertNewInstWith(R, *InsertBefore);
  }

  return R;
}

Instruction *InstCombiner::visitFMul(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyFMulInst(Op0, Op1, I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *FoldedMul = foldBinOpIntoSelectOrPhi(I))
    return FoldedMul;

  // X * -1.0 --> -X
  if (match(Op1, m_SpecificFP(-1.0)))
    return BinaryOperator::CreateFNegFMF(Op0, &I);

  // -X * -Y --> X * Y
  Value *X, *Y;
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_FNeg(m_Value(Y))))
    return BinaryOperator::CreateFMulFMF(X, Y, &I);

  // -X * C --> X * -C
  Constant *C;
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_Constant(C)))
    return BinaryOperator::CreateFMulFMF(X, ConstantExpr::getFNeg(C), &I);

  // Sink negation: -X * Y --> -(X * Y)
  if (match(Op0, m_OneUse(m_FNeg(m_Value(X)))))
    return BinaryOperator::CreateFNegFMF(Builder.CreateFMulFMF(X, Op1, &I), &I);

  // Sink negation: Y * -X --> -(X * Y)
  if (match(Op1, m_OneUse(m_FNeg(m_Value(X)))))
    return BinaryOperator::CreateFNegFMF(Builder.CreateFMulFMF(X, Op0, &I), &I);

  // fabs(X) * fabs(X) -> X * X
  if (Op0 == Op1 && match(Op0, m_Intrinsic<Intrinsic::fabs>(m_Value(X))))
    return BinaryOperator::CreateFMulFMF(X, X, &I);

  // (select A, B, C) * (select A, D, E) --> select A, (B*D), (C*E)
  if (Value *V = SimplifySelectsFeedingBinaryOp(I, Op0, Op1))
    return replaceInstUsesWith(I, V);

  // Reassociate constant RHS with another constant to form constant expression.
  if (I.isFast() && match(Op1, m_Constant(C)) && C->isFiniteNonZeroFP()) {
    Constant *C1;
    if (match(Op0, m_OneUse(m_FDiv(m_Constant(C1), m_Value(X))))) {
      // (C1 / X) * C --> (C * C1) / X
      Constant *CC1 = ConstantExpr::getFMul(C, C1);
      if (CC1->isNormalFP())
        return BinaryOperator::CreateFDivFMF(CC1, X, &I);
    }
    if (match(Op0, m_FDiv(m_Value(X), m_Constant(C1)))) {
      // (X / C1) * C --> X * (C / C1)
      Constant *CDivC1 = ConstantExpr::getFDiv(C, C1);
      if (CDivC1->isNormalFP())
        return BinaryOperator::CreateFMulFMF(X, CDivC1, &I);

      // If the constant was a denormal, try reassociating differently.
      // (X / C1) * C --> X / (C1 / C)
      Constant *C1DivC = ConstantExpr::getFDiv(C1, C);
      if (Op0->hasOneUse() && C1DivC->isNormalFP())
        return BinaryOperator::CreateFDivFMF(X, C1DivC, &I);
    }

    // Let MDC denote an expression in one of these forms:
    // X * C, C/X, X/C, where C is a constant.
    // (MDC +/- C1) * C => (MDC * C) +/- (C1 * C)
    Instruction *FAddSub = dyn_cast<Instruction>(Op0);
    if (FAddSub &&
        (FAddSub->getOpcode() == Instruction::FAdd ||
         FAddSub->getOpcode() == Instruction::FSub)) {
      Value *Opnd0 = FAddSub->getOperand(0);
      Value *Opnd1 = FAddSub->getOperand(1);
      Constant *C0 = dyn_cast<Constant>(Opnd0);
      Constant *C1 = dyn_cast<Constant>(Opnd1);
      bool Swap = false;
      if (C0) {
        std::swap(C0, C1);
        std::swap(Opnd0, Opnd1);
        Swap = true;
      }

      if (C1 && C1->isFiniteNonZeroFP() && isFMulOrFDivWithConstant(Opnd0)) {
        Value *M1 = ConstantExpr::getFMul(C1, C);
        Value *M0 = cast<Constant>(M1)->isNormalFP() ?
                        foldFMulConst(cast<Instruction>(Opnd0), C, &I) :
                        nullptr;
        if (M0 && M1) {
          if (Swap && FAddSub->getOpcode() == Instruction::FSub)
            std::swap(M0, M1);

          Instruction *RI = (FAddSub->getOpcode() == Instruction::FAdd)
                                ? BinaryOperator::CreateFAdd(M0, M1)
                                : BinaryOperator::CreateFSub(M0, M1);
          RI->copyFastMathFlags(&I);
          return RI;
        }
      }
    }
  }

  // log2(X * 0.5) * Y = log2(X) * Y - Y
  if (I.isFast()) {
    IntrinsicInst *Log2 = nullptr;
    if (match(Op0, m_OneUse(m_Intrinsic<Intrinsic::log2>(
            m_OneUse(m_FMul(m_Value(X), m_SpecificFP(0.5))))))) {
      Log2 = cast<IntrinsicInst>(Op0);
      Y = Op1;
    }
    if (match(Op1, m_OneUse(m_Intrinsic<Intrinsic::log2>(
            m_OneUse(m_FMul(m_Value(X), m_SpecificFP(0.5))))))) {
      Log2 = cast<IntrinsicInst>(Op1);
      Y = Op0;
    }
    if (Log2) {
      Log2->setArgOperand(0, X);
      Log2->copyFastMathFlags(&I);
      Value *LogXTimesY = Builder.CreateFMulFMF(Log2, Y, &I);
      return BinaryOperator::CreateFSubFMF(LogXTimesY, Y, &I);
    }
  }

  // sqrt(X) * sqrt(Y) -> sqrt(X * Y)
  // nnan disallows the possibility of returning a number if both operands are
  // negative (in that case, we should return NaN).
  if (I.hasAllowReassoc() && I.hasNoNaNs() &&
      match(Op0, m_OneUse(m_Intrinsic<Intrinsic::sqrt>(m_Value(X)))) &&
      match(Op1, m_OneUse(m_Intrinsic<Intrinsic::sqrt>(m_Value(Y))))) {
    Value *XY = Builder.CreateFMulFMF(X, Y, &I);
    Value *Sqrt = Builder.CreateIntrinsic(Intrinsic::sqrt, { XY }, &I);
    return replaceInstUsesWith(I, Sqrt);
  }

  // (X*Y) * X => (X*X) * Y where Y != X
  //  The purpose is two-fold:
  //   1) to form a power expression (of X).
  //   2) potentially shorten the critical path: After transformation, the
  //  latency of the instruction Y is amortized by the expression of X*X,
  //  and therefore Y is in a "less critical" position compared to what it
  //  was before the transformation.
  if (I.hasAllowReassoc()) {
    if (match(Op0, m_OneUse(m_c_FMul(m_Specific(Op1), m_Value(Y)))) &&
        Op1 != Y) {
      Value *XX = Builder.CreateFMulFMF(Op1, Op1, &I);
      return BinaryOperator::CreateFMulFMF(XX, Y, &I);
    }
    if (match(Op1, m_OneUse(m_c_FMul(m_Specific(Op0), m_Value(Y)))) &&
        Op0 != Y) {
      Value *XX = Builder.CreateFMulFMF(Op0, Op0, &I);
      return BinaryOperator::CreateFMulFMF(XX, Y, &I);
    }
  }

  return Changed ? &I : nullptr;
}

/// Fold a divide or remainder with a select instruction divisor when one of the
/// select operands is zero. In that case, we can use the other select operand
/// because div/rem by zero is undefined.
bool InstCombiner::simplifyDivRemOfSelectWithZeroOp(BinaryOperator &I) {
  SelectInst *SI = dyn_cast<SelectInst>(I.getOperand(1));
  if (!SI)
    return false;

  int NonNullOperand;
  if (match(SI->getTrueValue(), m_Zero()))
    // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
    NonNullOperand = 2;
  else if (match(SI->getFalseValue(), m_Zero()))
    // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
    NonNullOperand = 1;
  else
    return false;

  // Change the div/rem to use 'Y' instead of the select.
  I.setOperand(1, SI->getOperand(NonNullOperand));

  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.

  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  Value *SelectCond = SI->getCondition();
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;

  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = I.getIterator(), BBFront = I.getParent()->begin();
  Type *CondTy = SelectCond->getType();
  while (BBI != BBFront) {
    --BBI;
    // If we found a call to a function, we can't assume it will return, so
    // information from below it cannot be propagated above it.
    if (isa<CallInst>(BBI) && !isa<IntrinsicInst>(BBI))
      break;

    // Replace uses of the select or its condition with the known values.
    for (Instruction::op_iterator I = BBI->op_begin(), E = BBI->op_end();
         I != E; ++I) {
      if (*I == SI) {
        *I = SI->getOperand(NonNullOperand);
        Worklist.Add(&*BBI);
      } else if (*I == SelectCond) {
        *I = NonNullOperand == 1 ? ConstantInt::getTrue(CondTy)
                                 : ConstantInt::getFalse(CondTy);
        Worklist.Add(&*BBI);
      }
    }

    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = nullptr;
    if (&*BBI == SelectCond)
      SelectCond = nullptr;

    // If we ran out of things to eliminate, break out of the loop.
    if (!SelectCond && !SI)
      break;

  }
  return true;
}

/// True if the multiply can not be expressed in an int this size.
static bool multiplyOverflows(const APInt &C1, const APInt &C2, APInt &Product,
                              bool IsSigned) {
  bool Overflow;
  Product = IsSigned ? C1.smul_ov(C2, Overflow) : C1.umul_ov(C2, Overflow);
  return Overflow;
}

/// True if C2 is a multiple of C1. Quotient contains C2/C1.
static bool isMultiple(const APInt &C1, const APInt &C2, APInt &Quotient,
                       bool IsSigned) {
  assert(C1.getBitWidth() == C2.getBitWidth() && "Constant widths not equal");

  // Bail if we will divide by zero.
  if (C2.isNullValue())
    return false;

  // Bail if we would divide INT_MIN by -1.
  if (IsSigned && C1.isMinSignedValue() && C2.isAllOnesValue())
    return false;

  APInt Remainder(C1.getBitWidth(), /*Val=*/0ULL, IsSigned);
  if (IsSigned)
    APInt::sdivrem(C1, C2, Quotient, Remainder);
  else
    APInt::udivrem(C1, C2, Quotient, Remainder);

  return Remainder.isMinValue();
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  bool IsSigned = I.getOpcode() == Instruction::SDiv;
  Type *Ty = I.getType();

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (simplifyDivRemOfSelectWithZeroOp(I))
    return &I;

  const APInt *C2;
  if (match(Op1, m_APInt(C2))) {
    Value *X;
    const APInt *C1;

    // (X / C1) / C2  -> X / (C1*C2)
    if ((IsSigned && match(Op0, m_SDiv(m_Value(X), m_APInt(C1)))) ||
        (!IsSigned && match(Op0, m_UDiv(m_Value(X), m_APInt(C1))))) {
      APInt Product(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);
      if (!multiplyOverflows(*C1, *C2, Product, IsSigned))
        return BinaryOperator::Create(I.getOpcode(), X,
                                      ConstantInt::get(Ty, Product));
    }

    if ((IsSigned && match(Op0, m_NSWMul(m_Value(X), m_APInt(C1)))) ||
        (!IsSigned && match(Op0, m_NUWMul(m_Value(X), m_APInt(C1))))) {
      APInt Quotient(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);

      // (X * C1) / C2 -> X / (C2 / C1) if C2 is a multiple of C1.
      if (isMultiple(*C2, *C1, Quotient, IsSigned)) {
        auto *NewDiv = BinaryOperator::Create(I.getOpcode(), X,
                                              ConstantInt::get(Ty, Quotient));
        NewDiv->setIsExact(I.isExact());
        return NewDiv;
      }

      // (X * C1) / C2 -> X * (C1 / C2) if C1 is a multiple of C2.
      if (isMultiple(*C1, *C2, Quotient, IsSigned)) {
        auto *Mul = BinaryOperator::Create(Instruction::Mul, X,
                                           ConstantInt::get(Ty, Quotient));
        auto *OBO = cast<OverflowingBinaryOperator>(Op0);
        Mul->setHasNoUnsignedWrap(!IsSigned && OBO->hasNoUnsignedWrap());
        Mul->setHasNoSignedWrap(OBO->hasNoSignedWrap());
        return Mul;
      }
    }

    if ((IsSigned && match(Op0, m_NSWShl(m_Value(X), m_APInt(C1))) &&
         *C1 != C1->getBitWidth() - 1) ||
        (!IsSigned && match(Op0, m_NUWShl(m_Value(X), m_APInt(C1))))) {
      APInt Quotient(C1->getBitWidth(), /*Val=*/0ULL, IsSigned);
      APInt C1Shifted = APInt::getOneBitSet(
          C1->getBitWidth(), static_cast<unsigned>(C1->getLimitedValue()));

      // (X << C1) / C2 -> X / (C2 >> C1) if C2 is a multiple of C1.
      if (isMultiple(*C2, C1Shifted, Quotient, IsSigned)) {
        auto *BO = BinaryOperator::Create(I.getOpcode(), X,
                                          ConstantInt::get(Ty, Quotient));
        BO->setIsExact(I.isExact());
        return BO;
      }

      // (X << C1) / C2 -> X * (C2 >> C1) if C1 is a multiple of C2.
      if (isMultiple(C1Shifted, *C2, Quotient, IsSigned)) {
        auto *Mul = BinaryOperator::Create(Instruction::Mul, X,
                                           ConstantInt::get(Ty, Quotient));
        auto *OBO = cast<OverflowingBinaryOperator>(Op0);
        Mul->setHasNoUnsignedWrap(!IsSigned && OBO->hasNoUnsignedWrap());
        Mul->setHasNoSignedWrap(OBO->hasNoSignedWrap());
        return Mul;
      }
    }

    if (!C2->isNullValue()) // avoid X udiv 0
      if (Instruction *FoldedDiv = foldBinOpIntoSelectOrPhi(I))
        return FoldedDiv;
  }

  if (match(Op0, m_One())) {
    assert(!Ty->isIntOrIntVectorTy(1) && "i1 divide not removed?");
    if (IsSigned) {
      // If Op1 is 0 then it's undefined behaviour, if Op1 is 1 then the
      // result is one, if Op1 is -1 then the result is minus one, otherwise
      // it's zero.
      Value *Inc = Builder.CreateAdd(Op1, Op0);
      Value *Cmp = Builder.CreateICmpULT(Inc, ConstantInt::get(Ty, 3));
      return SelectInst::Create(Cmp, Op1, ConstantInt::get(Ty, 0));
    } else {
      // If Op1 is 0 then it's undefined behaviour. If Op1 is 1 then the
      // result is one, otherwise it's zero.
      return new ZExtInst(Builder.CreateICmpEQ(Op1, Op0), Ty);
    }
  }

  // See if we can fold away this div instruction.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // (X - (X rem Y)) / Y -> X / Y; usually originates as ((X / Y) * Y) / Y
  Value *X, *Z;
  if (match(Op0, m_Sub(m_Value(X), m_Value(Z)))) // (X - Z) / Y; Y = Op1
    if ((IsSigned && match(Z, m_SRem(m_Specific(X), m_Specific(Op1)))) ||
        (!IsSigned && match(Z, m_URem(m_Specific(X), m_Specific(Op1)))))
      return BinaryOperator::Create(I.getOpcode(), X, Op1);

  // (X << Y) / X -> 1 << Y
  Value *Y;
  if (IsSigned && match(Op0, m_NSWShl(m_Specific(Op1), m_Value(Y))))
    return BinaryOperator::CreateNSWShl(ConstantInt::get(Ty, 1), Y);
  if (!IsSigned && match(Op0, m_NUWShl(m_Specific(Op1), m_Value(Y))))
    return BinaryOperator::CreateNUWShl(ConstantInt::get(Ty, 1), Y);

  // X / (X * Y) -> 1 / Y if the multiplication does not overflow.
  if (match(Op1, m_c_Mul(m_Specific(Op0), m_Value(Y)))) {
    bool HasNSW = cast<OverflowingBinaryOperator>(Op1)->hasNoSignedWrap();
    bool HasNUW = cast<OverflowingBinaryOperator>(Op1)->hasNoUnsignedWrap();
    if ((IsSigned && HasNSW) || (!IsSigned && HasNUW)) {
      I.setOperand(0, ConstantInt::get(Ty, 1));
      I.setOperand(1, Y);
      return &I;
    }
  }

  return nullptr;
}

static const unsigned MaxDepth = 6;

namespace {

using FoldUDivOperandCb = Instruction *(*)(Value *Op0, Value *Op1,
                                           const BinaryOperator &I,
                                           InstCombiner &IC);

/// \brief Used to maintain state for visitUDivOperand().
struct UDivFoldAction {
  /// Informs visitUDiv() how to fold this operand.  This can be zero if this
  /// action joins two actions together.
  FoldUDivOperandCb FoldAction;

  /// Which operand to fold.
  Value *OperandToFold;

  union {
    /// The instruction returned when FoldAction is invoked.
    Instruction *FoldResult;

    /// Stores the LHS action index if this action joins two actions together.
    size_t SelectLHSIdx;
  };

  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand)
      : FoldAction(FA), OperandToFold(InputOperand), FoldResult(nullptr) {}
  UDivFoldAction(FoldUDivOperandCb FA, Value *InputOperand, size_t SLHS)
      : FoldAction(FA), OperandToFold(InputOperand), SelectLHSIdx(SLHS) {}
};

} // end anonymous namespace

// X udiv 2^C -> X >> C
static Instruction *foldUDivPow2Cst(Value *Op0, Value *Op1,
                                    const BinaryOperator &I, InstCombiner &IC) {
  Constant *C1 = getLogBase2(Op0->getType(), cast<Constant>(Op1));
  if (!C1)
    llvm_unreachable("Failed to constant fold udiv -> logbase2");
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, C1);
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// X udiv C, where C >= signbit
static Instruction *foldUDivNegCst(Value *Op0, Value *Op1,
                                   const BinaryOperator &I, InstCombiner &IC) {
  Value *ICI = IC.Builder.CreateICmpULT(Op0, cast<Constant>(Op1));
  return SelectInst::Create(ICI, Constant::getNullValue(I.getType()),
                            ConstantInt::get(I.getType(), 1));
}

// X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
// X udiv (zext (C1 << N)), where C1 is "1<<C2"  -->  X >> (N+C2)
static Instruction *foldUDivShl(Value *Op0, Value *Op1, const BinaryOperator &I,
                                InstCombiner &IC) {
  Value *ShiftLeft;
  if (!match(Op1, m_ZExt(m_Value(ShiftLeft))))
    ShiftLeft = Op1;

  Constant *CI;
  Value *N;
  if (!match(ShiftLeft, m_Shl(m_Constant(CI), m_Value(N))))
    llvm_unreachable("match should never fail here!");
  Constant *Log2Base = getLogBase2(N->getType(), CI);
  if (!Log2Base)
    llvm_unreachable("getLogBase2 should never fail here!");
  N = IC.Builder.CreateAdd(N, Log2Base);
  if (Op1 != ShiftLeft)
    N = IC.Builder.CreateZExt(N, Op1->getType());
  BinaryOperator *LShr = BinaryOperator::CreateLShr(Op0, N);
  if (I.isExact())
    LShr->setIsExact();
  return LShr;
}

// \brief Recursively visits the possible right hand operands of a udiv
// instruction, seeing through select instructions, to determine if we can
// replace the udiv with something simpler.  If we find that an operand is not
// able to simplify the udiv, we abort the entire transformation.
static size_t visitUDivOperand(Value *Op0, Value *Op1, const BinaryOperator &I,
                               SmallVectorImpl<UDivFoldAction> &Actions,
                               unsigned Depth = 0) {
  // Check to see if this is an unsigned division with an exact power of 2,
  // if so, convert to a right shift.
  if (match(Op1, m_Power2())) {
    Actions.push_back(UDivFoldAction(foldUDivPow2Cst, Op1));
    return Actions.size();
  }

  // X udiv C, where C >= signbit
  if (match(Op1, m_Negative())) {
    Actions.push_back(UDivFoldAction(foldUDivNegCst, Op1));
    return Actions.size();
  }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (match(Op1, m_Shl(m_Power2(), m_Value())) ||
      match(Op1, m_ZExt(m_Shl(m_Power2(), m_Value())))) {
    Actions.push_back(UDivFoldAction(foldUDivShl, Op1));
    return Actions.size();
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxDepth)
    return 0;

  if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
    if (size_t LHSIdx =
            visitUDivOperand(Op0, SI->getOperand(1), I, Actions, Depth))
      if (visitUDivOperand(Op0, SI->getOperand(2), I, Actions, Depth)) {
        Actions.push_back(UDivFoldAction(nullptr, Op1, LHSIdx - 1));
        return Actions.size();
      }

  return 0;
}

/// If we have zero-extended operands of an unsigned div or rem, we may be able
/// to narrow the operation (sink the zext below the math).
static Instruction *narrowUDivURem(BinaryOperator &I,
                                   InstCombiner::BuilderTy &Builder) {
  Instruction::BinaryOps Opcode = I.getOpcode();
  Value *N = I.getOperand(0);
  Value *D = I.getOperand(1);
  Type *Ty = I.getType();
  Value *X, *Y;
  if (match(N, m_ZExt(m_Value(X))) && match(D, m_ZExt(m_Value(Y))) &&
      X->getType() == Y->getType() && (N->hasOneUse() || D->hasOneUse())) {
    // udiv (zext X), (zext Y) --> zext (udiv X, Y)
    // urem (zext X), (zext Y) --> zext (urem X, Y)
    Value *NarrowOp = Builder.CreateBinOp(Opcode, X, Y);
    return new ZExtInst(NarrowOp, Ty);
  }

  Constant *C;
  if ((match(N, m_OneUse(m_ZExt(m_Value(X)))) && match(D, m_Constant(C))) ||
      (match(D, m_OneUse(m_ZExt(m_Value(X)))) && match(N, m_Constant(C)))) {
    // If the constant is the same in the smaller type, use the narrow version.
    Constant *TruncC = ConstantExpr::getTrunc(C, X->getType());
    if (ConstantExpr::getZExt(TruncC, Ty) != C)
      return nullptr;

    // udiv (zext X), C --> zext (udiv X, C')
    // urem (zext X), C --> zext (urem X, C')
    // udiv C, (zext X) --> zext (udiv C', X)
    // urem C, (zext X) --> zext (urem C', X)
    Value *NarrowOp = isa<Constant>(D) ? Builder.CreateBinOp(Opcode, X, TruncC)
                                       : Builder.CreateBinOp(Opcode, TruncC, X);
    return new ZExtInst(NarrowOp, Ty);
  }

  return nullptr;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyUDivInst(Op0, Op1, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  // (x lshr C1) udiv C2 --> x udiv (C2 << C1)
  {
    Value *X;
    const APInt *C1, *C2;
    if (match(Op0, m_LShr(m_Value(X), m_APInt(C1))) &&
        match(Op1, m_APInt(C2))) {
      bool Overflow;
      APInt C2ShlC1 = C2->ushl_ov(*C1, Overflow);
      if (!Overflow) {
        bool IsExact = I.isExact() && match(Op0, m_Exact(m_Value()));
        BinaryOperator *BO = BinaryOperator::CreateUDiv(
            X, ConstantInt::get(X->getType(), C2ShlC1));
        if (IsExact)
          BO->setIsExact();
        return BO;
      }
    }
  }

  if (Instruction *NarrowDiv = narrowUDivURem(I, Builder))
    return NarrowDiv;

  // (LHS udiv (select (select (...)))) -> (LHS >> (select (select (...))))
  SmallVector<UDivFoldAction, 6> UDivActions;
  if (visitUDivOperand(Op0, Op1, I, UDivActions))
    for (unsigned i = 0, e = UDivActions.size(); i != e; ++i) {
      FoldUDivOperandCb Action = UDivActions[i].FoldAction;
      Value *ActionOp1 = UDivActions[i].OperandToFold;
      Instruction *Inst;
      if (Action)
        Inst = Action(Op0, ActionOp1, I, *this);
      else {
        // This action joins two actions together.  The RHS of this action is
        // simply the last action we processed, we saved the LHS action index in
        // the joining action.
        size_t SelectRHSIdx = i - 1;
        Value *SelectRHS = UDivActions[SelectRHSIdx].FoldResult;
        size_t SelectLHSIdx = UDivActions[i].SelectLHSIdx;
        Value *SelectLHS = UDivActions[SelectLHSIdx].FoldResult;
        Inst = SelectInst::Create(cast<SelectInst>(ActionOp1)->getCondition(),
                                  SelectLHS, SelectRHS);
      }

      // If this is the last action to process, return it to the InstCombiner.
      // Otherwise, we insert it before the UDiv and record it so that we may
      // use it as part of a joining action (i.e., a SelectInst).
      if (e - i != 1) {
        Inst->insertBefore(&I);
        UDivActions[i].FoldResult = Inst;
      } else
        return Inst;
    }

  return nullptr;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifySDivInst(Op0, Op1, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  const APInt *Op1C;
  if (match(Op1, m_APInt(Op1C))) {
    // sdiv X, -1 == -X
    if (Op1C->isAllOnesValue())
      return BinaryOperator::CreateNeg(Op0);

    // sdiv exact X, C  -->  ashr exact X, log2(C)
    if (I.isExact() && Op1C->isNonNegative() && Op1C->isPowerOf2()) {
      Value *ShAmt = ConstantInt::get(Op1->getType(), Op1C->exactLogBase2());
      return BinaryOperator::CreateExactAShr(Op0, ShAmt, I.getName());
    }

    // If the dividend is sign-extended and the constant divisor is small enough
    // to fit in the source type, shrink the division to the narrower type:
    // (sext X) sdiv C --> sext (X sdiv C)
    Value *Op0Src;
    if (match(Op0, m_OneUse(m_SExt(m_Value(Op0Src)))) &&
        Op0Src->getType()->getScalarSizeInBits() >= Op1C->getMinSignedBits()) {

      // In the general case, we need to make sure that the dividend is not the
      // minimum signed value because dividing that by -1 is UB. But here, we
      // know that the -1 divisor case is already handled above.

      Constant *NarrowDivisor =
          ConstantExpr::getTrunc(cast<Constant>(Op1), Op0Src->getType());
      Value *NarrowOp = Builder.CreateSDiv(Op0Src, NarrowDivisor);
      return new SExtInst(NarrowOp, Op0->getType());
    }
  }

  if (Constant *RHS = dyn_cast<Constant>(Op1)) {
    // X/INT_MIN -> X == INT_MIN
    if (RHS->isMinSignedValue())
      return new ZExtInst(Builder.CreateICmpEQ(Op0, Op1), I.getType());

    // -X/C  -->  X/-C  provided the negation doesn't overflow.
    Value *X;
    if (match(Op0, m_NSWSub(m_Zero(), m_Value(X)))) {
      auto *BO = BinaryOperator::CreateSDiv(X, ConstantExpr::getNeg(RHS));
      BO->setIsExact(I.isExact());
      return BO;
    }
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  APInt Mask(APInt::getSignMask(I.getType()->getScalarSizeInBits()));
  if (MaskedValueIsZero(Op0, Mask, 0, &I)) {
    if (MaskedValueIsZero(Op1, Mask, 0, &I)) {
      // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
      auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      BO->setIsExact(I.isExact());
      return BO;
    }

    if (isKnownToBeAPowerOfTwo(Op1, /*OrZero*/ true, 0, &I)) {
      // X sdiv (1 << Y) -> X udiv (1 << Y) ( -> X u>> Y)
      // Safe because the only negative value (1 << Y) can take on is
      // INT_MIN, and X sdiv INT_MIN == X udiv INT_MIN == 0 if X doesn't have
      // the sign bit set.
      auto *BO = BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      BO->setIsExact(I.isExact());
      return BO;
    }
  }

  return nullptr;
}

/// Remove negation and try to convert division into multiplication.
static Instruction *foldFDivConstantDivisor(BinaryOperator &I) {
  Constant *C;
  if (!match(I.getOperand(1), m_Constant(C)))
    return nullptr;

  // -X / C --> X / -C
  Value *X;
  if (match(I.getOperand(0), m_FNeg(m_Value(X))))
    return BinaryOperator::CreateFDivFMF(X, ConstantExpr::getFNeg(C), &I);

  // If the constant divisor has an exact inverse, this is always safe. If not,
  // then we can still create a reciprocal if fast-math-flags allow it and the
  // constant is a regular number (not zero, infinite, or denormal).
  if (!(C->hasExactInverseFP() || (I.hasAllowReciprocal() && C->isNormalFP())))
    return nullptr;

  // Disallow denormal constants because we don't know what would happen
  // on all targets.
  // TODO: Use Intrinsic::canonicalize or let function attributes tell us that
  // denorms are flushed?
  auto *RecipC = ConstantExpr::getFDiv(ConstantFP::get(I.getType(), 1.0), C);
  if (!RecipC->isNormalFP())
    return nullptr;

  // X / C --> X * (1 / C)
  return BinaryOperator::CreateFMulFMF(I.getOperand(0), RecipC, &I);
}

/// Remove negation and try to reassociate constant math.
static Instruction *foldFDivConstantDividend(BinaryOperator &I) {
  Constant *C;
  if (!match(I.getOperand(0), m_Constant(C)))
    return nullptr;

  // C / -X --> -C / X
  Value *X;
  if (match(I.getOperand(1), m_FNeg(m_Value(X))))
    return BinaryOperator::CreateFDivFMF(ConstantExpr::getFNeg(C), X, &I);

  if (!I.hasAllowReassoc() || !I.hasAllowReciprocal())
    return nullptr;

  // Try to reassociate C / X expressions where X includes another constant.
  Constant *C2, *NewC = nullptr;
  if (match(I.getOperand(1), m_FMul(m_Value(X), m_Constant(C2)))) {
    // C / (X * C2) --> (C / C2) / X
    NewC = ConstantExpr::getFDiv(C, C2);
  } else if (match(I.getOperand(1), m_FDiv(m_Value(X), m_Constant(C2)))) {
    // C / (X / C2) --> (C * C2) / X
    NewC = ConstantExpr::getFMul(C, C2);
  }
  // Disallow denormal constants because we don't know what would happen
  // on all targets.
  // TODO: Use Intrinsic::canonicalize or let function attributes tell us that
  // denorms are flushed?
  if (!NewC || !NewC->isNormalFP())
    return nullptr;

  return BinaryOperator::CreateFDivFMF(NewC, X, &I);
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyFDivInst(Op0, Op1, I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *R = foldFDivConstantDivisor(I))
    return R;

  if (Instruction *R = foldFDivConstantDividend(I))
    return R;

  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (isa<Constant>(Op1))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (I.hasAllowReassoc() && I.hasAllowReciprocal()) {
    Value *X, *Y;
    if (match(Op0, m_OneUse(m_FDiv(m_Value(X), m_Value(Y)))) &&
        (!isa<Constant>(Y) || !isa<Constant>(Op1))) {
      // (X / Y) / Z => X / (Y * Z)
      Value *YZ = Builder.CreateFMulFMF(Y, Op1, &I);
      return BinaryOperator::CreateFDivFMF(X, YZ, &I);
    }
    if (match(Op1, m_OneUse(m_FDiv(m_Value(X), m_Value(Y)))) &&
        (!isa<Constant>(Y) || !isa<Constant>(Op0))) {
      // Z / (X / Y) => (Y * Z) / X
      Value *YZ = Builder.CreateFMulFMF(Y, Op0, &I);
      return BinaryOperator::CreateFDivFMF(YZ, X, &I);
    }
  }

  if (I.hasAllowReassoc() && Op0->hasOneUse() && Op1->hasOneUse()) {
    // sin(X) / cos(X) -> tan(X)
    // cos(X) / sin(X) -> 1/tan(X) (cotangent)
    Value *X;
    bool IsTan = match(Op0, m_Intrinsic<Intrinsic::sin>(m_Value(X))) &&
                 match(Op1, m_Intrinsic<Intrinsic::cos>(m_Specific(X)));
    bool IsCot =
        !IsTan && match(Op0, m_Intrinsic<Intrinsic::cos>(m_Value(X))) &&
                  match(Op1, m_Intrinsic<Intrinsic::sin>(m_Specific(X)));

    if ((IsTan || IsCot) && hasUnaryFloatFn(&TLI, I.getType(), LibFunc_tan,
                                            LibFunc_tanf, LibFunc_tanl)) {
      IRBuilder<> B(&I);
      IRBuilder<>::FastMathFlagGuard FMFGuard(B);
      B.setFastMathFlags(I.getFastMathFlags());
      AttributeList Attrs = CallSite(Op0).getCalledFunction()->getAttributes();
      Value *Res = emitUnaryFloatFnCall(X, TLI.getName(LibFunc_tan), B, Attrs);
      if (IsCot)
        Res = B.CreateFDiv(ConstantFP::get(I.getType(), 1.0), Res);
      return replaceInstUsesWith(I, Res);
    }
  }

  // -X / -Y -> X / Y
  Value *X, *Y;
  if (match(Op0, m_FNeg(m_Value(X))) && match(Op1, m_FNeg(m_Value(Y)))) {
    I.setOperand(0, X);
    I.setOperand(1, Y);
    return &I;
  }

  // X / (X * Y) --> 1.0 / Y
  // Reassociate to (X / X -> 1.0) is legal when NaNs are not allowed.
  // We can ignore the possibility that X is infinity because INF/INF is NaN.
  if (I.hasNoNaNs() && I.hasAllowReassoc() &&
      match(Op1, m_c_FMul(m_Specific(Op0), m_Value(Y)))) {
    I.setOperand(0, ConstantFP::get(I.getType(), 1.0));
    I.setOperand(1, Y);
    return &I;
  }

  return nullptr;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // The RHS is known non-zero.
  if (Value *V = simplifyValueKnownNonZero(I.getOperand(1), *this, I)) {
    I.setOperand(1, V);
    return &I;
  }

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (simplifyDivRemOfSelectWithZeroOp(I))
    return &I;

  if (isa<Constant>(Op1)) {
    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      } else if (auto *PN = dyn_cast<PHINode>(Op0I)) {
        const APInt *Op1Int;
        if (match(Op1, m_APInt(Op1Int)) && !Op1Int->isMinValue() &&
            (I.getOpcode() == Instruction::URem ||
             !Op1Int->isMinSignedValue())) {
          // foldOpIntoPhi will speculate instructions to the end of the PHI's
          // predecessor blocks, so do this only if we know the srem or urem
          // will not fault.
          if (Instruction *NV = foldOpIntoPhi(I, PN))
            return NV;
        }
      }

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitURem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyURemInst(Op0, Op1, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (Instruction *common = commonIRemTransforms(I))
    return common;

  if (Instruction *NarrowRem = narrowUDivURem(I, Builder))
    return NarrowRem;

  // X urem Y -> X and Y-1, where Y is a power of 2,
  if (isKnownToBeAPowerOfTwo(Op1, /*OrZero*/ true, 0, &I)) {
    Constant *N1 = Constant::getAllOnesValue(I.getType());
    Value *Add = Builder.CreateAdd(Op1, N1);
    return BinaryOperator::CreateAnd(Op0, Add);
  }

  // 1 urem X -> zext(X != 1)
  if (match(Op0, m_One())) {
    Value *Cmp = Builder.CreateICmpNE(Op1, Op0);
    Value *Ext = Builder.CreateZExt(Cmp, I.getType());
    return replaceInstUsesWith(I, Ext);
  }

  // X urem C -> X < C ? X : X - C, where C >= signbit.
  if (match(Op1, m_Negative())) {
    Value *Cmp = Builder.CreateICmpULT(Op0, Op1);
    Value *Sub = Builder.CreateSub(Op0, Op1);
    return SelectInst::Create(Cmp, Op0, Sub);
  }

  return nullptr;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifySRemInst(Op0, Op1, SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  // Handle the integer rem common cases
  if (Instruction *Common = commonIRemTransforms(I))
    return Common;

  {
    const APInt *Y;
    // X % -Y -> X % Y
    if (match(Op1, m_Negative(Y)) && !Y->isMinSignedValue()) {
      Worklist.AddValue(I.getOperand(1));
      I.setOperand(1, ConstantInt::get(I.getType(), -*Y));
      return &I;
    }
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  APInt Mask(APInt::getSignMask(I.getType()->getScalarSizeInBits()));
  if (MaskedValueIsZero(Op1, Mask, 0, &I) &&
      MaskedValueIsZero(Op0, Mask, 0, &I)) {
    // X srem Y -> X urem Y, iff X and Y don't have sign bit set
    return BinaryOperator::CreateURem(Op0, Op1, I.getName());
  }

  // If it's a constant vector, flip any negative values positive.
  if (isa<ConstantVector>(Op1) || isa<ConstantDataVector>(Op1)) {
    Constant *C = cast<Constant>(Op1);
    unsigned VWidth = C->getType()->getVectorNumElements();

    bool hasNegative = false;
    bool hasMissing = false;
    for (unsigned i = 0; i != VWidth; ++i) {
      Constant *Elt = C->getAggregateElement(i);
      if (!Elt) {
        hasMissing = true;
        break;
      }

      if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elt))
        if (RHS->isNegative())
          hasNegative = true;
    }

    if (hasNegative && !hasMissing) {
      SmallVector<Constant *, 16> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        Elts[i] = C->getAggregateElement(i);  // Handle undef, etc.
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(Elts[i])) {
          if (RHS->isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != C) {  // Don't loop on -MININT
        Worklist.AddValue(I.getOperand(1));
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyFRemInst(Op0, Op1, I.getFastMathFlags(),
                                  SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  return nullptr;
}
