//===- InstCombineAddSub.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for add, fadd, sub, and fsub.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/DataLayout.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

/// AddOne - Add one to a ConstantInt.
static Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// SubOne - Subtract one from a ConstantInt.
static Constant *SubOne(ConstantInt *C) {
  return ConstantInt::get(C->getContext(), C->getValue()-1);
}


// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply, and set CST to point to the multiplier.
// Otherwise, return null.
//
static inline Value *dyn_castFoldableMul(Value *V, ConstantInt *&CST) {
  if (!V->hasOneUse() || !V->getType()->isIntegerTy())
    return 0;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return 0;
  
  if (I->getOpcode() == Instruction::Mul)
    if ((CST = dyn_cast<ConstantInt>(I->getOperand(1))))
      return I->getOperand(0);
  if (I->getOpcode() == Instruction::Shl)
    if ((CST = dyn_cast<ConstantInt>(I->getOperand(1)))) {
      // The multiplier is really 1 << CST.
      uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
      uint32_t CSTVal = CST->getLimitedValue(BitWidth);
      CST = ConstantInt::get(V->getType()->getContext(),
                             APInt(BitWidth, 1).shl(CSTVal));
      return I->getOperand(0);
    }
  return 0;
}


/// WillNotOverflowSignedAdd - Return true if we can prove that:
///    (sext (add LHS, RHS))  === (add (sext LHS), (sext RHS))
/// This basically requires proving that the add in the original type would not
/// overflow to change the sign bit or have a carry out.
bool InstCombiner::WillNotOverflowSignedAdd(Value *LHS, Value *RHS) {
  // There are different heuristics we can use for this.  Here are some simple
  // ones.
  
  // Add has the property that adding any two 2's complement numbers can only 
  // have one carry bit which can change a sign.  As such, if LHS and RHS each
  // have at least two sign bits, we know that the addition of the two values
  // will sign extend fine.
  if (ComputeNumSignBits(LHS) > 1 && ComputeNumSignBits(RHS) > 1)
    return true;
  
  
  // If one of the operands only has one non-zero bit, and if the other operand
  // has a known-zero bit in a more significant place than it (not including the
  // sign bit) the ripple may go up to and fill the zero, but won't change the
  // sign.  For example, (X & ~4) + 1.
  
  // TODO: Implement.
  
  return false;
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Value *V = SimplifyAddInst(LHS, RHS, I.hasNoSignedWrap(),
                                 I.hasNoUnsignedWrap(), TD))
    return ReplaceInstUsesWith(I, V);

  // (A*B)+(A*C) -> A*(B+C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    // X + (signbit) --> X ^ signbit
    const APInt &Val = CI->getValue();
    if (Val.isSignBit())
      return BinaryOperator::CreateXor(LHS, RHS);
    
    // See if SimplifyDemandedBits can simplify this.  This handles stuff like
    // (X & 254)+1 -> (X&254)|1
    if (SimplifyDemandedInstructionBits(I))
      return &I;

    // zext(bool) + C -> bool ? C + 1 : C
    if (ZExtInst *ZI = dyn_cast<ZExtInst>(LHS))
      if (ZI->getSrcTy()->isIntegerTy(1))
        return SelectInst::Create(ZI->getOperand(0), AddOne(CI), CI);
    
    Value *XorLHS = 0; ConstantInt *XorRHS = 0;
    if (match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      uint32_t TySizeBits = I.getType()->getScalarSizeInBits();
      const APInt &RHSVal = CI->getValue();
      unsigned ExtendAmt = 0;
      // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
      // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
      if (XorRHS->getValue() == -RHSVal) {
        if (RHSVal.isPowerOf2())
          ExtendAmt = TySizeBits - RHSVal.logBase2() - 1;
        else if (XorRHS->getValue().isPowerOf2())
          ExtendAmt = TySizeBits - XorRHS->getValue().logBase2() - 1;
      }
      
      if (ExtendAmt) {
        APInt Mask = APInt::getHighBitsSet(TySizeBits, ExtendAmt);
        if (!MaskedValueIsZero(XorLHS, Mask))
          ExtendAmt = 0;
      }
      
      if (ExtendAmt) {
        Constant *ShAmt = ConstantInt::get(I.getType(), ExtendAmt);
        Value *NewShl = Builder->CreateShl(XorLHS, ShAmt, "sext");
        return BinaryOperator::CreateAShr(NewShl, ShAmt);
      }

      // If this is a xor that was canonicalized from a sub, turn it back into
      // a sub and fuse this add with it.
      if (LHS->hasOneUse() && (XorRHS->getValue()+1).isPowerOf2()) {
        IntegerType *IT = cast<IntegerType>(I.getType());
        APInt LHSKnownOne(IT->getBitWidth(), 0);
        APInt LHSKnownZero(IT->getBitWidth(), 0);
        ComputeMaskedBits(XorLHS, LHSKnownZero, LHSKnownOne);
        if ((XorRHS->getValue() | LHSKnownZero).isAllOnesValue())
          return BinaryOperator::CreateSub(ConstantExpr::getAdd(XorRHS, CI),
                                           XorLHS);
      }
    }
  }

  if (isa<Constant>(RHS) && isa<PHINode>(LHS))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;

  if (I.getType()->isIntegerTy(1))
    return BinaryOperator::CreateXor(LHS, RHS);

  // X + X --> X << 1
  if (LHS == RHS) {
    BinaryOperator *New =
      BinaryOperator::CreateShl(LHS, ConstantInt::get(I.getType(), 1));
    New->setHasNoSignedWrap(I.hasNoSignedWrap());
    New->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
    return New;
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castNegVal(LHS)) {
    if (!isa<Constant>(RHS))
      if (Value *RHSV = dyn_castNegVal(RHS)) {
        Value *NewAdd = Builder->CreateAdd(LHSV, RHSV, "sum");
        return BinaryOperator::CreateNeg(NewAdd);
      }
    
    return BinaryOperator::CreateSub(RHS, LHSV);
  }

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::CreateSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::CreateMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::CreateMul(X, ConstantExpr::getAdd(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::CreateMul(LHS, AddOne(C2));

  // A+B --> A|B iff A and B have no bits set in common.
  if (IntegerType *IT = dyn_cast<IntegerType>(I.getType())) {
    APInt LHSKnownOne(IT->getBitWidth(), 0);
    APInt LHSKnownZero(IT->getBitWidth(), 0);
    ComputeMaskedBits(LHS, LHSKnownZero, LHSKnownOne);
    if (LHSKnownZero != 0) {
      APInt RHSKnownOne(IT->getBitWidth(), 0);
      APInt RHSKnownZero(IT->getBitWidth(), 0);
      ComputeMaskedBits(RHS, RHSKnownZero, RHSKnownOne);
      
      // No bits in common -> bitwise or.
      if ((LHSKnownZero|RHSKnownZero).isAllOnesValue())
        return BinaryOperator::CreateOr(LHS, RHS);
    }
  }

  // W*X + Y*Z --> W * (X+Z)  iff W == Y
  {
    Value *W, *X, *Y, *Z;
    if (match(LHS, m_Mul(m_Value(W), m_Value(X))) &&
        match(RHS, m_Mul(m_Value(Y), m_Value(Z)))) {
      if (W != Y) {
        if (W == Z) {
          std::swap(Y, Z);
        } else if (Y == X) {
          std::swap(W, X);
        } else if (X == Z) {
          std::swap(Y, Z);
          std::swap(W, X);
        }
      }

      if (W == Y) {
        Value *NewAdd = Builder->CreateAdd(X, Z, LHS->getName());
        return BinaryOperator::CreateMul(W, NewAdd);
      }
    }
  }

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X))))    // ~X + C --> (C-1) - X
      return BinaryOperator::CreateSub(SubOne(CRHS), X);

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() &&
        match(LHS, m_And(m_Value(X), m_ConstantInt(C2))) &&
        CRHS->getValue() == (CRHS->getValue() & C2->getValue())) {
      // See if all bits from the first bit set in the Add RHS up are included
      // in the mask.  First, get the rightmost bit.
      const APInt &AddRHSV = CRHS->getValue();
      
      // Form a mask of all bits from the lowest bit added through the top.
      APInt AddRHSHighBits(~((AddRHSV & -AddRHSV)-1));

      // See if the and mask includes all of these bits.
      APInt AddRHSHighBitsAnd(AddRHSHighBits & C2->getValue());

      if (AddRHSHighBits == AddRHSHighBitsAnd) {
        // Okay, the xform is safe.  Insert the new add pronto.
        Value *NewAdd = Builder->CreateAdd(X, CRHS, LHS->getName());
        return BinaryOperator::CreateAnd(NewAdd, C2);
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
  }

  // add (select X 0 (sub n A)) A  -->  select X A n
  {
    SelectInst *SI = dyn_cast<SelectInst>(LHS);
    Value *A = RHS;
    if (!SI) {
      SI = dyn_cast<SelectInst>(RHS);
      A = LHS;
    }
    if (SI && SI->hasOneUse()) {
      Value *TV = SI->getTrueValue();
      Value *FV = SI->getFalseValue();
      Value *N;

      // Can we fold the add into the argument of the select?
      // We check both true and false select arguments for a matching subtract.
      if (match(FV, m_Zero()) && match(TV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the true select value.
        return SelectInst::Create(SI->getCondition(), N, A);
      
      if (match(TV, m_Zero()) && match(FV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the false select value.
        return SelectInst::Create(SI->getCondition(), A, N);
    }
  }

  // Check for (add (sext x), y), see if we can merge this into an
  // integer add followed by a sext.
  if (SExtInst *LHSConv = dyn_cast<SExtInst>(LHS)) {
    // (add (sext x), cst) --> (sext (add x, cst'))
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS)) {
      Constant *CI = 
        ConstantExpr::getTrunc(RHSC, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSExt(CI, I.getType()) == RHSC &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new, smaller add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              CI, "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
    
    // (add (sext x), (sext y)) --> (sext (add int x, y))
    if (SExtInst *RHSConv = dyn_cast<SExtInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of sexts), and if the
      // integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                             RHSConv->getOperand(0), "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
  }

  // Check for (x & y) + (x ^ y)
  {
    Value *A = 0, *B = 0;
    if (match(RHS, m_Xor(m_Value(A), m_Value(B))) &&
        (match(LHS, m_And(m_Specific(A), m_Specific(B))) ||
         match(LHS, m_And(m_Specific(B), m_Specific(A)))))
      return BinaryOperator::CreateOr(A, B);

    if (match(LHS, m_Xor(m_Value(A), m_Value(B))) &&
        (match(RHS, m_And(m_Specific(A), m_Specific(B))) ||
         match(RHS, m_And(m_Specific(B), m_Specific(A)))))
      return BinaryOperator::CreateOr(A, B);
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitFAdd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + 0 --> X
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(ConstantFP::getNegativeZero
                              (I.getType())->getValueAPF()))
        return ReplaceInstUsesWith(I, LHS);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castFNegVal(LHS))
    return BinaryOperator::CreateFSub(RHS, LHSV);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castFNegVal(RHS))
      return BinaryOperator::CreateFSub(LHS, V);

  // Check for X+0.0.  Simplify it to X if we know X is not -0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS))
    if (CFP->getValueAPF().isPosZero() && CannotBeNegativeZero(LHS))
      return ReplaceInstUsesWith(I, LHS);

  // Check for (fadd double (sitofp x), y), see if we can merge this into an
  // integer add followed by a promotion.
  if (SIToFPInst *LHSConv = dyn_cast<SIToFPInst>(LHS)) {
    // (fadd double (sitofp x), fpcst) --> (sitofp (add int x, intcst))
    // ... if the constant fits in the integer value.  This is useful for things
    // like (double)(x & 1234) + 4.0 -> (double)((X & 1234)+4) which no longer
    // requires a constant pool load, and generally allows the add to be better
    // instcombined.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS)) {
      Constant *CI = 
      ConstantExpr::getFPToSI(CFP, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSIToFP(CI, I.getType()) == CFP &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0),
                                              CI, "addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
    
    // (fadd double (sitofp x), (sitofp y)) --> (sitofp (add int x, y))
    if (SIToFPInst *RHSConv = dyn_cast<SIToFPInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of int->fp conversions),
      // and if the integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              RHSConv->getOperand(0),"addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
  }
  
  return Changed ? &I : 0;
}


/// Optimize pointer differences into the same array into a size.  Consider:
///  &A[10] - &A[0]: we should compile this to "10".  LHS/RHS are the pointer
/// operands to the ptrtoint instructions for the LHS/RHS of the subtract.
///
Value *InstCombiner::OptimizePointerDifference(Value *LHS, Value *RHS,
                                               Type *Ty) {
  assert(TD && "Must have target data info for this");
  
  // If LHS is a gep based on RHS or RHS is a gep based on LHS, we can optimize
  // this.
  bool Swapped = false;
  GEPOperator *GEP1 = 0, *GEP2 = 0;

  // For now we require one side to be the base pointer "A" or a constant
  // GEP derived from it.
  if (GEPOperator *LHSGEP = dyn_cast<GEPOperator>(LHS)) {
    // (gep X, ...) - X
    if (LHSGEP->getOperand(0) == RHS) {
      GEP1 = LHSGEP;
      Swapped = false;
    } else if (GEPOperator *RHSGEP = dyn_cast<GEPOperator>(RHS)) {
      // (gep X, ...) - (gep X, ...)
      if (LHSGEP->getOperand(0)->stripPointerCasts() ==
            RHSGEP->getOperand(0)->stripPointerCasts()) {
        GEP2 = RHSGEP;
        GEP1 = LHSGEP;
        Swapped = false;
      }
    }
  }
  
  if (GEPOperator *RHSGEP = dyn_cast<GEPOperator>(RHS)) {
    // X - (gep X, ...)
    if (RHSGEP->getOperand(0) == LHS) {
      GEP1 = RHSGEP;
      Swapped = true;
    } else if (GEPOperator *LHSGEP = dyn_cast<GEPOperator>(LHS)) {
      // (gep X, ...) - (gep X, ...)
      if (RHSGEP->getOperand(0)->stripPointerCasts() ==
            LHSGEP->getOperand(0)->stripPointerCasts()) {
        GEP2 = LHSGEP;
        GEP1 = RHSGEP;
        Swapped = true;
      }
    }
  }
  
  // Avoid duplicating the arithmetic if GEP2 has non-constant indices and
  // multiple users.
  if (GEP1 == 0 ||
      (GEP2 != 0 && !GEP2->hasAllConstantIndices() && !GEP2->hasOneUse()))
    return 0;
  
  // Emit the offset of the GEP and an intptr_t.
  Value *Result = EmitGEPOffset(GEP1);
  
  // If we had a constant expression GEP on the other side offsetting the
  // pointer, subtract it from the offset we have.
  if (GEP2) {
    Value *Offset = EmitGEPOffset(GEP2);
    Result = Builder->CreateSub(Result, Offset);
  }

  // If we have p - gep(p, ...)  then we have to negate the result.
  if (Swapped)
    Result = Builder->CreateNeg(Result, "diff.neg");

  return Builder->CreateIntCast(Result, Ty, true);
}


Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifySubInst(Op0, Op1, I.hasNoSignedWrap(),
                                 I.hasNoUnsignedWrap(), TD))
    return ReplaceInstUsesWith(I, V);

  // (A*B)-(A*C) -> A*(B-C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // If this is a 'B = x-(-A)', change to B = x+A.  This preserves NSW/NUW.
  if (Value *V = dyn_castNegVal(Op1)) {
    BinaryOperator *Res = BinaryOperator::CreateAdd(Op0, V);
    Res->setHasNoSignedWrap(I.hasNoSignedWrap());
    Res->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
    return Res;
  }

  if (I.getType()->isIntegerTy(1))
    return BinaryOperator::CreateXor(Op0, Op1);

  // Replace (-1 - A) with (~A).
  if (match(Op0, m_AllOnes()))
    return BinaryOperator::CreateNot(Op1);
  
  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::CreateAdd(X, AddOne(C));

    // -(X >>u 31) -> (X >>s 31)
    // -(X >>s 31) -> (X >>u 31)
    if (C->isZero()) {
      Value *X; ConstantInt *CI;
      if (match(Op1, m_LShr(m_Value(X), m_ConstantInt(CI))) &&
          // Verify we are shifting out everything but the sign bit.
          CI->getValue() == I.getType()->getPrimitiveSizeInBits()-1)
        return BinaryOperator::CreateAShr(X, CI);

      if (match(Op1, m_AShr(m_Value(X), m_ConstantInt(CI))) &&
          // Verify we are shifting out everything but the sign bit.
          CI->getValue() == I.getType()->getPrimitiveSizeInBits()-1)
        return BinaryOperator::CreateLShr(X, CI);
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    // C-(X+C2) --> (C-C2)-X
    ConstantInt *C2;
    if (match(Op1, m_Add(m_Value(X), m_ConstantInt(C2))))
      return BinaryOperator::CreateSub(ConstantExpr::getSub(C, C2), X);

    if (SimplifyDemandedInstructionBits(I))
      return &I;
  }

  
  { Value *Y;
    // X-(X+Y) == -Y    X-(Y+X) == -Y
    if (match(Op1, m_Add(m_Specific(Op0), m_Value(Y))) ||
        match(Op1, m_Add(m_Value(Y), m_Specific(Op0))))
      return BinaryOperator::CreateNeg(Y);
    
    // (X-Y)-X == -Y
    if (match(Op0, m_Sub(m_Specific(Op1), m_Value(Y))))
      return BinaryOperator::CreateNeg(Y);
  }
  
  if (Op1->hasOneUse()) {
    Value *X = 0, *Y = 0, *Z = 0;
    Constant *C = 0;
    ConstantInt *CI = 0;

    // (X - (Y - Z))  -->  (X + (Z - Y)).
    if (match(Op1, m_Sub(m_Value(Y), m_Value(Z))))
      return BinaryOperator::CreateAdd(Op0,
                                      Builder->CreateSub(Z, Y, Op1->getName()));

    // (X - (X & Y))   -->   (X & ~Y)
    //
    if (match(Op1, m_And(m_Value(Y), m_Specific(Op0))) ||
        match(Op1, m_And(m_Specific(Op0), m_Value(Y))))
      return BinaryOperator::CreateAnd(Op0,
                                  Builder->CreateNot(Y, Y->getName() + ".not"));
    
    // 0 - (X sdiv C)  -> (X sdiv -C)
    if (match(Op1, m_SDiv(m_Value(X), m_Constant(C))) &&
        match(Op0, m_Zero()))
      return BinaryOperator::CreateSDiv(X, ConstantExpr::getNeg(C));

    // 0 - (X << Y)  -> (-X << Y)   when X is freely negatable.
    if (match(Op1, m_Shl(m_Value(X), m_Value(Y))) && match(Op0, m_Zero()))
      if (Value *XNeg = dyn_castNegVal(X))
        return BinaryOperator::CreateShl(XNeg, Y);

    // X - X*C --> X * (1-C)
    if (match(Op1, m_Mul(m_Specific(Op0), m_ConstantInt(CI)))) {
      Constant *CP1 = ConstantExpr::getSub(ConstantInt::get(I.getType(),1), CI);
      return BinaryOperator::CreateMul(Op0, CP1);
    }

    // X - X<<C --> X * (1-(1<<C))
    if (match(Op1, m_Shl(m_Specific(Op0), m_ConstantInt(CI)))) {
      Constant *One = ConstantInt::get(I.getType(), 1);
      C = ConstantExpr::getSub(One, ConstantExpr::getShl(One, CI));
      return BinaryOperator::CreateMul(Op0, C);
    }
    
    // X - A*-B -> X + A*B
    // X - -A*B -> X + A*B
    Value *A, *B;
    if (match(Op1, m_Mul(m_Value(A), m_Neg(m_Value(B)))) ||
        match(Op1, m_Mul(m_Neg(m_Value(A)), m_Value(B))))
      return BinaryOperator::CreateAdd(Op0, Builder->CreateMul(A, B));
      
    // X - A*CI -> X + A*-CI
    // X - CI*A -> X + A*-CI
    if (match(Op1, m_Mul(m_Value(A), m_ConstantInt(CI))) ||
        match(Op1, m_Mul(m_ConstantInt(CI), m_Value(A)))) {
      Value *NewMul = Builder->CreateMul(A, ConstantExpr::getNeg(CI));
      return BinaryOperator::CreateAdd(Op0, NewMul);
    }
  }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1)  // X*C - X --> X * (C-1)
      return BinaryOperator::CreateMul(Op1, SubOne(C1));

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::CreateMul(X, ConstantExpr::getSub(C1, C2));
  }
  
  // Optimize pointer differences into the same array into a size.  Consider:
  //  &A[10] - &A[0]: we should compile this to "10".
  if (TD) {
    Value *LHSOp, *RHSOp;
    if (match(Op0, m_PtrToInt(m_Value(LHSOp))) &&
        match(Op1, m_PtrToInt(m_Value(RHSOp))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
    
    // trunc(p)-trunc(q) -> trunc(p-q)
    if (match(Op0, m_Trunc(m_PtrToInt(m_Value(LHSOp)))) &&
        match(Op1, m_Trunc(m_PtrToInt(m_Value(RHSOp)))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
  }
  
  return 0;
}

Instruction *InstCombiner::visitFSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castFNegVal(Op1))
    return BinaryOperator::CreateFAdd(Op0, V);

  return 0;
}
