//===- InstCombineShifts.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitShl, visitLShr, and visitAShr functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Support/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

Instruction *InstCombiner::commonShiftTransforms(BinaryOperator &I) {
  assert(I.getOperand(1)->getType() == I.getOperand(0)->getType());
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // See if we can fold away this shift.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;

  // X shift (A srem B) -> X shift (A and B-1) iff B is a power of 2.
  // Because shifts by negative values (which could occur if A were negative)
  // are undefined.
  Value *A; const APInt *B;
  if (Op1->hasOneUse() && match(Op1, m_SRem(m_Value(A), m_Power2(B)))) {
    // FIXME: Should this get moved into SimplifyDemandedBits by saying we don't
    // demand the sign bit (and many others) here??
    Value *Rem = Builder->CreateAnd(A, ConstantInt::get(I.getType(), *B-1),
                                    Op1->getName());
    I.setOperand(1, Rem);
    return &I;
  }
  
  return 0;
}

/// CanEvaluateShifted - See if we can compute the specified value, but shifted
/// logically to the left or right by some number of bits.  This should return
/// true if the expression can be computed for the same cost as the current
/// expression tree.  This is used to eliminate extraneous shifting from things
/// like:
///      %C = shl i128 %A, 64
///      %D = shl i128 %B, 96
///      %E = or i128 %C, %D
///      %F = lshr i128 %E, 64
/// where the client will ask if E can be computed shifted right by 64-bits.  If
/// this succeeds, the GetShiftedValue function will be called to produce the
/// value.
static bool CanEvaluateShifted(Value *V, unsigned NumBits, bool isLeftShift,
                               InstCombiner &IC) {
  // We can always evaluate constants shifted.
  if (isa<Constant>(V))
    return true;
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  
  // If this is the opposite shift, we can directly reuse the input of the shift
  // if the needed bits are already zero in the input.  This allows us to reuse
  // the value which means that we don't care if the shift has multiple uses.
  //  TODO:  Handle opposite shift by exact value.
  ConstantInt *CI = 0;
  if ((isLeftShift && match(I, m_LShr(m_Value(), m_ConstantInt(CI)))) ||
      (!isLeftShift && match(I, m_Shl(m_Value(), m_ConstantInt(CI))))) {
    if (CI->getZExtValue() == NumBits) {
      // TODO: Check that the input bits are already zero with MaskedValueIsZero
#if 0
      // If this is a truncate of a logical shr, we can truncate it to a smaller
      // lshr iff we know that the bits we would otherwise be shifting in are
      // already zeros.
      uint32_t OrigBitWidth = OrigTy->getScalarSizeInBits();
      uint32_t BitWidth = Ty->getScalarSizeInBits();
      if (MaskedValueIsZero(I->getOperand(0),
            APInt::getHighBitsSet(OrigBitWidth, OrigBitWidth-BitWidth)) &&
          CI->getLimitedValue(BitWidth) < BitWidth) {
        return CanEvaluateTruncated(I->getOperand(0), Ty);
      }
#endif
      
    }
  }
  
  // We can't mutate something that has multiple uses: doing so would
  // require duplicating the instruction in general, which isn't profitable.
  if (!I->hasOneUse()) return false;
  
  switch (I->getOpcode()) {
  default: return false;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // Bitwise operators can all arbitrarily be arbitrarily evaluated shifted.
    return CanEvaluateShifted(I->getOperand(0), NumBits, isLeftShift, IC) &&
           CanEvaluateShifted(I->getOperand(1), NumBits, isLeftShift, IC);
      
  case Instruction::Shl: {
    // We can often fold the shift into shifts-by-a-constant.
    CI = dyn_cast<ConstantInt>(I->getOperand(1));
    if (CI == 0) return false;

    // We can always fold shl(c1)+shl(c2) -> shl(c1+c2).
    if (isLeftShift) return true;
    
    // We can always turn shl(c)+shr(c) -> and(c2).
    if (CI->getValue() == NumBits) return true;
      
    unsigned TypeWidth = I->getType()->getScalarSizeInBits();

    // We can turn shl(c1)+shr(c2) -> shl(c3)+and(c4), but it isn't
    // profitable unless we know the and'd out bits are already zero.
    if (CI->getZExtValue() > NumBits) {
      unsigned LowBits = TypeWidth - CI->getZExtValue();
      if (MaskedValueIsZero(I->getOperand(0),
                       APInt::getLowBitsSet(TypeWidth, NumBits) << LowBits))
        return true;
    }
      
    return false;
  }
  case Instruction::LShr: {
    // We can often fold the shift into shifts-by-a-constant.
    CI = dyn_cast<ConstantInt>(I->getOperand(1));
    if (CI == 0) return false;
    
    // We can always fold lshr(c1)+lshr(c2) -> lshr(c1+c2).
    if (!isLeftShift) return true;
    
    // We can always turn lshr(c)+shl(c) -> and(c2).
    if (CI->getValue() == NumBits) return true;
      
    unsigned TypeWidth = I->getType()->getScalarSizeInBits();

    // We can always turn lshr(c1)+shl(c2) -> lshr(c3)+and(c4), but it isn't
    // profitable unless we know the and'd out bits are already zero.
    if (CI->getValue().ult(TypeWidth) && CI->getZExtValue() > NumBits) {
      unsigned LowBits = CI->getZExtValue() - NumBits;
      if (MaskedValueIsZero(I->getOperand(0),
                          APInt::getLowBitsSet(TypeWidth, NumBits) << LowBits))
        return true;
    }
      
    return false;
  }
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    return CanEvaluateShifted(SI->getTrueValue(), NumBits, isLeftShift, IC) &&
           CanEvaluateShifted(SI->getFalseValue(), NumBits, isLeftShift, IC);
  }
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!CanEvaluateShifted(PN->getIncomingValue(i), NumBits, isLeftShift,IC))
        return false;
    return true;
  }
  }      
}

/// GetShiftedValue - When CanEvaluateShifted returned true for an expression,
/// this value inserts the new computation that produces the shifted value.
static Value *GetShiftedValue(Value *V, unsigned NumBits, bool isLeftShift,
                              InstCombiner &IC) {
  // We can always evaluate constants shifted.
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (isLeftShift)
      V = IC.Builder->CreateShl(C, NumBits);
    else
      V = IC.Builder->CreateLShr(C, NumBits);
    // If we got a constantexpr back, try to simplify it with TD info.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      V = ConstantFoldConstantExpression(CE, IC.getDataLayout(),
                                         IC.getTargetLibraryInfo());
    return V;
  }
  
  Instruction *I = cast<Instruction>(V);
  IC.Worklist.Add(I);

  switch (I->getOpcode()) {
  default: llvm_unreachable("Inconsistency with CanEvaluateShifted");
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // Bitwise operators can all arbitrarily be arbitrarily evaluated shifted.
    I->setOperand(0, GetShiftedValue(I->getOperand(0), NumBits,isLeftShift,IC));
    I->setOperand(1, GetShiftedValue(I->getOperand(1), NumBits,isLeftShift,IC));
    return I;
    
  case Instruction::Shl: {
    BinaryOperator *BO = cast<BinaryOperator>(I);
    unsigned TypeWidth = BO->getType()->getScalarSizeInBits();

    // We only accept shifts-by-a-constant in CanEvaluateShifted.
    ConstantInt *CI = cast<ConstantInt>(BO->getOperand(1));

    // We can always fold shl(c1)+shl(c2) -> shl(c1+c2).
    if (isLeftShift) {
      // If this is oversized composite shift, then unsigned shifts get 0.
      unsigned NewShAmt = NumBits+CI->getZExtValue();
      if (NewShAmt >= TypeWidth)
        return Constant::getNullValue(I->getType());

      BO->setOperand(1, ConstantInt::get(BO->getType(), NewShAmt));
      BO->setHasNoUnsignedWrap(false);
      BO->setHasNoSignedWrap(false);
      return I;
    }
    
    // We turn shl(c)+lshr(c) -> and(c2) if the input doesn't already have
    // zeros.
    if (CI->getValue() == NumBits) {
      APInt Mask(APInt::getLowBitsSet(TypeWidth, TypeWidth - NumBits));
      V = IC.Builder->CreateAnd(BO->getOperand(0),
                                ConstantInt::get(BO->getContext(), Mask));
      if (Instruction *VI = dyn_cast<Instruction>(V)) {
        VI->moveBefore(BO);
        VI->takeName(BO);
      }
      return V;
    }
    
    // We turn shl(c1)+shr(c2) -> shl(c3)+and(c4), but only when we know that
    // the and won't be needed.
    assert(CI->getZExtValue() > NumBits);
    BO->setOperand(1, ConstantInt::get(BO->getType(),
                                       CI->getZExtValue() - NumBits));
    BO->setHasNoUnsignedWrap(false);
    BO->setHasNoSignedWrap(false);
    return BO;
  }
  case Instruction::LShr: {
    BinaryOperator *BO = cast<BinaryOperator>(I);
    unsigned TypeWidth = BO->getType()->getScalarSizeInBits();
    // We only accept shifts-by-a-constant in CanEvaluateShifted.
    ConstantInt *CI = cast<ConstantInt>(BO->getOperand(1));
    
    // We can always fold lshr(c1)+lshr(c2) -> lshr(c1+c2).
    if (!isLeftShift) {
      // If this is oversized composite shift, then unsigned shifts get 0.
      unsigned NewShAmt = NumBits+CI->getZExtValue();
      if (NewShAmt >= TypeWidth)
        return Constant::getNullValue(BO->getType());
      
      BO->setOperand(1, ConstantInt::get(BO->getType(), NewShAmt));
      BO->setIsExact(false);
      return I;
    }
    
    // We turn lshr(c)+shl(c) -> and(c2) if the input doesn't already have
    // zeros.
    if (CI->getValue() == NumBits) {
      APInt Mask(APInt::getHighBitsSet(TypeWidth, TypeWidth - NumBits));
      V = IC.Builder->CreateAnd(I->getOperand(0),
                                ConstantInt::get(BO->getContext(), Mask));
      if (Instruction *VI = dyn_cast<Instruction>(V)) {
        VI->moveBefore(I);
        VI->takeName(I);
      }
      return V;
    }
    
    // We turn lshr(c1)+shl(c2) -> lshr(c3)+and(c4), but only when we know that
    // the and won't be needed.
    assert(CI->getZExtValue() > NumBits);
    BO->setOperand(1, ConstantInt::get(BO->getType(),
                                       CI->getZExtValue() - NumBits));
    BO->setIsExact(false);
    return BO;
  }
    
  case Instruction::Select:
    I->setOperand(1, GetShiftedValue(I->getOperand(1), NumBits,isLeftShift,IC));
    I->setOperand(2, GetShiftedValue(I->getOperand(2), NumBits,isLeftShift,IC));
    return I;
  case Instruction::PHI: {
    // We can change a phi if we can change all operands.  Note that we never
    // get into trouble with cyclic PHIs here because we only consider
    // instructions with a single use.
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      PN->setIncomingValue(i, GetShiftedValue(PN->getIncomingValue(i),
                                              NumBits, isLeftShift, IC));
    return PN;
  }
  }      
}



Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                               BinaryOperator &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;
  
  
  // See if we can propagate this shift into the input, this covers the trivial
  // cast of lshr(shl(x,c1),c2) as well as other more complex cases.
  if (I.getOpcode() != Instruction::AShr &&
      CanEvaluateShifted(Op0, Op1->getZExtValue(), isLeftShift, *this)) {
    DEBUG(dbgs() << "ICE: GetShiftedValue propagating shift through expression"
              " to eliminate shift:\n  IN: " << *Op0 << "\n  SH: " << I <<"\n");
    
    return ReplaceInstUsesWith(I, 
                 GetShiftedValue(Op0, Op1->getZExtValue(), isLeftShift, *this));
  }
  
  
  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint32_t TypeBits = Op0->getType()->getScalarSizeInBits();
  
  // shl i32 X, 32 = 0 and srl i8 Y, 9 = 0, ... just don't eliminate
  // a signed shift.
  //
  if (Op1->uge(TypeBits)) {
    if (I.getOpcode() != Instruction::AShr)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    // ashr i32 X, 32 --> ashr i32 X, 31
    I.setOperand(1, ConstantInt::get(I.getType(), TypeBits-1));
    return &I;
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::CreateMul(BO->getOperand(0),
                                        ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  // Fold shift2(trunc(shift1(x,c1)), c2) -> trunc(shift2(shift1(x,c1),c2))
  if (TruncInst *TI = dyn_cast<TruncInst>(Op0)) {
    Instruction *TrOp = dyn_cast<Instruction>(TI->getOperand(0));
    // If 'shift2' is an ashr, we would have to get the sign bit into a funny
    // place.  Don't try to do this transformation in this case.  Also, we
    // require that the input operand is a shift-by-constant so that we have
    // confidence that the shifts will get folded together.  We could do this
    // xform in more cases, but it is unlikely to be profitable.
    if (TrOp && I.isLogicalShift() && TrOp->isShift() && 
        isa<ConstantInt>(TrOp->getOperand(1))) {
      // Okay, we'll do this xform.  Make the shift of shift.
      Constant *ShAmt = ConstantExpr::getZExt(Op1, TrOp->getType());
      // (shift2 (shift1 & 0x00FF), c2)
      Value *NSh = Builder->CreateBinOp(I.getOpcode(), TrOp, ShAmt,I.getName());

      // For logical shifts, the truncation has the effect of making the high
      // part of the register be zeros.  Emulate this by inserting an AND to
      // clear the top bits as needed.  This 'and' will usually be zapped by
      // other xforms later if dead.
      unsigned SrcSize = TrOp->getType()->getScalarSizeInBits();
      unsigned DstSize = TI->getType()->getScalarSizeInBits();
      APInt MaskV(APInt::getLowBitsSet(SrcSize, DstSize));
      
      // The mask we constructed says what the trunc would do if occurring
      // between the shifts.  We want to know the effect *after* the second
      // shift.  We know that it is a logical shift by a constant, so adjust the
      // mask as appropriate.
      if (I.getOpcode() == Instruction::Shl)
        MaskV <<= Op1->getZExtValue();
      else {
        assert(I.getOpcode() == Instruction::LShr && "Unknown logical shift");
        MaskV = MaskV.lshr(Op1->getZExtValue());
      }

      // shift1 & 0x00FF
      Value *And = Builder->CreateAnd(NSh,
                                      ConstantInt::get(I.getContext(), MaskV),
                                      TI->getName());

      // Return the value truncated to the interesting size.
      return new TruncInst(And, I.getType());
    }
  }
  
  if (Op0->hasOneUse()) {
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1, *V2;
      ConstantInt *CC;
      switch (Op0BO->getOpcode()) {
      default: break;
      case Instruction::Add:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor: {
        // These operators commute.
        // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
            match(Op0BO->getOperand(1), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =         // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(0), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), YS, V1,
                                          Op0BO->getOperand(1)->getName());
          uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
          return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                     APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
        }
        
        // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
        Value *Op0BOOp1 = Op0BO->getOperand(1);
        if (isLeftShift && Op0BOOp1->hasOneUse() &&
            match(Op0BOOp1, 
                  m_And(m_Shr(m_Value(V1), m_Specific(Op1)),
                        m_ConstantInt(CC))) &&
            cast<BinaryOperator>(Op0BOOp1)->getOperand(0)->hasOneUse()) {
          Value *YS =   // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(0), Op1,
                                         Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                         V1->getName()+".mask");
          return BinaryOperator::Create(Op0BO->getOpcode(), YS, XM);
        }
      }
        
      // FALL THROUGH.
      case Instruction::Sub: {
        // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0), m_Shr(m_Value(V1),
                  m_Specific(Op1)))) {
          Value *YS =  // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // (X + (Y << C))
          Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), V1, YS,
                                          Op0BO->getOperand(0)->getName());
          uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
          return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                     APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
        }
        
        // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
        if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
            match(Op0BO->getOperand(0),
                  m_And(m_Shr(m_Value(V1), m_Value(V2)),
                        m_ConstantInt(CC))) && V2 == Op1 &&
            cast<BinaryOperator>(Op0BO->getOperand(0))
                ->getOperand(0)->hasOneUse()) {
          Value *YS = // (Y << C)
            Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
          // X & (CC << C)
          Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                         V1->getName()+".mask");
          
          return BinaryOperator::Create(Op0BO->getOpcode(), XM, YS);
        }
        
        break;
      }
      }
      
      
      // If the operand is an bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
        bool isValid = true;     // Valid only for And, Or, Xor
        bool highBitSet = false; // Transform if high bit of constant set?
        
        switch (Op0BO->getOpcode()) {
        default: isValid = false; break;   // Do not perform transform!
        case Instruction::Add:
          isValid = isLeftShift;
          break;
        case Instruction::Or:
        case Instruction::Xor:
          highBitSet = false;
          break;
        case Instruction::And:
          highBitSet = true;
          break;
        }
        
        // If this is a signed shift right, and the high bit is modified
        // by the logical operation, do not perform the transformation.
        // The highBitSet boolean indicates the value of the high bit of
        // the constant which would cause it to be modified for this
        // operation.
        //
        if (isValid && I.getOpcode() == Instruction::AShr)
          isValid = Op0C->getValue()[TypeBits-1] == highBitSet;
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Value *NewShift =
            Builder->CreateBinOp(I.getOpcode(), Op0BO->getOperand(0), Op1);
          NewShift->takeName(Op0BO);
          
          return BinaryOperator::Create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  BinaryOperator *ShiftOp = dyn_cast<BinaryOperator>(Op0);
  if (ShiftOp && !ShiftOp->isShift())
    ShiftOp = 0;
  
  if (ShiftOp && isa<ConstantInt>(ShiftOp->getOperand(1))) {

    // This is a constant shift of a constant shift. Be careful about hiding
    // shl instructions behind bit masks. They are used to represent multiplies
    // by a constant, and it is important that simple arithmetic expressions
    // are still recognizable by scalar evolution.
    //
    // The transforms applied to shl are very similar to the transforms applied
    // to mul by constant. We can be more aggressive about optimizing right
    // shifts.
    //
    // Combinations of right and left shifts will still be optimized in
    // DAGCombine where scalar evolution no longer applies.

    ConstantInt *ShiftAmt1C = cast<ConstantInt>(ShiftOp->getOperand(1));
    uint32_t ShiftAmt1 = ShiftAmt1C->getLimitedValue(TypeBits);
    uint32_t ShiftAmt2 = Op1->getLimitedValue(TypeBits);
    assert(ShiftAmt2 != 0 && "Should have been simplified earlier");
    if (ShiftAmt1 == 0) return 0;  // Will be simplified in the future.
    Value *X = ShiftOp->getOperand(0);
    
    IntegerType *Ty = cast<IntegerType>(I.getType());
    
    // Check for (X << c1) << c2  and  (X >> c1) >> c2
    if (I.getOpcode() == ShiftOp->getOpcode()) {
      uint32_t AmtSum = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
      // If this is oversized composite shift, then unsigned shifts get 0, ashr
      // saturates.
      if (AmtSum >= TypeBits) {
        if (I.getOpcode() != Instruction::AShr)
          return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
        AmtSum = TypeBits-1;  // Saturate to 31 for i32 ashr.
      }
      
      return BinaryOperator::Create(I.getOpcode(), X,
                                    ConstantInt::get(Ty, AmtSum));
    }
    
    if (ShiftAmt1 == ShiftAmt2) {
      // If we have ((X << C) >>u C), turn this into X & (-1 >>u C).
      if (I.getOpcode() == Instruction::LShr &&
          ShiftOp->getOpcode() == Instruction::Shl) {
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X,
                                        ConstantInt::get(I.getContext(), Mask));
      }
    } else if (ShiftAmt1 < ShiftAmt2) {
      uint32_t ShiftDiff = ShiftAmt2-ShiftAmt1;

      // (X >>?,exact C1) << C2 --> X << (C2-C1)
      // The inexact version is deferred to DAGCombine so we don't hide shl
      // behind a bit mask.
      if (I.getOpcode() == Instruction::Shl &&
          ShiftOp->getOpcode() != Instruction::Shl &&
          ShiftOp->isExact()) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
        BinaryOperator *NewShl = BinaryOperator::Create(Instruction::Shl,
                                                        X, ShiftDiffCst);
        NewShl->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
        NewShl->setHasNoSignedWrap(I.hasNoSignedWrap());
        return NewShl;
      }

      // (X << C1) >>u C2  --> X >>u (C2-C1) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr &&
          ShiftOp->getOpcode() == Instruction::Shl) {
        ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
        // (X <<nuw C1) >>u C2 --> X >>u (C2-C1)
        if (ShiftOp->hasNoUnsignedWrap()) {
          BinaryOperator *NewLShr = BinaryOperator::Create(Instruction::LShr,
                                                           X, ShiftDiffCst);
          NewLShr->setIsExact(I.isExact());
          return NewLShr;
        }
        Value *Shift = Builder->CreateLShr(X, ShiftDiffCst);
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }

      // We can't handle (X << C1) >>s C2, it shifts arbitrary bits in. However,
      // we can handle (X <<nsw C1) >>s C2 since it only shifts in sign bits.
      if (I.getOpcode() == Instruction::AShr &&
          ShiftOp->getOpcode() == Instruction::Shl) {
        if (ShiftOp->hasNoSignedWrap()) {
          // (X <<nsw C1) >>s C2 --> X >>s (C2-C1)
          ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
          BinaryOperator *NewAShr = BinaryOperator::Create(Instruction::AShr,
                                                           X, ShiftDiffCst);
          NewAShr->setIsExact(I.isExact());
          return NewAShr;
        }
      }
    } else {
      assert(ShiftAmt2 < ShiftAmt1);
      uint32_t ShiftDiff = ShiftAmt1-ShiftAmt2;

      // (X >>?exact C1) << C2 --> X >>?exact (C1-C2)
      // The inexact version is deferred to DAGCombine so we don't hide shl
      // behind a bit mask.
      if (I.getOpcode() == Instruction::Shl &&
          ShiftOp->getOpcode() != Instruction::Shl &&
          ShiftOp->isExact()) {
        ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
        BinaryOperator *NewShr = BinaryOperator::Create(ShiftOp->getOpcode(),
                                                        X, ShiftDiffCst);
        NewShr->setIsExact(true);
        return NewShr;
      }

      // (X << C1) >>u C2  --> X << (C1-C2) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr &&
          ShiftOp->getOpcode() == Instruction::Shl) {
        ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
        if (ShiftOp->hasNoUnsignedWrap()) {
          // (X <<nuw C1) >>u C2 --> X <<nuw (C1-C2)
          BinaryOperator *NewShl = BinaryOperator::Create(Instruction::Shl,
                                                          X, ShiftDiffCst);
          NewShl->setHasNoUnsignedWrap(true);
          return NewShl;
        }
        Value *Shift = Builder->CreateShl(X, ShiftDiffCst);
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // We can't handle (X << C1) >>s C2, it shifts arbitrary bits in. However,
      // we can handle (X <<nsw C1) >>s C2 since it only shifts in sign bits.
      if (I.getOpcode() == Instruction::AShr &&
          ShiftOp->getOpcode() == Instruction::Shl) {
        if (ShiftOp->hasNoSignedWrap()) {
          // (X <<nsw C1) >>s C2 --> X <<nsw (C1-C2)
          ConstantInt *ShiftDiffCst = ConstantInt::get(Ty, ShiftDiff);
          BinaryOperator *NewShl = BinaryOperator::Create(Instruction::Shl,
                                                          X, ShiftDiffCst);
          NewShl->setHasNoSignedWrap(true);
          return NewShl;
        }
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitShl(BinaryOperator &I) {
  if (Value *V = SimplifyShlInst(I.getOperand(0), I.getOperand(1),
                                 I.hasNoSignedWrap(), I.hasNoUnsignedWrap(),
                                 TD))
    return ReplaceInstUsesWith(I, V);
  
  if (Instruction *V = commonShiftTransforms(I))
    return V;
  
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(I.getOperand(1))) {
    unsigned ShAmt = Op1C->getZExtValue();
    
    // If the shifted-out value is known-zero, then this is a NUW shift.
    if (!I.hasNoUnsignedWrap() && 
        MaskedValueIsZero(I.getOperand(0),
                          APInt::getHighBitsSet(Op1C->getBitWidth(), ShAmt))) {
          I.setHasNoUnsignedWrap();
          return &I;
        }
    
    // If the shifted out value is all signbits, this is a NSW shift.
    if (!I.hasNoSignedWrap() &&
        ComputeNumSignBits(I.getOperand(0)) > ShAmt) {
      I.setHasNoSignedWrap();
      return &I;
    }
  }

  // (C1 << A) << C2 -> (C1 << C2) << A
  Constant *C1, *C2;
  Value *A;
  if (match(I.getOperand(0), m_OneUse(m_Shl(m_Constant(C1), m_Value(A)))) &&
      match(I.getOperand(1), m_Constant(C2)))
    return BinaryOperator::CreateShl(ConstantExpr::getShl(C1, C2), A);

  return 0;    
}

Instruction *InstCombiner::visitLShr(BinaryOperator &I) {
  if (Value *V = SimplifyLShrInst(I.getOperand(0), I.getOperand(1),
                                  I.isExact(), TD))
    return ReplaceInstUsesWith(I, V);

  if (Instruction *R = commonShiftTransforms(I))
    return R;
  
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    unsigned ShAmt = Op1C->getZExtValue();

    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Op0)) {
      unsigned BitWidth = Op0->getType()->getScalarSizeInBits();
      // ctlz.i32(x)>>5  --> zext(x == 0)
      // cttz.i32(x)>>5  --> zext(x == 0)
      // ctpop.i32(x)>>5 --> zext(x == -1)
      if ((II->getIntrinsicID() == Intrinsic::ctlz ||
           II->getIntrinsicID() == Intrinsic::cttz ||
           II->getIntrinsicID() == Intrinsic::ctpop) &&
          isPowerOf2_32(BitWidth) && Log2_32(BitWidth) == ShAmt) {
        bool isCtPop = II->getIntrinsicID() == Intrinsic::ctpop;
        Constant *RHS = ConstantInt::getSigned(Op0->getType(), isCtPop ? -1:0);
        Value *Cmp = Builder->CreateICmpEQ(II->getArgOperand(0), RHS);
        return new ZExtInst(Cmp, II->getType());
      }
    }
  
    // If the shifted-out value is known-zero, then this is an exact shift.
    if (!I.isExact() && 
        MaskedValueIsZero(Op0,APInt::getLowBitsSet(Op1C->getBitWidth(),ShAmt))){
      I.setIsExact();
      return &I;
    }    
  }
  
  return 0;
}

Instruction *InstCombiner::visitAShr(BinaryOperator &I) {
  if (Value *V = SimplifyAShrInst(I.getOperand(0), I.getOperand(1),
                                  I.isExact(), TD))
    return ReplaceInstUsesWith(I, V);

  if (Instruction *R = commonShiftTransforms(I))
    return R;
  
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    unsigned ShAmt = Op1C->getZExtValue();
    
    // If the input is a SHL by the same constant (ashr (shl X, C), C), then we
    // have a sign-extend idiom.
    Value *X;
    if (match(Op0, m_Shl(m_Value(X), m_Specific(Op1)))) {
      // If the left shift is just shifting out partial signbits, delete the
      // extension.
      if (cast<OverflowingBinaryOperator>(Op0)->hasNoSignedWrap())
        return ReplaceInstUsesWith(I, X);

      // If the input is an extension from the shifted amount value, e.g.
      //   %x = zext i8 %A to i32
      //   %y = shl i32 %x, 24
      //   %z = ashr %y, 24
      // then turn this into "z = sext i8 A to i32".
      if (ZExtInst *ZI = dyn_cast<ZExtInst>(X)) {
        uint32_t SrcBits = ZI->getOperand(0)->getType()->getScalarSizeInBits();
        uint32_t DestBits = ZI->getType()->getScalarSizeInBits();
        if (Op1C->getZExtValue() == DestBits-SrcBits)
          return new SExtInst(ZI->getOperand(0), ZI->getType());
      }
    }

    // If the shifted-out value is known-zero, then this is an exact shift.
    if (!I.isExact() && 
        MaskedValueIsZero(Op0,APInt::getLowBitsSet(Op1C->getBitWidth(),ShAmt))){
      I.setIsExact();
      return &I;
    }
  }            
  
  // See if we can turn a signed shr into an unsigned shr.
  if (MaskedValueIsZero(Op0,
                        APInt::getSignBit(I.getType()->getScalarSizeInBits())))
    return BinaryOperator::CreateLShr(Op0, Op1);
  
  // Arithmetic shifting an all-sign-bit value is a no-op.
  unsigned NumSignBits = ComputeNumSignBits(Op0);
  if (NumSignBits == Op0->getType()->getScalarSizeInBits())
    return ReplaceInstUsesWith(I, Op0);
  
  return 0;
}

