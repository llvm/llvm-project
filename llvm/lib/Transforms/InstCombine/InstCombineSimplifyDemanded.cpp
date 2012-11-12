//===- InstCombineSimplifyDemanded.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains logic for simplifying instructions based on information
// about how they are used.
//
//===----------------------------------------------------------------------===//


#include "InstCombine.h"
#include "llvm/DataLayout.h"
#include "llvm/IntrinsicInst.h"

using namespace llvm;


/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
static bool ShrinkDemandedConstant(Instruction *I, unsigned OpNo, 
                                   APInt Demanded) {
  assert(I && "No instruction?");
  assert(OpNo < I->getNumOperands() && "Operand index too large");

  // If the operand is not a constant integer, nothing to do.
  ConstantInt *OpC = dyn_cast<ConstantInt>(I->getOperand(OpNo));
  if (!OpC) return false;

  // If there are no bits set that aren't demanded, nothing to do.
  Demanded = Demanded.zextOrTrunc(OpC->getValue().getBitWidth());
  if ((~Demanded & OpC->getValue()) == 0)
    return false;

  // This instruction is producing bits that are not demanded. Shrink the RHS.
  Demanded &= OpC->getValue();
  I->setOperand(OpNo, ConstantInt::get(OpC->getType(), Demanded));
  return true;
}



/// SimplifyDemandedInstructionBits - Inst is an integer instruction that
/// SimplifyDemandedBits knows about.  See if the instruction has any
/// properties that allow us to simplify its operands.
bool InstCombiner::SimplifyDemandedInstructionBits(Instruction &Inst) {
  unsigned BitWidth = Inst.getType()->getScalarSizeInBits();
  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  APInt DemandedMask(APInt::getAllOnesValue(BitWidth));
  
  Value *V = SimplifyDemandedUseBits(&Inst, DemandedMask, 
                                     KnownZero, KnownOne, 0);
  if (V == 0) return false;
  if (V == &Inst) return true;
  ReplaceInstUsesWith(Inst, V);
  return true;
}

/// SimplifyDemandedBits - This form of SimplifyDemandedBits simplifies the
/// specified instruction operand if possible, updating it in place.  It returns
/// true if it made any change and false otherwise.
bool InstCombiner::SimplifyDemandedBits(Use &U, APInt DemandedMask, 
                                        APInt &KnownZero, APInt &KnownOne,
                                        unsigned Depth) {
  Value *NewVal = SimplifyDemandedUseBits(U.get(), DemandedMask,
                                          KnownZero, KnownOne, Depth);
  if (NewVal == 0) return false;
  U = NewVal;
  return true;
}


/// SimplifyDemandedUseBits - This function attempts to replace V with a simpler
/// value based on the demanded bits.  When this function is called, it is known
/// that only the bits set in DemandedMask of the result of V are ever used
/// downstream. Consequently, depending on the mask and V, it may be possible
/// to replace V with a constant or one of its operands. In such cases, this
/// function does the replacement and returns true. In all other cases, it
/// returns false after analyzing the expression and setting KnownOne and known
/// to be one in the expression.  KnownZero contains all the bits that are known
/// to be zero in the expression. These are provided to potentially allow the
/// caller (which might recursively be SimplifyDemandedBits itself) to simplify
/// the expression. KnownOne and KnownZero always follow the invariant that 
/// KnownOne & KnownZero == 0. That is, a bit can't be both 1 and 0. Note that
/// the bits in KnownOne and KnownZero may only be accurate for those bits set
/// in DemandedMask. Note also that the bitwidth of V, DemandedMask, KnownZero
/// and KnownOne must all be the same.
///
/// This returns null if it did not change anything and it permits no
/// simplification.  This returns V itself if it did some simplification of V's
/// operands based on the information about what bits are demanded. This returns
/// some other non-null value if it found out that V is equal to another value
/// in the context where the specified bits are demanded, but not for all users.
Value *InstCombiner::SimplifyDemandedUseBits(Value *V, APInt DemandedMask,
                                             APInt &KnownZero, APInt &KnownOne,
                                             unsigned Depth) {
  assert(V != 0 && "Null pointer of Value???");
  assert(Depth <= 6 && "Limit Search Depth");
  uint32_t BitWidth = DemandedMask.getBitWidth();
  Type *VTy = V->getType();
  assert((TD || !VTy->isPointerTy()) &&
         "SimplifyDemandedBits needs to know bit widths!");
  assert((!TD || TD->getTypeSizeInBits(VTy->getScalarType()) == BitWidth) &&
         (!VTy->isIntOrIntVectorTy() ||
          VTy->getScalarSizeInBits() == BitWidth) &&
         KnownZero.getBitWidth() == BitWidth &&
         KnownOne.getBitWidth() == BitWidth &&
         "Value *V, DemandedMask, KnownZero and KnownOne "
         "must have same BitWidth");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getValue() & DemandedMask;
    KnownZero = ~KnownOne & DemandedMask;
    return 0;
  }
  if (isa<ConstantPointerNull>(V)) {
    // We know all of the bits for a constant!
    KnownOne.clearAllBits();
    KnownZero = DemandedMask;
    return 0;
  }

  KnownZero.clearAllBits();
  KnownOne.clearAllBits();
  if (DemandedMask == 0) {   // Not demanding any bits from V.
    if (isa<UndefValue>(V))
      return 0;
    return UndefValue::get(VTy);
  }
  
  if (Depth == 6)        // Limit search depth.
    return 0;
  
  APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
  APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    ComputeMaskedBits(V, KnownZero, KnownOne, Depth);
    return 0;        // Only analyze instructions.
  }

  // If there are multiple uses of this value and we aren't at the root, then
  // we can't do any simplifications of the operands, because DemandedMask
  // only reflects the bits demanded by *one* of the users.
  if (Depth != 0 && !I->hasOneUse()) {
    // Despite the fact that we can't simplify this instruction in all User's
    // context, we can at least compute the knownzero/knownone bits, and we can
    // do simplifications that apply to *just* the one user if we know that
    // this instruction has a simpler value in that context.
    if (I->getOpcode() == Instruction::And) {
      // If either the LHS or the RHS are Zero, the result is zero.
      ComputeMaskedBits(I->getOperand(1), RHSKnownZero, RHSKnownOne, Depth+1);
      ComputeMaskedBits(I->getOperand(0), LHSKnownZero, LHSKnownOne, Depth+1);
      
      // If all of the demanded bits are known 1 on one side, return the other.
      // These bits cannot contribute to the result of the 'and' in this
      // context.
      if ((DemandedMask & ~LHSKnownZero & RHSKnownOne) == 
          (DemandedMask & ~LHSKnownZero))
        return I->getOperand(0);
      if ((DemandedMask & ~RHSKnownZero & LHSKnownOne) == 
          (DemandedMask & ~RHSKnownZero))
        return I->getOperand(1);
      
      // If all of the demanded bits in the inputs are known zeros, return zero.
      if ((DemandedMask & (RHSKnownZero|LHSKnownZero)) == DemandedMask)
        return Constant::getNullValue(VTy);
      
    } else if (I->getOpcode() == Instruction::Or) {
      // We can simplify (X|Y) -> X or Y in the user's context if we know that
      // only bits from X or Y are demanded.
      
      // If either the LHS or the RHS are One, the result is One.
      ComputeMaskedBits(I->getOperand(1), RHSKnownZero, RHSKnownOne, Depth+1);
      ComputeMaskedBits(I->getOperand(0), LHSKnownZero, LHSKnownOne, Depth+1);
      
      // If all of the demanded bits are known zero on one side, return the
      // other.  These bits cannot contribute to the result of the 'or' in this
      // context.
      if ((DemandedMask & ~LHSKnownOne & RHSKnownZero) == 
          (DemandedMask & ~LHSKnownOne))
        return I->getOperand(0);
      if ((DemandedMask & ~RHSKnownOne & LHSKnownZero) == 
          (DemandedMask & ~RHSKnownOne))
        return I->getOperand(1);
      
      // If all of the potentially set bits on one side are known to be set on
      // the other side, just use the 'other' side.
      if ((DemandedMask & (~RHSKnownZero) & LHSKnownOne) == 
          (DemandedMask & (~RHSKnownZero)))
        return I->getOperand(0);
      if ((DemandedMask & (~LHSKnownZero) & RHSKnownOne) == 
          (DemandedMask & (~LHSKnownZero)))
        return I->getOperand(1);
    }
    
    // Compute the KnownZero/KnownOne bits to simplify things downstream.
    ComputeMaskedBits(I, KnownZero, KnownOne, Depth);
    return 0;
  }
  
  // If this is the root being simplified, allow it to have multiple uses,
  // just set the DemandedMask to all bits so that we can try to simplify the
  // operands.  This allows visitTruncInst (for example) to simplify the
  // operand of a trunc without duplicating all the logic below.
  if (Depth == 0 && !V->hasOneUse())
    DemandedMask = APInt::getAllOnesValue(BitWidth);
  
  switch (I->getOpcode()) {
  default:
    ComputeMaskedBits(I, KnownZero, KnownOne, Depth);
    break;
  case Instruction::And:
    // If either the LHS or the RHS are Zero, the result is zero.
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask & ~RHSKnownZero,
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 

    // If all of the demanded bits are known 1 on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((DemandedMask & ~LHSKnownZero & RHSKnownOne) == 
        (DemandedMask & ~LHSKnownZero))
      return I->getOperand(0);
    if ((DemandedMask & ~RHSKnownZero & LHSKnownOne) == 
        (DemandedMask & ~RHSKnownZero))
      return I->getOperand(1);
    
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((DemandedMask & (RHSKnownZero|LHSKnownZero)) == DemandedMask)
      return Constant::getNullValue(VTy);
      
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask & ~LHSKnownZero))
      return I;
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne = RHSKnownOne & LHSKnownOne;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero = RHSKnownZero | LHSKnownZero;
    break;
  case Instruction::Or:
    // If either the LHS or the RHS are One, the result is One.
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask, 
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask & ~RHSKnownOne, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((DemandedMask & ~LHSKnownOne & RHSKnownZero) == 
        (DemandedMask & ~LHSKnownOne))
      return I->getOperand(0);
    if ((DemandedMask & ~RHSKnownOne & LHSKnownZero) == 
        (DemandedMask & ~RHSKnownOne))
      return I->getOperand(1);

    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((DemandedMask & (~RHSKnownZero) & LHSKnownOne) == 
        (DemandedMask & (~RHSKnownZero)))
      return I->getOperand(0);
    if ((DemandedMask & (~LHSKnownZero) & RHSKnownOne) == 
        (DemandedMask & (~LHSKnownZero)))
      return I->getOperand(1);
        
    // If the RHS is a constant, see if we can simplify it.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return I;
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero = RHSKnownZero & LHSKnownZero;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne = RHSKnownOne | LHSKnownOne;
    break;
  case Instruction::Xor: {
    if (SimplifyDemandedBits(I->getOperandUse(1), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(0), DemandedMask, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((DemandedMask & RHSKnownZero) == DemandedMask)
      return I->getOperand(0);
    if ((DemandedMask & LHSKnownZero) == DemandedMask)
      return I->getOperand(1);
    
    // If all of the demanded bits are known to be zero on one side or the
    // other, turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((DemandedMask & ~RHSKnownZero & ~LHSKnownZero) == 0) {
      Instruction *Or = 
        BinaryOperator::CreateOr(I->getOperand(0), I->getOperand(1),
                                 I->getName());
      return InsertNewInstWith(Or, *I);
    }
    
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((DemandedMask & (RHSKnownZero|RHSKnownOne)) == DemandedMask) { 
      // all known
      if ((RHSKnownOne & LHSKnownOne) == RHSKnownOne) {
        Constant *AndC = Constant::getIntegerValue(VTy,
                                                   ~RHSKnownOne & DemandedMask);
        Instruction *And = BinaryOperator::CreateAnd(I->getOperand(0), AndC);
        return InsertNewInstWith(And, *I);
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // FIXME: for XOR, we prefer to force bits to 1 if they will make a -1.
    if (ShrinkDemandedConstant(I, 1, DemandedMask))
      return I;
    
    // If our LHS is an 'and' and if it has one use, and if any of the bits we
    // are flipping are known to be set, then the xor is just resetting those
    // bits to zero.  We can just knock out bits from the 'and' and the 'xor',
    // simplifying both of them.
    if (Instruction *LHSInst = dyn_cast<Instruction>(I->getOperand(0)))
      if (LHSInst->getOpcode() == Instruction::And && LHSInst->hasOneUse() &&
          isa<ConstantInt>(I->getOperand(1)) &&
          isa<ConstantInt>(LHSInst->getOperand(1)) &&
          (LHSKnownOne & RHSKnownOne & DemandedMask) != 0) {
        ConstantInt *AndRHS = cast<ConstantInt>(LHSInst->getOperand(1));
        ConstantInt *XorRHS = cast<ConstantInt>(I->getOperand(1));
        APInt NewMask = ~(LHSKnownOne & RHSKnownOne & DemandedMask);
        
        Constant *AndC =
          ConstantInt::get(I->getType(), NewMask & AndRHS->getValue());
        Instruction *NewAnd = BinaryOperator::CreateAnd(I->getOperand(0), AndC);
        InsertNewInstWith(NewAnd, *I);
        
        Constant *XorC =
          ConstantInt::get(I->getType(), NewMask & XorRHS->getValue());
        Instruction *NewXor = BinaryOperator::CreateXor(NewAnd, XorC);
        return InsertNewInstWith(NewXor, *I);
      }

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZero= (RHSKnownZero & LHSKnownZero) | (RHSKnownOne & LHSKnownOne);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (RHSKnownZero & LHSKnownOne) | (RHSKnownOne & LHSKnownZero);
    break;
  }
  case Instruction::Select:
    if (SimplifyDemandedBits(I->getOperandUse(2), DemandedMask,
                             RHSKnownZero, RHSKnownOne, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(1), DemandedMask, 
                             LHSKnownZero, LHSKnownOne, Depth+1))
      return I;
    assert(!(RHSKnownZero & RHSKnownOne) && "Bits known to be one AND zero?"); 
    assert(!(LHSKnownZero & LHSKnownOne) && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (ShrinkDemandedConstant(I, 1, DemandedMask) ||
        ShrinkDemandedConstant(I, 2, DemandedMask))
      return I;
    
    // Only known if known in both the LHS and RHS.
    KnownOne = RHSKnownOne & LHSKnownOne;
    KnownZero = RHSKnownZero & LHSKnownZero;
    break;
  case Instruction::Trunc: {
    unsigned truncBf = I->getOperand(0)->getType()->getScalarSizeInBits();
    DemandedMask = DemandedMask.zext(truncBf);
    KnownZero = KnownZero.zext(truncBf);
    KnownOne = KnownOne.zext(truncBf);
    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask, 
                             KnownZero, KnownOne, Depth+1))
      return I;
    DemandedMask = DemandedMask.trunc(BitWidth);
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);
    assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
    break;
  }
  case Instruction::BitCast:
    if (!I->getOperand(0)->getType()->isIntOrIntVectorTy())
      return 0;  // vector->int or fp->int?

    if (VectorType *DstVTy = dyn_cast<VectorType>(I->getType())) {
      if (VectorType *SrcVTy =
            dyn_cast<VectorType>(I->getOperand(0)->getType())) {
        if (DstVTy->getNumElements() != SrcVTy->getNumElements())
          // Don't touch a bitcast between vectors of different element counts.
          return 0;
      } else
        // Don't touch a scalar-to-vector bitcast.
        return 0;
    } else if (I->getOperand(0)->getType()->isVectorTy())
      // Don't touch a vector-to-scalar bitcast.
      return 0;

    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask,
                             KnownZero, KnownOne, Depth+1))
      return I;
    assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
    break;
  case Instruction::ZExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth =I->getOperand(0)->getType()->getScalarSizeInBits();
    
    DemandedMask = DemandedMask.trunc(SrcBitWidth);
    KnownZero = KnownZero.trunc(SrcBitWidth);
    KnownOne = KnownOne.trunc(SrcBitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMask,
                             KnownZero, KnownOne, Depth+1))
      return I;
    DemandedMask = DemandedMask.zext(BitWidth);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
    // The top bits are known to be zero.
    KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth =I->getOperand(0)->getType()->getScalarSizeInBits();
    
    APInt InputDemandedBits = DemandedMask & 
                              APInt::getLowBitsSet(BitWidth, SrcBitWidth);

    APInt NewBits(APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth));
    // If any of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    if ((NewBits & DemandedMask) != 0)
      InputDemandedBits.setBit(SrcBitWidth-1);
      
    InputDemandedBits = InputDemandedBits.trunc(SrcBitWidth);
    KnownZero = KnownZero.trunc(SrcBitWidth);
    KnownOne = KnownOne.trunc(SrcBitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), InputDemandedBits,
                             KnownZero, KnownOne, Depth+1))
      return I;
    InputDemandedBits = InputDemandedBits.zext(BitWidth);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
      
    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.

    // If the input sign bit is known zero, or if the NewBits are not demanded
    // convert this into a zero extension.
    if (KnownZero[SrcBitWidth-1] || (NewBits & ~DemandedMask) == NewBits) {
      // Convert to ZExt cast
      CastInst *NewCast = new ZExtInst(I->getOperand(0), VTy, I->getName());
      return InsertNewInstWith(NewCast, *I);
    } else if (KnownOne[SrcBitWidth-1]) {    // Input sign bit known set
      KnownOne |= NewBits;
    }
    break;
  }
  case Instruction::Add: {
    // Figure out what the input bits are.  If the top bits of the and result
    // are not demanded, then the add doesn't demand them from its input
    // either.
    unsigned NLZ = DemandedMask.countLeadingZeros();
      
    // If there is a constant on the RHS, there are a variety of xformations
    // we can do.
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // If null, this should be simplified elsewhere.  Some of the xforms here
      // won't work if the RHS is zero.
      if (RHS->isZero())
        break;
      
      // If the top bit of the output is demanded, demand everything from the
      // input.  Otherwise, we demand all the input bits except NLZ top bits.
      APInt InDemandedBits(APInt::getLowBitsSet(BitWidth, BitWidth - NLZ));

      // Find information about known zero/one bits in the input.
      if (SimplifyDemandedBits(I->getOperandUse(0), InDemandedBits, 
                               LHSKnownZero, LHSKnownOne, Depth+1))
        return I;

      // If the RHS of the add has bits set that can't affect the input, reduce
      // the constant.
      if (ShrinkDemandedConstant(I, 1, InDemandedBits))
        return I;
      
      // Avoid excess work.
      if (LHSKnownZero == 0 && LHSKnownOne == 0)
        break;
      
      // Turn it into OR if input bits are zero.
      if ((LHSKnownZero & RHS->getValue()) == RHS->getValue()) {
        Instruction *Or =
          BinaryOperator::CreateOr(I->getOperand(0), I->getOperand(1),
                                   I->getName());
        return InsertNewInstWith(Or, *I);
      }
      
      // We can say something about the output known-zero and known-one bits,
      // depending on potential carries from the input constant and the
      // unknowns.  For example if the LHS is known to have at most the 0x0F0F0
      // bits set and the RHS constant is 0x01001, then we know we have a known
      // one mask of 0x00001 and a known zero mask of 0xE0F0E.
      
      // To compute this, we first compute the potential carry bits.  These are
      // the bits which may be modified.  I'm not aware of a better way to do
      // this scan.
      const APInt &RHSVal = RHS->getValue();
      APInt CarryBits((~LHSKnownZero + RHSVal) ^ (~LHSKnownZero ^ RHSVal));
      
      // Now that we know which bits have carries, compute the known-1/0 sets.
      
      // Bits are known one if they are known zero in one operand and one in the
      // other, and there is no input carry.
      KnownOne = ((LHSKnownZero & RHSVal) | 
                  (LHSKnownOne & ~RHSVal)) & ~CarryBits;
      
      // Bits are known zero if they are known zero in both operands and there
      // is no input carry.
      KnownZero = LHSKnownZero & ~RHSVal & ~CarryBits;
    } else {
      // If the high-bits of this ADD are not demanded, then it does not demand
      // the high bits of its LHS or RHS.
      if (DemandedMask[BitWidth-1] == 0) {
        // Right fill the mask of bits for this ADD to demand the most
        // significant bit and all those below it.
        APInt DemandedFromOps(APInt::getLowBitsSet(BitWidth, BitWidth-NLZ));
        if (SimplifyDemandedBits(I->getOperandUse(0), DemandedFromOps,
                                 LHSKnownZero, LHSKnownOne, Depth+1) ||
            SimplifyDemandedBits(I->getOperandUse(1), DemandedFromOps,
                                 LHSKnownZero, LHSKnownOne, Depth+1))
          return I;
      }
    }
    break;
  }
  case Instruction::Sub:
    // If the high-bits of this SUB are not demanded, then it does not demand
    // the high bits of its LHS or RHS.
    if (DemandedMask[BitWidth-1] == 0) {
      // Right fill the mask of bits for this SUB to demand the most
      // significant bit and all those below it.
      uint32_t NLZ = DemandedMask.countLeadingZeros();
      APInt DemandedFromOps(APInt::getLowBitsSet(BitWidth, BitWidth-NLZ));
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedFromOps,
                               LHSKnownZero, LHSKnownOne, Depth+1) ||
          SimplifyDemandedBits(I->getOperandUse(1), DemandedFromOps,
                               LHSKnownZero, LHSKnownOne, Depth+1))
        return I;
    }

    // Otherwise just hand the sub off to ComputeMaskedBits to fill in
    // the known zeros and ones.
    ComputeMaskedBits(V, KnownZero, KnownOne, Depth);

    // Turn this into a xor if LHS is 2^n-1 and the remaining bits are known
    // zero.
    if (ConstantInt *C0 = dyn_cast<ConstantInt>(I->getOperand(0))) {
      APInt I0 = C0->getValue();
      if ((I0 + 1).isPowerOf2() && (I0 | KnownZero).isAllOnesValue()) {
        Instruction *Xor = BinaryOperator::CreateXor(I->getOperand(1), C0);
        return InsertNewInstWith(Xor, *I);
      }
    }
    break;
  case Instruction::Shl:
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth-1);
      APInt DemandedMaskIn(DemandedMask.lshr(ShiftAmt));
      
      // If the shift is NUW/NSW, then it does demand the high bits.
      ShlOperator *IOp = cast<ShlOperator>(I);
      if (IOp->hasNoSignedWrap())
        DemandedMaskIn |= APInt::getHighBitsSet(BitWidth, ShiftAmt+1);
      else if (IOp->hasNoUnsignedWrap())
        DemandedMaskIn |= APInt::getHighBitsSet(BitWidth, ShiftAmt);
      
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn, 
                               KnownZero, KnownOne, Depth+1))
        return I;
      assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?");
      KnownZero <<= ShiftAmt;
      KnownOne  <<= ShiftAmt;
      // low bits known zero.
      if (ShiftAmt)
        KnownZero |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
    }
    break;
  case Instruction::LShr:
    // For a logical shift right
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth-1);
      
      // Unsigned shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));
      
      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<LShrOperator>(I)->isExact())
        DemandedMaskIn |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
      
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn,
                               KnownZero, KnownOne, Depth+1))
        return I;
      assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?");
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
      if (ShiftAmt) {
        // Compute the new bits that are at the top now.
        APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
        KnownZero |= HighBits;  // high bits known zero.
      }
    }
    break;
  case Instruction::AShr:
    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (DemandedMask == 1) {
      // Perform the logical shift right.
      Instruction *NewVal = BinaryOperator::CreateLShr(
                        I->getOperand(0), I->getOperand(1), I->getName());
      return InsertNewInstWith(NewVal, *I);
    }    

    // If the sign bit is the only bit demanded by this ashr, then there is no
    // need to do it, the shift doesn't change the high bit.
    if (DemandedMask.isSignBit())
      return I->getOperand(0);
    
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint32_t ShiftAmt = SA->getLimitedValue(BitWidth-1);
      
      // Signed shift right.
      APInt DemandedMaskIn(DemandedMask.shl(ShiftAmt));
      // If any of the "high bits" are demanded, we should set the sign bit as
      // demanded.
      if (DemandedMask.countLeadingZeros() <= ShiftAmt)
        DemandedMaskIn.setBit(BitWidth-1);
      
      // If the shift is exact, then it does demand the low bits (and knows that
      // they are zero).
      if (cast<AShrOperator>(I)->isExact())
        DemandedMaskIn |= APInt::getLowBitsSet(BitWidth, ShiftAmt);
      
      if (SimplifyDemandedBits(I->getOperandUse(0), DemandedMaskIn,
                               KnownZero, KnownOne, Depth+1))
        return I;
      assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?");
      // Compute the new bits that are at the top now.
      APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
        
      // Handle the sign bits.
      APInt SignBit(APInt::getSignBit(BitWidth));
      // Adjust to where it is now in the mask.
      SignBit = APIntOps::lshr(SignBit, ShiftAmt);  
        
      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if (BitWidth <= ShiftAmt || KnownZero[BitWidth-ShiftAmt-1] || 
          (HighBits & ~DemandedMask) == HighBits) {
        // Perform the logical shift right.
        BinaryOperator *NewVal = BinaryOperator::CreateLShr(I->getOperand(0),
                                                            SA, I->getName());
        NewVal->setIsExact(cast<BinaryOperator>(I)->isExact());
        return InsertNewInstWith(NewVal, *I);
      } else if ((KnownOne & SignBit) != 0) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    break;
  case Instruction::SRem:
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // X % -1 demands all the bits because we don't want to introduce
      // INT_MIN % -1 (== undef) by accident.
      if (Rem->isAllOnesValue())
        break;
      APInt RA = Rem->getValue().abs();
      if (RA.isPowerOf2()) {
        if (DemandedMask.ult(RA))    // srem won't affect demanded bits
          return I->getOperand(0);

        APInt LowBits = RA - 1;
        APInt Mask2 = LowBits | APInt::getSignBit(BitWidth);
        if (SimplifyDemandedBits(I->getOperandUse(0), Mask2,
                                 LHSKnownZero, LHSKnownOne, Depth+1))
          return I;

        // The low bits of LHS are unchanged by the srem.
        KnownZero = LHSKnownZero & LowBits;
        KnownOne = LHSKnownOne & LowBits;

        // If LHS is non-negative or has all low bits zero, then the upper bits
        // are all zero.
        if (LHSKnownZero[BitWidth-1] || ((LHSKnownZero & LowBits) == LowBits))
          KnownZero |= ~LowBits;

        // If LHS is negative and not all low bits are zero, then the upper bits
        // are all one.
        if (LHSKnownOne[BitWidth-1] && ((LHSKnownOne & LowBits) != 0))
          KnownOne |= ~LowBits;

        assert(!(KnownZero & KnownOne) && "Bits known to be one AND zero?"); 
      }
    }

    // The sign bit is the LHS's sign bit, except when the result of the
    // remainder is zero.
    if (DemandedMask.isNegative() && KnownZero.isNonNegative()) {
      APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
      ComputeMaskedBits(I->getOperand(0), LHSKnownZero, LHSKnownOne, Depth+1);
      // If it's known zero, our sign bit is also zero.
      if (LHSKnownZero.isNegative())
        KnownZero |= LHSKnownZero;
    }
    break;
  case Instruction::URem: {
    APInt KnownZero2(BitWidth, 0), KnownOne2(BitWidth, 0);
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    if (SimplifyDemandedBits(I->getOperandUse(0), AllOnes,
                             KnownZero2, KnownOne2, Depth+1) ||
        SimplifyDemandedBits(I->getOperandUse(1), AllOnes,
                             KnownZero2, KnownOne2, Depth+1))
      return I;

    unsigned Leaders = KnownZero2.countLeadingOnes();
    Leaders = std::max(Leaders,
                       KnownZero2.countLeadingOnes());
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders) & DemandedMask;
    break;
  }
  case Instruction::Call:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::bswap: {
        // If the only bits demanded come from one byte of the bswap result,
        // just shift the input byte into position to eliminate the bswap.
        unsigned NLZ = DemandedMask.countLeadingZeros();
        unsigned NTZ = DemandedMask.countTrailingZeros();
          
        // Round NTZ down to the next byte.  If we have 11 trailing zeros, then
        // we need all the bits down to bit 8.  Likewise, round NLZ.  If we
        // have 14 leading zeros, round to 8.
        NLZ &= ~7;
        NTZ &= ~7;
        // If we need exactly one byte, we can do this transformation.
        if (BitWidth-NLZ-NTZ == 8) {
          unsigned ResultBit = NTZ;
          unsigned InputBit = BitWidth-NTZ-8;
          
          // Replace this with either a left or right shift to get the byte into
          // the right place.
          Instruction *NewVal;
          if (InputBit > ResultBit)
            NewVal = BinaryOperator::CreateLShr(II->getArgOperand(0),
                    ConstantInt::get(I->getType(), InputBit-ResultBit));
          else
            NewVal = BinaryOperator::CreateShl(II->getArgOperand(0),
                    ConstantInt::get(I->getType(), ResultBit-InputBit));
          NewVal->takeName(I);
          return InsertNewInstWith(NewVal, *I);
        }
          
        // TODO: Could compute known zero/one bits based on the input.
        break;
      }
      case Intrinsic::x86_sse42_crc32_64_8:
      case Intrinsic::x86_sse42_crc32_64_64:
        KnownZero = APInt::getHighBitsSet(64, 32);
        return 0;
      }
    }
    ComputeMaskedBits(V, KnownZero, KnownOne, Depth);
    break;
  }
  
  // If the client is only demanding bits that we know, return the known
  // constant.
  if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask)
    return Constant::getIntegerValue(VTy, KnownOne);
  return 0;
}


/// SimplifyDemandedVectorElts - The specified value produces a vector with
/// any number of elements. DemandedElts contains the set of elements that are
/// actually used by the caller.  This method analyzes which elements of the
/// operand are undef and returns that information in UndefElts.
///
/// If the information about demanded elements can be used to simplify the
/// operation, the operation is simplified, then the resultant value is
/// returned.  This returns null if no change was made.
Value *InstCombiner::SimplifyDemandedVectorElts(Value *V, APInt DemandedElts,
                                                APInt &UndefElts,
                                                unsigned Depth) {
  unsigned VWidth = cast<VectorType>(V->getType())->getNumElements();
  APInt EltMask(APInt::getAllOnesValue(VWidth));
  assert((DemandedElts & ~EltMask) == 0 && "Invalid DemandedElts!");

  if (isa<UndefValue>(V)) {
    // If the entire vector is undefined, just return this info.
    UndefElts = EltMask;
    return 0;
  }
  
  if (DemandedElts == 0) { // If nothing is demanded, provide undef.
    UndefElts = EltMask;
    return UndefValue::get(V->getType());
  }

  UndefElts = 0;
  
  // Handle ConstantAggregateZero, ConstantVector, ConstantDataSequential.
  if (Constant *C = dyn_cast<Constant>(V)) {
    // Check if this is identity. If so, return 0 since we are not simplifying
    // anything.
    if (DemandedElts.isAllOnesValue())
      return 0;

    Type *EltTy = cast<VectorType>(V->getType())->getElementType();
    Constant *Undef = UndefValue::get(EltTy);
    
    SmallVector<Constant*, 16> Elts;
    for (unsigned i = 0; i != VWidth; ++i) {
      if (!DemandedElts[i]) {   // If not demanded, set to undef.
        Elts.push_back(Undef);
        UndefElts.setBit(i);
        continue;
      }
      
      Constant *Elt = C->getAggregateElement(i);
      if (Elt == 0) return 0;
      
      if (isa<UndefValue>(Elt)) {   // Already undef.
        Elts.push_back(Undef);
        UndefElts.setBit(i);
      } else {                               // Otherwise, defined.
        Elts.push_back(Elt);
      }
    }
    
    // If we changed the constant, return it.
    Constant *NewCV = ConstantVector::get(Elts);
    return NewCV != C ? NewCV : 0;
  }
  
  // Limit search depth.
  if (Depth == 10)
    return 0;

  // If multiple users are using the root value, proceed with
  // simplification conservatively assuming that all elements
  // are needed.
  if (!V->hasOneUse()) {
    // Quit if we find multiple users of a non-root value though.
    // They'll be handled when it's their turn to be visited by
    // the main instcombine process.
    if (Depth != 0)
      // TODO: Just compute the UndefElts information recursively.
      return 0;

    // Conservatively assume that all elements are needed.
    DemandedElts = EltMask;
  }
  
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return 0;        // Only analyze instructions.
  
  bool MadeChange = false;
  APInt UndefElts2(VWidth, 0);
  Value *TmpV;
  switch (I->getOpcode()) {
  default: break;
    
  case Instruction::InsertElement: {
    // If this is a variable index, we don't know which element it overwrites.
    // demand exactly the same input as we produce.
    ConstantInt *Idx = dyn_cast<ConstantInt>(I->getOperand(2));
    if (Idx == 0) {
      // Note that we can't propagate undef elt info, because we don't know
      // which elt is getting updated.
      TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts,
                                        UndefElts2, Depth+1);
      if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }
      break;
    }
    
    // If this is inserting an element that isn't demanded, remove this
    // insertelement.
    unsigned IdxNo = Idx->getZExtValue();
    if (IdxNo >= VWidth || !DemandedElts[IdxNo]) {
      Worklist.Add(I);
      return I->getOperand(0);
    }
    
    // Otherwise, the element inserted overwrites whatever was there, so the
    // input demanded set is simpler than the output set.
    APInt DemandedElts2 = DemandedElts;
    DemandedElts2.clearBit(IdxNo);
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts2,
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }

    // The inserted element is defined.
    UndefElts.clearBit(IdxNo);
    break;
  }
  case Instruction::ShuffleVector: {
    ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
    uint64_t LHSVWidth =
      cast<VectorType>(Shuffle->getOperand(0)->getType())->getNumElements();
    APInt LeftDemanded(LHSVWidth, 0), RightDemanded(LHSVWidth, 0);
    for (unsigned i = 0; i < VWidth; i++) {
      if (DemandedElts[i]) {
        unsigned MaskVal = Shuffle->getMaskValue(i);
        if (MaskVal != -1u) {
          assert(MaskVal < LHSVWidth * 2 &&
                 "shufflevector mask index out of range!");
          if (MaskVal < LHSVWidth)
            LeftDemanded.setBit(MaskVal);
          else
            RightDemanded.setBit(MaskVal - LHSVWidth);
        }
      }
    }

    APInt UndefElts4(LHSVWidth, 0);
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), LeftDemanded,
                                      UndefElts4, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }

    APInt UndefElts3(LHSVWidth, 0);
    TmpV = SimplifyDemandedVectorElts(I->getOperand(1), RightDemanded,
                                      UndefElts3, Depth+1);
    if (TmpV) { I->setOperand(1, TmpV); MadeChange = true; }

    bool NewUndefElts = false;
    for (unsigned i = 0; i < VWidth; i++) {
      unsigned MaskVal = Shuffle->getMaskValue(i);
      if (MaskVal == -1u) {
        UndefElts.setBit(i);
      } else if (!DemandedElts[i]) {
        NewUndefElts = true;
        UndefElts.setBit(i);
      } else if (MaskVal < LHSVWidth) {
        if (UndefElts4[MaskVal]) {
          NewUndefElts = true;
          UndefElts.setBit(i);
        }
      } else {
        if (UndefElts3[MaskVal - LHSVWidth]) {
          NewUndefElts = true;
          UndefElts.setBit(i);
        }
      }
    }

    if (NewUndefElts) {
      // Add additional discovered undefs.
      SmallVector<Constant*, 16> Elts;
      for (unsigned i = 0; i < VWidth; ++i) {
        if (UndefElts[i])
          Elts.push_back(UndefValue::get(Type::getInt32Ty(I->getContext())));
        else
          Elts.push_back(ConstantInt::get(Type::getInt32Ty(I->getContext()),
                                          Shuffle->getMaskValue(i)));
      }
      I->setOperand(2, ConstantVector::get(Elts));
      MadeChange = true;
    }
    break;
  }
  case Instruction::Select: {
    APInt LeftDemanded(DemandedElts), RightDemanded(DemandedElts);
    if (ConstantVector* CV = dyn_cast<ConstantVector>(I->getOperand(0))) {
      for (unsigned i = 0; i < VWidth; i++) {
        if (CV->getAggregateElement(i)->isNullValue())
          LeftDemanded.clearBit(i);
        else
          RightDemanded.clearBit(i);
      }
    }

    TmpV = SimplifyDemandedVectorElts(I->getOperand(1), LeftDemanded,
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(1, TmpV); MadeChange = true; }

    TmpV = SimplifyDemandedVectorElts(I->getOperand(2), RightDemanded,
                                      UndefElts2, Depth+1);
    if (TmpV) { I->setOperand(2, TmpV); MadeChange = true; }
      
    // Output elements are undefined if both are undefined.
    UndefElts &= UndefElts2;
    break;
  }
  case Instruction::BitCast: {
    // Vector->vector casts only.
    VectorType *VTy = dyn_cast<VectorType>(I->getOperand(0)->getType());
    if (!VTy) break;
    unsigned InVWidth = VTy->getNumElements();
    APInt InputDemandedElts(InVWidth, 0);
    unsigned Ratio;

    if (VWidth == InVWidth) {
      // If we are converting from <4 x i32> -> <4 x f32>, we demand the same
      // elements as are demanded of us.
      Ratio = 1;
      InputDemandedElts = DemandedElts;
    } else if (VWidth > InVWidth) {
      // Untested so far.
      break;
      
      // If there are more elements in the result than there are in the source,
      // then an input element is live if any of the corresponding output
      // elements are live.
      Ratio = VWidth/InVWidth;
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx) {
        if (DemandedElts[OutIdx])
          InputDemandedElts.setBit(OutIdx/Ratio);
      }
    } else {
      // Untested so far.
      break;
      
      // If there are more elements in the source than there are in the result,
      // then an input element is live if the corresponding output element is
      // live.
      Ratio = InVWidth/VWidth;
      for (unsigned InIdx = 0; InIdx != InVWidth; ++InIdx)
        if (DemandedElts[InIdx/Ratio])
          InputDemandedElts.setBit(InIdx);
    }
    
    // div/rem demand all inputs, because they don't want divide by zero.
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), InputDemandedElts,
                                      UndefElts2, Depth+1);
    if (TmpV) {
      I->setOperand(0, TmpV);
      MadeChange = true;
    }
    
    UndefElts = UndefElts2;
    if (VWidth > InVWidth) {
      llvm_unreachable("Unimp");
      // If there are more elements in the result than there are in the source,
      // then an output element is undef if the corresponding input element is
      // undef.
      for (unsigned OutIdx = 0; OutIdx != VWidth; ++OutIdx)
        if (UndefElts2[OutIdx/Ratio])
          UndefElts.setBit(OutIdx);
    } else if (VWidth < InVWidth) {
      llvm_unreachable("Unimp");
      // If there are more elements in the source than there are in the result,
      // then a result element is undef if all of the corresponding input
      // elements are undef.
      UndefElts = ~0ULL >> (64-VWidth);  // Start out all undef.
      for (unsigned InIdx = 0; InIdx != InVWidth; ++InIdx)
        if (!UndefElts2[InIdx])            // Not undef?
          UndefElts.clearBit(InIdx/Ratio);    // Clear undef bit.
    }
    break;
  }
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // div/rem demand all inputs, because they don't want divide by zero.
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts,
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }
    TmpV = SimplifyDemandedVectorElts(I->getOperand(1), DemandedElts,
                                      UndefElts2, Depth+1);
    if (TmpV) { I->setOperand(1, TmpV); MadeChange = true; }
      
    // Output elements are undefined if both are undefined.  Consider things
    // like undef&0.  The result is known zero, not undef.
    UndefElts &= UndefElts2;
    break;
  case Instruction::FPTrunc:
  case Instruction::FPExt:
    TmpV = SimplifyDemandedVectorElts(I->getOperand(0), DemandedElts,
                                      UndefElts, Depth+1);
    if (TmpV) { I->setOperand(0, TmpV); MadeChange = true; }
    break;
    
  case Instruction::Call: {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
    if (!II) break;
    switch (II->getIntrinsicID()) {
    default: break;
      
    // Binary vector operations that work column-wise.  A dest element is a
    // function of the corresponding input elements from the two inputs.
    case Intrinsic::x86_sse_sub_ss:
    case Intrinsic::x86_sse_mul_ss:
    case Intrinsic::x86_sse_min_ss:
    case Intrinsic::x86_sse_max_ss:
    case Intrinsic::x86_sse2_sub_sd:
    case Intrinsic::x86_sse2_mul_sd:
    case Intrinsic::x86_sse2_min_sd:
    case Intrinsic::x86_sse2_max_sd:
      TmpV = SimplifyDemandedVectorElts(II->getArgOperand(0), DemandedElts,
                                        UndefElts, Depth+1);
      if (TmpV) { II->setArgOperand(0, TmpV); MadeChange = true; }
      TmpV = SimplifyDemandedVectorElts(II->getArgOperand(1), DemandedElts,
                                        UndefElts2, Depth+1);
      if (TmpV) { II->setArgOperand(1, TmpV); MadeChange = true; }

      // If only the low elt is demanded and this is a scalarizable intrinsic,
      // scalarize it now.
      if (DemandedElts == 1) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::x86_sse_sub_ss:
        case Intrinsic::x86_sse_mul_ss:
        case Intrinsic::x86_sse2_sub_sd:
        case Intrinsic::x86_sse2_mul_sd:
          // TODO: Lower MIN/MAX/ABS/etc
          Value *LHS = II->getArgOperand(0);
          Value *RHS = II->getArgOperand(1);
          // Extract the element as scalars.
          LHS = InsertNewInstWith(ExtractElementInst::Create(LHS, 
            ConstantInt::get(Type::getInt32Ty(I->getContext()), 0U)), *II);
          RHS = InsertNewInstWith(ExtractElementInst::Create(RHS,
            ConstantInt::get(Type::getInt32Ty(I->getContext()), 0U)), *II);
          
          switch (II->getIntrinsicID()) {
          default: llvm_unreachable("Case stmts out of sync!");
          case Intrinsic::x86_sse_sub_ss:
          case Intrinsic::x86_sse2_sub_sd:
            TmpV = InsertNewInstWith(BinaryOperator::CreateFSub(LHS, RHS,
                                                        II->getName()), *II);
            break;
          case Intrinsic::x86_sse_mul_ss:
          case Intrinsic::x86_sse2_mul_sd:
            TmpV = InsertNewInstWith(BinaryOperator::CreateFMul(LHS, RHS,
                                                         II->getName()), *II);
            break;
          }
          
          Instruction *New =
            InsertElementInst::Create(
              UndefValue::get(II->getType()), TmpV,
              ConstantInt::get(Type::getInt32Ty(I->getContext()), 0U, false),
                                      II->getName());
          InsertNewInstWith(New, *II);
          return New;
        }            
      }
        
      // Output elements are undefined if both are undefined.  Consider things
      // like undef&0.  The result is known zero, not undef.
      UndefElts &= UndefElts2;
      break;
    }
    break;
  }
  }
  return MadeChange ? I : 0;
}
