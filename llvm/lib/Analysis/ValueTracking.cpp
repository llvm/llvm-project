//===- ValueTracking.cpp - Walk computations to compute properties --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// computations have.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Operator.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cstring>
using namespace llvm;
using namespace llvm::PatternMatch;

const unsigned MaxDepth = 6;

/// getBitWidth - Returns the bitwidth of the given scalar or pointer type (if
/// unknown returns 0).  For vector types, returns the element type's bitwidth.
static unsigned getBitWidth(Type *Ty, const TargetData *TD) {
  if (unsigned BitWidth = Ty->getScalarSizeInBits())
    return BitWidth;
  assert(isa<PointerType>(Ty) && "Expected a pointer type!");
  return TD ? TD->getPointerSizeInBits() : 0;
}

static void ComputeMaskedBitsAddSub(bool Add, Value *Op0, Value *Op1, bool NSW,
                                    APInt &KnownZero, APInt &KnownOne,
                                    APInt &KnownZero2, APInt &KnownOne2,
                                    const TargetData *TD, unsigned Depth) {
  if (!Add) {
    if (ConstantInt *CLHS = dyn_cast<ConstantInt>(Op0)) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (!CLHS->getValue().isNegative()) {
        unsigned BitWidth = KnownZero.getBitWidth();
        unsigned NLZ = (CLHS->getValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        llvm::ComputeMaskedBits(Op1, KnownZero2, KnownOne2, TD, Depth+1);
    
        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((KnownZero2 & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getValue().countLeadingZeros();
          // Top bits known zero.
          KnownZero = APInt::getHighBitsSet(BitWidth, NLZ2);
        }
      }
    }
  }

  unsigned BitWidth = KnownZero.getBitWidth();

  // If one of the operands has trailing zeros, then the bits that the
  // other operand has in those bit positions will be preserved in the
  // result. For an add, this works with either operand. For a subtract,
  // this only works if the known zeros are in the right operand.
  APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
  llvm::ComputeMaskedBits(Op0, LHSKnownZero, LHSKnownOne, TD, Depth+1);
  assert((LHSKnownZero & LHSKnownOne) == 0 &&
         "Bits known to be one AND zero?");
  unsigned LHSKnownZeroOut = LHSKnownZero.countTrailingOnes();

  llvm::ComputeMaskedBits(Op1, KnownZero2, KnownOne2, TD, Depth+1);
  assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
  unsigned RHSKnownZeroOut = KnownZero2.countTrailingOnes();

  // Determine which operand has more trailing zeros, and use that
  // many bits from the other operand.
  if (LHSKnownZeroOut > RHSKnownZeroOut) {
    if (Add) {
      APInt Mask = APInt::getLowBitsSet(BitWidth, LHSKnownZeroOut);
      KnownZero |= KnownZero2 & Mask;
      KnownOne  |= KnownOne2 & Mask;
    } else {
      // If the known zeros are in the left operand for a subtract,
      // fall back to the minimum known zeros in both operands.
      KnownZero |= APInt::getLowBitsSet(BitWidth,
                                        std::min(LHSKnownZeroOut,
                                                 RHSKnownZeroOut));
    }
  } else if (RHSKnownZeroOut >= LHSKnownZeroOut) {
    APInt Mask = APInt::getLowBitsSet(BitWidth, RHSKnownZeroOut);
    KnownZero |= LHSKnownZero & Mask;
    KnownOne  |= LHSKnownOne & Mask;
  }

  // Are we still trying to solve for the sign bit?
  if (!KnownZero.isNegative() && !KnownOne.isNegative()) {
    if (NSW) {
      if (Add) {
        // Adding two positive numbers can't wrap into negative
        if (LHSKnownZero.isNegative() && KnownZero2.isNegative())
          KnownZero |= APInt::getSignBit(BitWidth);
        // and adding two negative numbers can't wrap into positive.
        else if (LHSKnownOne.isNegative() && KnownOne2.isNegative())
          KnownOne |= APInt::getSignBit(BitWidth);
      } else {
        // Subtracting a negative number from a positive one can't wrap
        if (LHSKnownZero.isNegative() && KnownOne2.isNegative())
          KnownZero |= APInt::getSignBit(BitWidth);
        // neither can subtracting a positive number from a negative one.
        else if (LHSKnownOne.isNegative() && KnownZero2.isNegative())
          KnownOne |= APInt::getSignBit(BitWidth);
      }
    }
  }
}

static void ComputeMaskedBitsMul(Value *Op0, Value *Op1, bool NSW,
                                 APInt &KnownZero, APInt &KnownOne,
                                 APInt &KnownZero2, APInt &KnownOne2,
                                 const TargetData *TD, unsigned Depth) {
  unsigned BitWidth = KnownZero.getBitWidth();
  ComputeMaskedBits(Op1, KnownZero, KnownOne, TD, Depth+1);
  ComputeMaskedBits(Op0, KnownZero2, KnownOne2, TD, Depth+1);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
  assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

  bool isKnownNegative = false;
  bool isKnownNonNegative = false;
  // If the multiplication is known not to overflow, compute the sign bit.
  if (NSW) {
    if (Op0 == Op1) {
      // The product of a number with itself is non-negative.
      isKnownNonNegative = true;
    } else {
      bool isKnownNonNegativeOp1 = KnownZero.isNegative();
      bool isKnownNonNegativeOp0 = KnownZero2.isNegative();
      bool isKnownNegativeOp1 = KnownOne.isNegative();
      bool isKnownNegativeOp0 = KnownOne2.isNegative();
      // The product of two numbers with the same sign is non-negative.
      isKnownNonNegative = (isKnownNegativeOp1 && isKnownNegativeOp0) ||
        (isKnownNonNegativeOp1 && isKnownNonNegativeOp0);
      // The product of a negative number and a non-negative number is either
      // negative or zero.
      if (!isKnownNonNegative)
        isKnownNegative = (isKnownNegativeOp1 && isKnownNonNegativeOp0 &&
                           isKnownNonZero(Op0, TD, Depth)) ||
                          (isKnownNegativeOp0 && isKnownNonNegativeOp1 &&
                           isKnownNonZero(Op1, TD, Depth));
    }
  }

  // If low bits are zero in either operand, output low known-0 bits.
  // Also compute a conserative estimate for high known-0 bits.
  // More trickiness is possible, but this is sufficient for the
  // interesting case of alignment computation.
  KnownOne.clearAllBits();
  unsigned TrailZ = KnownZero.countTrailingOnes() +
                    KnownZero2.countTrailingOnes();
  unsigned LeadZ =  std::max(KnownZero.countLeadingOnes() +
                             KnownZero2.countLeadingOnes(),
                             BitWidth) - BitWidth;

  TrailZ = std::min(TrailZ, BitWidth);
  LeadZ = std::min(LeadZ, BitWidth);
  KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) |
              APInt::getHighBitsSet(BitWidth, LeadZ);

  // Only make use of no-wrap flags if we failed to compute the sign bit
  // directly.  This matters if the multiplication always overflows, in
  // which case we prefer to follow the result of the direct computation,
  // though as the program is invoking undefined behaviour we can choose
  // whatever we like here.
  if (isKnownNonNegative && !KnownOne.isNegative())
    KnownZero.setBit(BitWidth - 1);
  else if (isKnownNegative && !KnownZero.isNegative())
    KnownOne.setBit(BitWidth - 1);
}

void llvm::computeMaskedBitsLoad(const MDNode &Ranges, APInt &KnownZero) {
  unsigned BitWidth = KnownZero.getBitWidth();
  unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1);

  // Use the high end of the ranges to find leading zeros.
  unsigned MinLeadingZeros = BitWidth;
  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower = cast<ConstantInt>(Ranges.getOperand(2*i + 0));
    ConstantInt *Upper = cast<ConstantInt>(Ranges.getOperand(2*i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());
    if (Range.isWrappedSet())
      MinLeadingZeros = 0; // -1 has no zeros
    unsigned LeadingZeros = (Upper->getValue() - 1).countLeadingZeros();
    MinLeadingZeros = std::min(LeadingZeros, MinLeadingZeros);
  }

  KnownZero = APInt::getHighBitsSet(BitWidth, MinLeadingZeros);
}
/// ComputeMaskedBits - Determine which of the bits are known to be either zero
/// or one and return them in the KnownZero/KnownOne bit sets.
///
/// NOTE: we cannot consider 'undef' to be "IsZero" here.  The problem is that
/// we cannot optimize based on the assumption that it is zero without changing
/// it to be an explicit zero.  If we don't change it to zero, other code could
/// optimized based on the contradictory assumption that it is non-zero.
/// Because instcombine aggressively folds operations with undef args anyway,
/// this won't lose us code quality.
///
/// This function is defined on values with integer type, values with pointer
/// type (but only if TD is non-null), and vectors of integers.  In the case
/// where V is a vector, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the elements in the vector.
void llvm::ComputeMaskedBits(Value *V, APInt &KnownZero, APInt &KnownOne,
                             const TargetData *TD, unsigned Depth) {
  assert(V && "No Value?");
  assert(Depth <= MaxDepth && "Limit Search Depth");
  unsigned BitWidth = KnownZero.getBitWidth();

  assert((V->getType()->isIntOrIntVectorTy() ||
          V->getType()->getScalarType()->isPointerTy()) &&
         "Not integer or pointer type!");
  assert((!TD ||
          TD->getTypeSizeInBits(V->getType()->getScalarType()) == BitWidth) &&
         (!V->getType()->isIntOrIntVectorTy() ||
          V->getType()->getScalarSizeInBits() == BitWidth) &&
         KnownZero.getBitWidth() == BitWidth &&
         KnownOne.getBitWidth() == BitWidth &&
         "V, Mask, KnownOne and KnownZero should have same BitWidth");

  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getValue();
    KnownZero = ~KnownOne;
    return;
  }
  // Null and aggregate-zero are all-zeros.
  if (isa<ConstantPointerNull>(V) ||
      isa<ConstantAggregateZero>(V)) {
    KnownOne.clearAllBits();
    KnownZero = APInt::getAllOnesValue(BitWidth);
    return;
  }
  // Handle a constant vector by taking the intersection of the known bits of
  // each element.  There is no real need to handle ConstantVector here, because
  // we don't handle undef in any particularly useful way.
  if (ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(V)) {
    // We know that CDS must be a vector of integers. Take the intersection of
    // each element.
    KnownZero.setAllBits(); KnownOne.setAllBits();
    APInt Elt(KnownZero.getBitWidth(), 0);
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      Elt = CDS->getElementAsInteger(i);
      KnownZero &= ~Elt;
      KnownOne &= Elt;      
    }
    return;
  }
  
  // The address of an aligned GlobalValue has trailing zeros.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    unsigned Align = GV->getAlignment();
    if (Align == 0 && TD) {
      if (GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
        Type *ObjectType = GVar->getType()->getElementType();
        if (ObjectType->isSized()) {
          // If the object is defined in the current Module, we'll be giving
          // it the preferred alignment. Otherwise, we have to assume that it
          // may only have the minimum ABI alignment.
          if (!GVar->isDeclaration() && !GVar->isWeakForLinker())
            Align = TD->getPreferredAlignment(GVar);
          else
            Align = TD->getABITypeAlignment(ObjectType);
        }
      }
    }
    if (Align > 0)
      KnownZero = APInt::getLowBitsSet(BitWidth,
                                       CountTrailingZeros_32(Align));
    else
      KnownZero.clearAllBits();
    KnownOne.clearAllBits();
    return;
  }
  // A weak GlobalAlias is totally unknown. A non-weak GlobalAlias has
  // the bits of its aliasee.
  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
    if (GA->mayBeOverridden()) {
      KnownZero.clearAllBits(); KnownOne.clearAllBits();
    } else {
      ComputeMaskedBits(GA->getAliasee(), KnownZero, KnownOne, TD, Depth+1);
    }
    return;
  }
  
  if (Argument *A = dyn_cast<Argument>(V)) {
    // Get alignment information off byval arguments if specified in the IR.
    if (A->hasByValAttr())
      if (unsigned Align = A->getParamAlignment())
        KnownZero = APInt::getLowBitsSet(BitWidth,
                                         CountTrailingZeros_32(Align));
    return;
  }

  // Start out not knowing anything.
  KnownZero.clearAllBits(); KnownOne.clearAllBits();

  if (Depth == MaxDepth)
    return;  // Limit search depth.

  Operator *I = dyn_cast<Operator>(V);
  if (!I) return;

  APInt KnownZero2(KnownZero), KnownOne2(KnownOne);
  switch (I->getOpcode()) {
  default: break;
  case Instruction::Load:
    if (MDNode *MD = cast<LoadInst>(I)->getMetadata(LLVMContext::MD_range))
      computeMaskedBitsLoad(*MD, KnownZero);
    return;
  case Instruction::And: {
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(I->getOperand(1), KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(0), KnownZero2, KnownOne2, TD, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  }
  case Instruction::Or: {
    ComputeMaskedBits(I->getOperand(1), KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(0), KnownZero2, KnownOne2, TD, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  }
  case Instruction::Xor: {
    ComputeMaskedBits(I->getOperand(1), KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(0), KnownZero2, KnownOne2, TD, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case Instruction::Mul: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    ComputeMaskedBitsMul(I->getOperand(0), I->getOperand(1), NSW,
                         KnownZero, KnownOne, KnownZero2, KnownOne2, TD, Depth);
    break;
  }
  case Instruction::UDiv: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    ComputeMaskedBits(I->getOperand(0), KnownZero2, KnownOne2, TD, Depth+1);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clearAllBits();
    KnownZero2.clearAllBits();
    ComputeMaskedBits(I->getOperand(1), KnownZero2, KnownOne2, TD, Depth+1);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ);
    return;
  }
  case Instruction::Select:
    ComputeMaskedBits(I->getOperand(2), KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(1), KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return; // Can't work with floating point.
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    // We can't handle these if we don't know the pointer size.
    if (!TD) return;
    // FALL THROUGH and handle them the same as zext/trunc.
  case Instruction::ZExt:
  case Instruction::Trunc: {
    Type *SrcTy = I->getOperand(0)->getType();
    
    unsigned SrcBitWidth;
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    if (SrcTy->isPointerTy())
      SrcBitWidth = TD->getTypeSizeInBits(SrcTy);
    else
      SrcBitWidth = SrcTy->getScalarSizeInBits();
    
    KnownZero = KnownZero.zextOrTrunc(SrcBitWidth);
    KnownOne = KnownOne.zextOrTrunc(SrcBitWidth);
    ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
    KnownZero = KnownZero.zextOrTrunc(BitWidth);
    KnownOne = KnownOne.zextOrTrunc(BitWidth);
    // Any top bits are known to be zero.
    if (BitWidth > SrcBitWidth)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    return;
  }
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    if ((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
        // TODO: For now, not handling conversions like:
        // (bitcast i64 %x to <2 x i32>)
        !I->getType()->isVectorTy()) {
      ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
      return;
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();
      
    KnownZero = KnownZero.trunc(SrcBitWidth);
    KnownOne = KnownOne.trunc(SrcBitWidth);
    ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero[SrcBitWidth-1])             // Input sign bit known zero
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    else if (KnownOne[SrcBitWidth-1])           // Input sign bit known set
      KnownOne |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    return;
  }
  case Instruction::Shl:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= ShiftAmt;
      KnownOne  <<= ShiftAmt;
      KnownZero |= APInt::getLowBitsSet(BitWidth, ShiftAmt); // low bits known 0
      return;
    }
    break;
  case Instruction::LShr:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // Compute the new bits that are at the top now.
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      
      // Unsigned shift right.
      ComputeMaskedBits(I->getOperand(0), KnownZero,KnownOne, TD, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
      // high bits known zero.
      KnownZero |= APInt::getHighBitsSet(BitWidth, ShiftAmt);
      return;
    }
    break;
  case Instruction::AShr:
    // (ashr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // Compute the new bits that are at the top now.
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth-1);
      
      // Signed shift right.
      ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
        
      APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
      if (KnownZero[BitWidth-ShiftAmt-1])    // New bits are known zero.
        KnownZero |= HighBits;
      else if (KnownOne[BitWidth-ShiftAmt-1])  // New bits are known one.
        KnownOne |= HighBits;
      return;
    }
    break;
  case Instruction::Sub: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    ComputeMaskedBitsAddSub(false, I->getOperand(0), I->getOperand(1), NSW,
                            KnownZero, KnownOne, KnownZero2, KnownOne2, TD,
                            Depth);
    break;
  }
  case Instruction::Add: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    ComputeMaskedBitsAddSub(true, I->getOperand(0), I->getOperand(1), NSW,
                            KnownZero, KnownOne, KnownZero2, KnownOne2, TD,
                            Depth);
    break;
  }
  case Instruction::SRem:
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue().abs();
      if (RA.isPowerOf2()) {
        APInt LowBits = RA - 1;
        ComputeMaskedBits(I->getOperand(0), KnownZero2, KnownOne2, TD, Depth+1);

        // The low bits of the first operand are unchanged by the srem.
        KnownZero = KnownZero2 & LowBits;
        KnownOne = KnownOne2 & LowBits;

        // If the first operand is non-negative or has all low bits zero, then
        // the upper bits are all zero.
        if (KnownZero2[BitWidth-1] || ((KnownZero2 & LowBits) == LowBits))
          KnownZero |= ~LowBits;

        // If the first operand is negative and not all low bits are zero, then
        // the upper bits are all one.
        if (KnownOne2[BitWidth-1] && ((KnownOne2 & LowBits) != 0))
          KnownOne |= ~LowBits;

        assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      }
    }

    // The sign bit is the LHS's sign bit, except when the result of the
    // remainder is zero.
    if (KnownZero.isNonNegative()) {
      APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
      ComputeMaskedBits(I->getOperand(0), LHSKnownZero, LHSKnownOne, TD,
                        Depth+1);
      // If it's known zero, our sign bit is also zero.
      if (LHSKnownZero.isNegative())
        KnownZero.setBit(BitWidth - 1);
    }

    break;
  case Instruction::URem: {
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD,
                          Depth+1);
        assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
        KnownZero |= ~LowBits;
        KnownOne &= LowBits;
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    ComputeMaskedBits(I->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(1), KnownZero2, KnownOne2, TD, Depth+1);

    unsigned Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clearAllBits();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders);
    break;
  }

  case Instruction::Alloca: {
    AllocaInst *AI = cast<AllocaInst>(V);
    unsigned Align = AI->getAlignment();
    if (Align == 0 && TD)
      Align = TD->getABITypeAlignment(AI->getType()->getElementType());
    
    if (Align > 0)
      KnownZero = APInt::getLowBitsSet(BitWidth, CountTrailingZeros_32(Align));
    break;
  }
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    APInt LocalKnownZero(BitWidth, 0), LocalKnownOne(BitWidth, 0);
    ComputeMaskedBits(I->getOperand(0), LocalKnownZero, LocalKnownOne, TD,
                      Depth+1);
    unsigned TrailZ = LocalKnownZero.countTrailingOnes();

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      Value *Index = I->getOperand(i);
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        // Handle struct member offset arithmetic.
        if (!TD) return;
        const StructLayout *SL = TD->getStructLayout(STy);
        unsigned Idx = cast<ConstantInt>(Index)->getZExtValue();
        uint64_t Offset = SL->getElementOffset(Idx);
        TrailZ = std::min(TrailZ,
                          CountTrailingZeros_64(Offset));
      } else {
        // Handle array index arithmetic.
        Type *IndexedTy = GTI.getIndexedType();
        if (!IndexedTy->isSized()) return;
        unsigned GEPOpiBits = Index->getType()->getScalarSizeInBits();
        uint64_t TypeSize = TD ? TD->getTypeAllocSize(IndexedTy) : 1;
        LocalKnownZero = LocalKnownOne = APInt(GEPOpiBits, 0);
        ComputeMaskedBits(Index, LocalKnownZero, LocalKnownOne, TD, Depth+1);
        TrailZ = std::min(TrailZ,
                          unsigned(CountTrailingZeros_64(TypeSize) +
                                   LocalKnownZero.countTrailingOnes()));
      }
    }
    
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ);
    break;
  }
  case Instruction::PHI: {
    PHINode *P = cast<PHINode>(I);
    // Handle the case of a simple two-predecessor recurrence PHI.
    // There's a lot more that could theoretically be done here, but
    // this is sufficient to catch some interesting cases.
    if (P->getNumIncomingValues() == 2) {
      for (unsigned i = 0; i != 2; ++i) {
        Value *L = P->getIncomingValue(i);
        Value *R = P->getIncomingValue(!i);
        Operator *LU = dyn_cast<Operator>(L);
        if (!LU)
          continue;
        unsigned Opcode = LU->getOpcode();
        // Check for operations that have the property that if
        // both their operands have low zero bits, the result
        // will have low zero bits.
        if (Opcode == Instruction::Add ||
            Opcode == Instruction::Sub ||
            Opcode == Instruction::And ||
            Opcode == Instruction::Or ||
            Opcode == Instruction::Mul) {
          Value *LL = LU->getOperand(0);
          Value *LR = LU->getOperand(1);
          // Find a recurrence.
          if (LL == I)
            L = LR;
          else if (LR == I)
            L = LL;
          else
            break;
          // Ok, we have a PHI of the form L op= R. Check for low
          // zero bits.
          ComputeMaskedBits(R, KnownZero2, KnownOne2, TD, Depth+1);

          // We need to take the minimum number of known bits
          APInt KnownZero3(KnownZero), KnownOne3(KnownOne);
          ComputeMaskedBits(L, KnownZero3, KnownOne3, TD, Depth+1);

          KnownZero = APInt::getLowBitsSet(BitWidth,
                                           std::min(KnownZero2.countTrailingOnes(),
                                                    KnownZero3.countTrailingOnes()));
          break;
        }
      }
    }

    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      return;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    if (Depth < MaxDepth - 1 && !KnownZero && !KnownOne) {
      // Skip if every incoming value references to ourself.
      if (P->hasConstantValue() == P)
        break;

      KnownZero = APInt::getAllOnesValue(BitWidth);
      KnownOne = APInt::getAllOnesValue(BitWidth);
      for (unsigned i = 0, e = P->getNumIncomingValues(); i != e; ++i) {
        // Skip direct self references.
        if (P->getIncomingValue(i) == P) continue;

        KnownZero2 = APInt(BitWidth, 0);
        KnownOne2 = APInt(BitWidth, 0);
        // Recurse, but cap the recursion to one level, because we don't
        // want to waste time spinning around in loops.
        ComputeMaskedBits(P->getIncomingValue(i), KnownZero2, KnownOne2, TD,
                          MaxDepth-1);
        KnownZero &= KnownZero2;
        KnownOne &= KnownOne2;
        // If all bits have been ruled out, there's no need to check
        // more operands.
        if (!KnownZero && !KnownOne)
          break;
      }
    }
    break;
  }
  case Instruction::Call:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::ctlz:
      case Intrinsic::cttz: {
        unsigned LowBits = Log2_32(BitWidth)+1;
        // If this call is undefined for 0, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          LowBits -= 1;
        KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
        break;
      }
      case Intrinsic::ctpop: {
        unsigned LowBits = Log2_32(BitWidth)+1;
        KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
        break;
      }
      case Intrinsic::x86_sse42_crc32_64_8:
      case Intrinsic::x86_sse42_crc32_64_64:
        KnownZero = APInt::getHighBitsSet(64, 32);
        break;
      }
    }
    break;
  case Instruction::ExtractValue:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->getOperand(0))) {
      ExtractValueInst *EVI = cast<ExtractValueInst>(I);
      if (EVI->getNumIndices() != 1) break;
      if (EVI->getIndices()[0] == 0) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          ComputeMaskedBitsAddSub(true, II->getArgOperand(0),
                                  II->getArgOperand(1), false, KnownZero,
                                  KnownOne, KnownZero2, KnownOne2, TD, Depth);
          break;
        case Intrinsic::usub_with_overflow:
        case Intrinsic::ssub_with_overflow:
          ComputeMaskedBitsAddSub(false, II->getArgOperand(0),
                                  II->getArgOperand(1), false, KnownZero,
                                  KnownOne, KnownZero2, KnownOne2, TD, Depth);
          break;
        case Intrinsic::umul_with_overflow:
        case Intrinsic::smul_with_overflow:
          ComputeMaskedBitsMul(II->getArgOperand(0), II->getArgOperand(1),
                               false, KnownZero, KnownOne,
                               KnownZero2, KnownOne2, TD, Depth);
          break;
        }
      }
    }
  }
}

/// ComputeSignBit - Determine whether the sign bit is known to be zero or
/// one.  Convenience wrapper around ComputeMaskedBits.
void llvm::ComputeSignBit(Value *V, bool &KnownZero, bool &KnownOne,
                          const TargetData *TD, unsigned Depth) {
  unsigned BitWidth = getBitWidth(V->getType(), TD);
  if (!BitWidth) {
    KnownZero = false;
    KnownOne = false;
    return;
  }
  APInt ZeroBits(BitWidth, 0);
  APInt OneBits(BitWidth, 0);
  ComputeMaskedBits(V, ZeroBits, OneBits, TD, Depth);
  KnownOne = OneBits[BitWidth - 1];
  KnownZero = ZeroBits[BitWidth - 1];
}

/// isPowerOfTwo - Return true if the given value is known to have exactly one
/// bit set when defined. For vectors return true if every element is known to
/// be a power of two when defined.  Supports values with integer or pointer
/// types and vectors of integers.
bool llvm::isPowerOfTwo(Value *V, const TargetData *TD, bool OrZero,
                        unsigned Depth) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return OrZero;
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return CI->getValue().isPowerOf2();
    // TODO: Handle vector constants.
  }

  // 1 << X is clearly a power of two if the one is not shifted off the end.  If
  // it is shifted off the end then the result is undefined.
  if (match(V, m_Shl(m_One(), m_Value())))
    return true;

  // (signbit) >>l X is clearly a power of two if the one is not shifted off the
  // bottom.  If it is shifted off the bottom then the result is undefined.
  if (match(V, m_LShr(m_SignBit(), m_Value())))
    return true;

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ == MaxDepth)
    return false;

  Value *X = 0, *Y = 0;
  // A shift of a power of two is a power of two or zero.
  if (OrZero && (match(V, m_Shl(m_Value(X), m_Value())) ||
                 match(V, m_Shr(m_Value(X), m_Value()))))
    return isPowerOfTwo(X, TD, /*OrZero*/true, Depth);

  if (ZExtInst *ZI = dyn_cast<ZExtInst>(V))
    return isPowerOfTwo(ZI->getOperand(0), TD, OrZero, Depth);

  if (SelectInst *SI = dyn_cast<SelectInst>(V))
    return isPowerOfTwo(SI->getTrueValue(), TD, OrZero, Depth) &&
      isPowerOfTwo(SI->getFalseValue(), TD, OrZero, Depth);

  if (OrZero && match(V, m_And(m_Value(X), m_Value(Y)))) {
    // A power of two and'd with anything is a power of two or zero.
    if (isPowerOfTwo(X, TD, /*OrZero*/true, Depth) ||
        isPowerOfTwo(Y, TD, /*OrZero*/true, Depth))
      return true;
    // X & (-X) is always a power of two or zero.
    if (match(X, m_Neg(m_Specific(Y))) || match(Y, m_Neg(m_Specific(X))))
      return true;
    return false;
  }

  // An exact divide or right shift can only shift off zero bits, so the result
  // is a power of two only if the first operand is a power of two and not
  // copying a sign bit (sdiv int_min, 2).
  if (match(V, m_Exact(m_LShr(m_Value(), m_Value()))) ||
      match(V, m_Exact(m_UDiv(m_Value(), m_Value())))) {
    return isPowerOfTwo(cast<Operator>(V)->getOperand(0), TD, OrZero, Depth);
  }

  return false;
}

/// isKnownNonZero - Return true if the given value is known to be non-zero
/// when defined.  For vectors return true if every element is known to be
/// non-zero when defined.  Supports values with integer or pointer type and
/// vectors of integers.
bool llvm::isKnownNonZero(Value *V, const TargetData *TD, unsigned Depth) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return false;
    if (isa<ConstantInt>(C))
      // Must be non-zero due to null test above.
      return true;
    // TODO: Handle vectors
    return false;
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ >= MaxDepth)
    return false;

  unsigned BitWidth = getBitWidth(V->getType(), TD);

  // X | Y != 0 if X != 0 or Y != 0.
  Value *X = 0, *Y = 0;
  if (match(V, m_Or(m_Value(X), m_Value(Y))))
    return isKnownNonZero(X, TD, Depth) || isKnownNonZero(Y, TD, Depth);

  // ext X != 0 if X != 0.
  if (isa<SExtInst>(V) || isa<ZExtInst>(V))
    return isKnownNonZero(cast<Instruction>(V)->getOperand(0), TD, Depth);

  // shl X, Y != 0 if X is odd.  Note that the value of the shift is undefined
  // if the lowest bit is shifted off the end.
  if (BitWidth && match(V, m_Shl(m_Value(X), m_Value(Y)))) {
    // shl nuw can't remove any non-zero bits.
    OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    if (BO->hasNoUnsignedWrap())
      return isKnownNonZero(X, TD, Depth);

    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(X, KnownZero, KnownOne, TD, Depth);
    if (KnownOne[0])
      return true;
  }
  // shr X, Y != 0 if X is negative.  Note that the value of the shift is not
  // defined if the sign bit is shifted off the end.
  else if (match(V, m_Shr(m_Value(X), m_Value(Y)))) {
    // shr exact can only shift out zero bits.
    PossiblyExactOperator *BO = cast<PossiblyExactOperator>(V);
    if (BO->isExact())
      return isKnownNonZero(X, TD, Depth);

    bool XKnownNonNegative, XKnownNegative;
    ComputeSignBit(X, XKnownNonNegative, XKnownNegative, TD, Depth);
    if (XKnownNegative)
      return true;
  }
  // div exact can only produce a zero if the dividend is zero.
  else if (match(V, m_Exact(m_IDiv(m_Value(X), m_Value())))) {
    return isKnownNonZero(X, TD, Depth);
  }
  // X + Y.
  else if (match(V, m_Add(m_Value(X), m_Value(Y)))) {
    bool XKnownNonNegative, XKnownNegative;
    bool YKnownNonNegative, YKnownNegative;
    ComputeSignBit(X, XKnownNonNegative, XKnownNegative, TD, Depth);
    ComputeSignBit(Y, YKnownNonNegative, YKnownNegative, TD, Depth);

    // If X and Y are both non-negative (as signed values) then their sum is not
    // zero unless both X and Y are zero.
    if (XKnownNonNegative && YKnownNonNegative)
      if (isKnownNonZero(X, TD, Depth) || isKnownNonZero(Y, TD, Depth))
        return true;

    // If X and Y are both negative (as signed values) then their sum is not
    // zero unless both X and Y equal INT_MIN.
    if (BitWidth && XKnownNegative && YKnownNegative) {
      APInt KnownZero(BitWidth, 0);
      APInt KnownOne(BitWidth, 0);
      APInt Mask = APInt::getSignedMaxValue(BitWidth);
      // The sign bit of X is set.  If some other bit is set then X is not equal
      // to INT_MIN.
      ComputeMaskedBits(X, KnownZero, KnownOne, TD, Depth);
      if ((KnownOne & Mask) != 0)
        return true;
      // The sign bit of Y is set.  If some other bit is set then Y is not equal
      // to INT_MIN.
      ComputeMaskedBits(Y, KnownZero, KnownOne, TD, Depth);
      if ((KnownOne & Mask) != 0)
        return true;
    }

    // The sum of a non-negative number and a power of two is not zero.
    if (XKnownNonNegative && isPowerOfTwo(Y, TD, /*OrZero*/false, Depth))
      return true;
    if (YKnownNonNegative && isPowerOfTwo(X, TD, /*OrZero*/false, Depth))
      return true;
  }
  // X * Y.
  else if (match(V, m_Mul(m_Value(X), m_Value(Y)))) {
    OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    // If X and Y are non-zero then so is X * Y as long as the multiplication
    // does not overflow.
    if ((BO->hasNoSignedWrap() || BO->hasNoUnsignedWrap()) &&
        isKnownNonZero(X, TD, Depth) && isKnownNonZero(Y, TD, Depth))
      return true;
  }
  // (C ? X : Y) != 0 if X != 0 and Y != 0.
  else if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    if (isKnownNonZero(SI->getTrueValue(), TD, Depth) &&
        isKnownNonZero(SI->getFalseValue(), TD, Depth))
      return true;
  }

  if (!BitWidth) return false;
  APInt KnownZero(BitWidth, 0);
  APInt KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, KnownZero, KnownOne, TD, Depth);
  return KnownOne != 0;
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
///
/// This function is defined on values with integer type, values with pointer
/// type (but only if TD is non-null), and vectors of integers.  In the case
/// where V is a vector, the mask, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the elements in the vector.
bool llvm::MaskedValueIsZero(Value *V, const APInt &Mask,
                             const TargetData *TD, unsigned Depth) {
  APInt KnownZero(Mask.getBitWidth(), 0), KnownOne(Mask.getBitWidth(), 0);
  ComputeMaskedBits(V, KnownZero, KnownOne, TD, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}



/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "ashr X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
///
/// 'Op' must have a scalar integer type.
///
unsigned llvm::ComputeNumSignBits(Value *V, const TargetData *TD,
                                  unsigned Depth) {
  assert((TD || V->getType()->isIntOrIntVectorTy()) &&
         "ComputeNumSignBits requires a TargetData object to operate "
         "on non-integer values!");
  Type *Ty = V->getType();
  unsigned TyBits = TD ? TD->getTypeSizeInBits(V->getType()->getScalarType()) :
                         Ty->getScalarSizeInBits();
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  // Note that ConstantInt is handled by the general ComputeMaskedBits case
  // below.

  if (Depth == 6)
    return 1;  // Limit search depth.
  
  Operator *U = dyn_cast<Operator>(V);
  switch (Operator::getOpcode(V)) {
  default: break;
  case Instruction::SExt:
    Tmp = TyBits - U->getOperand(0)->getType()->getScalarSizeInBits();
    return ComputeNumSignBits(U->getOperand(0), TD, Depth+1) + Tmp;
    
  case Instruction::AShr: {
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    // ashr X, C   -> adds C sign bits.  Vectors too.
    const APInt *ShAmt;
    if (match(U->getOperand(1), m_APInt(ShAmt))) {
      Tmp += ShAmt->getZExtValue();
      if (Tmp > TyBits) Tmp = TyBits;
    }
    return Tmp;
  }
  case Instruction::Shl: {
    const APInt *ShAmt;
    if (match(U->getOperand(1), m_APInt(ShAmt))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
      Tmp2 = ShAmt->getZExtValue();
      if (Tmp2 >= TyBits ||      // Bad shift.
          Tmp2 >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - Tmp2;
    }
    break;
  }
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits at the worst.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // ComputeMaskedBits, and pick whichever answer is better.
    }
    break;

  case Instruction::Select:
    Tmp = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(U->getOperand(2), TD, Depth+1);
    return std::min(Tmp, Tmp2);
    
  case Instruction::Add:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      
    // Special case decrementing a value (ADD X, -1):
    if (ConstantInt *CRHS = dyn_cast<ConstantInt>(U->getOperand(1)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        ComputeMaskedBits(U->getOperand(0), KnownZero, KnownOne, TD, Depth+1);
        
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(TyBits, 1)).isAllOnesValue())
          return TyBits;
        
        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }
      
    Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp2 == 1) return 1;
    return std::min(Tmp, Tmp2)-1;
    
  case Instruction::Sub:
    Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp2 == 1) return 1;
      
    // Handle NEG.
    if (ConstantInt *CLHS = dyn_cast<ConstantInt>(U->getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        ComputeMaskedBits(U->getOperand(1), KnownZero, KnownOne, TD, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(TyBits, 1)).isAllOnesValue())
          return TyBits;
        
        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero.isNegative())
          return Tmp2;
        
        // Otherwise, we treat this like a SUB.
      }
    
    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    return std::min(Tmp, Tmp2)-1;
      
  case Instruction::PHI: {
    PHINode *PN = cast<PHINode>(U);
    // Don't analyze large in-degree PHIs.
    if (PN->getNumIncomingValues() > 4) break;
    
    // Take the minimum of all incoming values.  This can't infinitely loop
    // because of our depth threshold.
    Tmp = ComputeNumSignBits(PN->getIncomingValue(0), TD, Depth+1);
    for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i) {
      if (Tmp == 1) return Tmp;
      Tmp = std::min(Tmp,
                     ComputeNumSignBits(PN->getIncomingValue(i), TD, Depth+1));
    }
    return Tmp;
  }

  case Instruction::Trunc:
    // FIXME: it's tricky to do anything useful for this, but it is an important
    // case for targets like X86.
    break;
  }
  
  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
  APInt Mask;
  ComputeMaskedBits(V, KnownZero, KnownOne, TD, Depth);
  
  if (KnownZero.isNegative()) {        // sign bit is 0
    Mask = KnownZero;
  } else if (KnownOne.isNegative()) {  // sign bit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return FirstAnswer;
  }
  
  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask = ~Mask;
  Mask <<= Mask.getBitWidth()-TyBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::max(FirstAnswer, std::min(TyBits, Mask.countLeadingZeros()));
}

/// ComputeMultiple - This function computes the integer multiple of Base that
/// equals V.  If successful, it returns true and returns the multiple in
/// Multiple.  If unsuccessful, it returns false. It looks
/// through SExt instructions only if LookThroughSExt is true.
bool llvm::ComputeMultiple(Value *V, unsigned Base, Value *&Multiple,
                           bool LookThroughSExt, unsigned Depth) {
  const unsigned MaxDepth = 6;

  assert(V && "No Value?");
  assert(Depth <= MaxDepth && "Limit Search Depth");
  assert(V->getType()->isIntegerTy() && "Not integer or pointer type!");

  Type *T = V->getType();

  ConstantInt *CI = dyn_cast<ConstantInt>(V);

  if (Base == 0)
    return false;
    
  if (Base == 1) {
    Multiple = V;
    return true;
  }

  ConstantExpr *CO = dyn_cast<ConstantExpr>(V);
  Constant *BaseVal = ConstantInt::get(T, Base);
  if (CO && CO == BaseVal) {
    // Multiple is 1.
    Multiple = ConstantInt::get(T, 1);
    return true;
  }

  if (CI && CI->getZExtValue() % Base == 0) {
    Multiple = ConstantInt::get(T, CI->getZExtValue() / Base);
    return true;  
  }
  
  if (Depth == MaxDepth) return false;  // Limit search depth.
        
  Operator *I = dyn_cast<Operator>(V);
  if (!I) return false;

  switch (I->getOpcode()) {
  default: break;
  case Instruction::SExt:
    if (!LookThroughSExt) return false;
    // otherwise fall through to ZExt
  case Instruction::ZExt:
    return ComputeMultiple(I->getOperand(0), Base, Multiple,
                           LookThroughSExt, Depth+1);
  case Instruction::Shl:
  case Instruction::Mul: {
    Value *Op0 = I->getOperand(0);
    Value *Op1 = I->getOperand(1);

    if (I->getOpcode() == Instruction::Shl) {
      ConstantInt *Op1CI = dyn_cast<ConstantInt>(Op1);
      if (!Op1CI) return false;
      // Turn Op0 << Op1 into Op0 * 2^Op1
      APInt Op1Int = Op1CI->getValue();
      uint64_t BitToSet = Op1Int.getLimitedValue(Op1Int.getBitWidth() - 1);
      APInt API(Op1Int.getBitWidth(), 0);
      API.setBit(BitToSet);
      Op1 = ConstantInt::get(V->getContext(), API);
    }

    Value *Mul0 = NULL;
    if (ComputeMultiple(Op0, Base, Mul0, LookThroughSExt, Depth+1)) {
      if (Constant *Op1C = dyn_cast<Constant>(Op1))
        if (Constant *MulC = dyn_cast<Constant>(Mul0)) {
          if (Op1C->getType()->getPrimitiveSizeInBits() < 
              MulC->getType()->getPrimitiveSizeInBits())
            Op1C = ConstantExpr::getZExt(Op1C, MulC->getType());
          if (Op1C->getType()->getPrimitiveSizeInBits() > 
              MulC->getType()->getPrimitiveSizeInBits())
            MulC = ConstantExpr::getZExt(MulC, Op1C->getType());
          
          // V == Base * (Mul0 * Op1), so return (Mul0 * Op1)
          Multiple = ConstantExpr::getMul(MulC, Op1C);
          return true;
        }

      if (ConstantInt *Mul0CI = dyn_cast<ConstantInt>(Mul0))
        if (Mul0CI->getValue() == 1) {
          // V == Base * Op1, so return Op1
          Multiple = Op1;
          return true;
        }
    }

    Value *Mul1 = NULL;
    if (ComputeMultiple(Op1, Base, Mul1, LookThroughSExt, Depth+1)) {
      if (Constant *Op0C = dyn_cast<Constant>(Op0))
        if (Constant *MulC = dyn_cast<Constant>(Mul1)) {
          if (Op0C->getType()->getPrimitiveSizeInBits() < 
              MulC->getType()->getPrimitiveSizeInBits())
            Op0C = ConstantExpr::getZExt(Op0C, MulC->getType());
          if (Op0C->getType()->getPrimitiveSizeInBits() > 
              MulC->getType()->getPrimitiveSizeInBits())
            MulC = ConstantExpr::getZExt(MulC, Op0C->getType());
          
          // V == Base * (Mul1 * Op0), so return (Mul1 * Op0)
          Multiple = ConstantExpr::getMul(MulC, Op0C);
          return true;
        }

      if (ConstantInt *Mul1CI = dyn_cast<ConstantInt>(Mul1))
        if (Mul1CI->getValue() == 1) {
          // V == Base * Op0, so return Op0
          Multiple = Op0;
          return true;
        }
    }
  }
  }

  // We could not determine if V is a multiple of Base.
  return false;
}

/// CannotBeNegativeZero - Return true if we can prove that the specified FP 
/// value is never equal to -0.0.
///
/// NOTE: this function will need to be revisited when we support non-default
/// rounding modes!
///
bool llvm::CannotBeNegativeZero(const Value *V, unsigned Depth) {
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->getValueAPF().isNegZero();
  
  if (Depth == 6)
    return 1;  // Limit search depth.

  const Operator *I = dyn_cast<Operator>(V);
  if (I == 0) return false;
  
  // (add x, 0.0) is guaranteed to return +0.0, not -0.0.
  if (I->getOpcode() == Instruction::FAdd &&
      isa<ConstantFP>(I->getOperand(1)) && 
      cast<ConstantFP>(I->getOperand(1))->isNullValue())
    return true;
    
  // sitofp and uitofp turn into +0.0 for zero.
  if (isa<SIToFPInst>(I) || isa<UIToFPInst>(I))
    return true;
  
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    // sqrt(-0.0) = -0.0, no other negative results are possible.
    if (II->getIntrinsicID() == Intrinsic::sqrt)
      return CannotBeNegativeZero(II->getArgOperand(0), Depth+1);
  
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction()) {
      if (F->isDeclaration()) {
        // abs(x) != -0.0
        if (F->getName() == "abs") return true;
        // fabs[lf](x) != -0.0
        if (F->getName() == "fabs") return true;
        if (F->getName() == "fabsf") return true;
        if (F->getName() == "fabsl") return true;
        if (F->getName() == "sqrt" || F->getName() == "sqrtf" ||
            F->getName() == "sqrtl")
          return CannotBeNegativeZero(CI->getArgOperand(0), Depth+1);
      }
    }
  
  return false;
}

/// isBytewiseValue - If the specified value can be set by repeating the same
/// byte in memory, return the i8 value that it is represented with.  This is
/// true for all i8 values obviously, but is also true for i32 0, i32 -1,
/// i16 0xF0F0, double 0.0 etc.  If the value can't be handled with a repeated
/// byte store (e.g. i16 0x1234), return null.
Value *llvm::isBytewiseValue(Value *V) {
  // All byte-wide stores are splatable, even of arbitrary variables.
  if (V->getType()->isIntegerTy(8)) return V;

  // Handle 'null' ConstantArrayZero etc.
  if (Constant *C = dyn_cast<Constant>(V))
    if (C->isNullValue())
      return Constant::getNullValue(Type::getInt8Ty(V->getContext()));
  
  // Constant float and double values can be handled as integer values if the
  // corresponding integer value is "byteable".  An important case is 0.0. 
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    if (CFP->getType()->isFloatTy())
      V = ConstantExpr::getBitCast(CFP, Type::getInt32Ty(V->getContext()));
    if (CFP->getType()->isDoubleTy())
      V = ConstantExpr::getBitCast(CFP, Type::getInt64Ty(V->getContext()));
    // Don't handle long double formats, which have strange constraints.
  }
  
  // We can handle constant integers that are power of two in size and a 
  // multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    unsigned Width = CI->getBitWidth();
    if (isPowerOf2_32(Width) && Width > 8) {
      // We can handle this value if the recursive binary decomposition is the
      // same at all levels.
      APInt Val = CI->getValue();
      APInt Val2;
      while (Val.getBitWidth() != 8) {
        unsigned NextWidth = Val.getBitWidth()/2;
        Val2  = Val.lshr(NextWidth);
        Val2 = Val2.trunc(Val.getBitWidth()/2);
        Val = Val.trunc(Val.getBitWidth()/2);
        
        // If the top/bottom halves aren't the same, reject it.
        if (Val != Val2)
          return 0;
      }
      return ConstantInt::get(V->getContext(), Val);
    }
  }
  
  // A ConstantDataArray/Vector is splatable if all its members are equal and
  // also splatable.
  if (ConstantDataSequential *CA = dyn_cast<ConstantDataSequential>(V)) {
    Value *Elt = CA->getElementAsConstant(0);
    Value *Val = isBytewiseValue(Elt);
    if (!Val)
      return 0;
    
    for (unsigned I = 1, E = CA->getNumElements(); I != E; ++I)
      if (CA->getElementAsConstant(I) != Elt)
        return 0;
    
    return Val;
  }

  // Conceptually, we could handle things like:
  //   %a = zext i8 %X to i16
  //   %b = shl i16 %a, 8
  //   %c = or i16 %a, %b
  // but until there is an example that actually needs this, it doesn't seem
  // worth worrying about.
  return 0;
}


// This is the recursive version of BuildSubAggregate. It takes a few different
// arguments. Idxs is the index within the nested struct From that we are
// looking at now (which is of type IndexedType). IdxSkip is the number of
// indices from Idxs that should be left out when inserting into the resulting
// struct. To is the result struct built so far, new insertvalue instructions
// build on that.
static Value *BuildSubAggregate(Value *From, Value* To, Type *IndexedType,
                                SmallVector<unsigned, 10> &Idxs,
                                unsigned IdxSkip,
                                Instruction *InsertBefore) {
  llvm::StructType *STy = llvm::dyn_cast<llvm::StructType>(IndexedType);
  if (STy) {
    // Save the original To argument so we can modify it
    Value *OrigTo = To;
    // General case, the type indexed by Idxs is a struct
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      // Process each struct element recursively
      Idxs.push_back(i);
      Value *PrevTo = To;
      To = BuildSubAggregate(From, To, STy->getElementType(i), Idxs, IdxSkip,
                             InsertBefore);
      Idxs.pop_back();
      if (!To) {
        // Couldn't find any inserted value for this index? Cleanup
        while (PrevTo != OrigTo) {
          InsertValueInst* Del = cast<InsertValueInst>(PrevTo);
          PrevTo = Del->getAggregateOperand();
          Del->eraseFromParent();
        }
        // Stop processing elements
        break;
      }
    }
    // If we successfully found a value for each of our subaggregates
    if (To)
      return To;
  }
  // Base case, the type indexed by SourceIdxs is not a struct, or not all of
  // the struct's elements had a value that was inserted directly. In the latter
  // case, perhaps we can't determine each of the subelements individually, but
  // we might be able to find the complete struct somewhere.
  
  // Find the value that is at that particular spot
  Value *V = FindInsertedValue(From, Idxs);

  if (!V)
    return NULL;

  // Insert the value in the new (sub) aggregrate
  return llvm::InsertValueInst::Create(To, V, makeArrayRef(Idxs).slice(IdxSkip),
                                       "tmp", InsertBefore);
}

// This helper takes a nested struct and extracts a part of it (which is again a
// struct) into a new value. For example, given the struct:
// { a, { b, { c, d }, e } }
// and the indices "1, 1" this returns
// { c, d }.
//
// It does this by inserting an insertvalue for each element in the resulting
// struct, as opposed to just inserting a single struct. This will only work if
// each of the elements of the substruct are known (ie, inserted into From by an
// insertvalue instruction somewhere).
//
// All inserted insertvalue instructions are inserted before InsertBefore
static Value *BuildSubAggregate(Value *From, ArrayRef<unsigned> idx_range,
                                Instruction *InsertBefore) {
  assert(InsertBefore && "Must have someplace to insert!");
  Type *IndexedType = ExtractValueInst::getIndexedType(From->getType(),
                                                             idx_range);
  Value *To = UndefValue::get(IndexedType);
  SmallVector<unsigned, 10> Idxs(idx_range.begin(), idx_range.end());
  unsigned IdxSkip = Idxs.size();

  return BuildSubAggregate(From, To, IndexedType, Idxs, IdxSkip, InsertBefore);
}

/// FindInsertedValue - Given an aggregrate and an sequence of indices, see if
/// the scalar value indexed is already around as a register, for example if it
/// were inserted directly into the aggregrate.
///
/// If InsertBefore is not null, this function will duplicate (modified)
/// insertvalues when a part of a nested struct is extracted.
Value *llvm::FindInsertedValue(Value *V, ArrayRef<unsigned> idx_range,
                               Instruction *InsertBefore) {
  // Nothing to index? Just return V then (this is useful at the end of our
  // recursion).
  if (idx_range.empty())
    return V;
  // We have indices, so V should have an indexable type.
  assert((V->getType()->isStructTy() || V->getType()->isArrayTy()) &&
         "Not looking at a struct or array?");
  assert(ExtractValueInst::getIndexedType(V->getType(), idx_range) &&
         "Invalid indices for type?");

  if (Constant *C = dyn_cast<Constant>(V)) {
    C = C->getAggregateElement(idx_range[0]);
    if (C == 0) return 0;
    return FindInsertedValue(C, idx_range.slice(1), InsertBefore);
  }
    
  if (InsertValueInst *I = dyn_cast<InsertValueInst>(V)) {
    // Loop the indices for the insertvalue instruction in parallel with the
    // requested indices
    const unsigned *req_idx = idx_range.begin();
    for (const unsigned *i = I->idx_begin(), *e = I->idx_end();
         i != e; ++i, ++req_idx) {
      if (req_idx == idx_range.end()) {
        // We can't handle this without inserting insertvalues
        if (!InsertBefore)
          return 0;

        // The requested index identifies a part of a nested aggregate. Handle
        // this specially. For example,
        // %A = insertvalue { i32, {i32, i32 } } undef, i32 10, 1, 0
        // %B = insertvalue { i32, {i32, i32 } } %A, i32 11, 1, 1
        // %C = extractvalue {i32, { i32, i32 } } %B, 1
        // This can be changed into
        // %A = insertvalue {i32, i32 } undef, i32 10, 0
        // %C = insertvalue {i32, i32 } %A, i32 11, 1
        // which allows the unused 0,0 element from the nested struct to be
        // removed.
        return BuildSubAggregate(V, makeArrayRef(idx_range.begin(), req_idx),
                                 InsertBefore);
      }
      
      // This insert value inserts something else than what we are looking for.
      // See if the (aggregrate) value inserted into has the value we are
      // looking for, then.
      if (*req_idx != *i)
        return FindInsertedValue(I->getAggregateOperand(), idx_range,
                                 InsertBefore);
    }
    // If we end up here, the indices of the insertvalue match with those
    // requested (though possibly only partially). Now we recursively look at
    // the inserted value, passing any remaining indices.
    return FindInsertedValue(I->getInsertedValueOperand(),
                             makeArrayRef(req_idx, idx_range.end()),
                             InsertBefore);
  }
  
  if (ExtractValueInst *I = dyn_cast<ExtractValueInst>(V)) {
    // If we're extracting a value from an aggregrate that was extracted from
    // something else, we can extract from that something else directly instead.
    // However, we will need to chain I's indices with the requested indices.
   
    // Calculate the number of indices required 
    unsigned size = I->getNumIndices() + idx_range.size();
    // Allocate some space to put the new indices in
    SmallVector<unsigned, 5> Idxs;
    Idxs.reserve(size);
    // Add indices from the extract value instruction
    Idxs.append(I->idx_begin(), I->idx_end());
    
    // Add requested indices
    Idxs.append(idx_range.begin(), idx_range.end());

    assert(Idxs.size() == size 
           && "Number of indices added not correct?");
    
    return FindInsertedValue(I->getAggregateOperand(), Idxs, InsertBefore);
  }
  // Otherwise, we don't know (such as, extracting from a function return value
  // or load instruction)
  return 0;
}

/// GetPointerBaseWithConstantOffset - Analyze the specified pointer to see if
/// it can be expressed as a base pointer plus a constant offset.  Return the
/// base and offset to the caller.
Value *llvm::GetPointerBaseWithConstantOffset(Value *Ptr, int64_t &Offset,
                                              const TargetData &TD) {
  Operator *PtrOp = dyn_cast<Operator>(Ptr);
  if (PtrOp == 0 || Ptr->getType()->isVectorTy())
    return Ptr;
  
  // Just look through bitcasts.
  if (PtrOp->getOpcode() == Instruction::BitCast)
    return GetPointerBaseWithConstantOffset(PtrOp->getOperand(0), Offset, TD);
  
  // If this is a GEP with constant indices, we can look through it.
  GEPOperator *GEP = dyn_cast<GEPOperator>(PtrOp);
  if (GEP == 0 || !GEP->hasAllConstantIndices()) return Ptr;
  
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator I = GEP->idx_begin(), E = GEP->idx_end(); I != E;
       ++I, ++GTI) {
    ConstantInt *OpC = cast<ConstantInt>(*I);
    if (OpC->isZero()) continue;
    
    // Handle a struct and array indices which add their offset to the pointer.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      Offset += TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
    } else {
      uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType());
      Offset += OpC->getSExtValue()*Size;
    }
  }
  
  // Re-sign extend from the pointer size if needed to get overflow edge cases
  // right.
  unsigned PtrSize = TD.getPointerSizeInBits();
  if (PtrSize < 64)
    Offset = (Offset << (64-PtrSize)) >> (64-PtrSize);
  
  return GetPointerBaseWithConstantOffset(GEP->getPointerOperand(), Offset, TD);
}


/// getConstantStringInfo - This function computes the length of a
/// null-terminated C string pointed to by V.  If successful, it returns true
/// and returns the string in Str.  If unsuccessful, it returns false.
bool llvm::getConstantStringInfo(const Value *V, StringRef &Str,
                                 uint64_t Offset, bool TrimAtNul) {
  assert(V);

  // Look through bitcast instructions and geps.
  V = V->stripPointerCasts();
  
  // If the value is a GEP instructionor  constant expression, treat it as an
  // offset.
  if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    // Make sure the GEP has exactly three arguments.
    if (GEP->getNumOperands() != 3)
      return false;
    
    // Make sure the index-ee is a pointer to array of i8.
    PointerType *PT = cast<PointerType>(GEP->getOperand(0)->getType());
    ArrayType *AT = dyn_cast<ArrayType>(PT->getElementType());
    if (AT == 0 || !AT->getElementType()->isIntegerTy(8))
      return false;
    
    // Check to make sure that the first operand of the GEP is an integer and
    // has value 0 so that we are sure we're indexing into the initializer.
    const ConstantInt *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
    if (FirstIdx == 0 || !FirstIdx->isZero())
      return false;
    
    // If the second index isn't a ConstantInt, then this is a variable index
    // into the array.  If this occurs, we can't say anything meaningful about
    // the string.
    uint64_t StartIdx = 0;
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
      StartIdx = CI->getZExtValue();
    else
      return false;
    return getConstantStringInfo(GEP->getOperand(0), Str, StartIdx+Offset);
  }

  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return false;

  // Handle the all-zeros case
  if (GV->getInitializer()->isNullValue()) {
    // This is a degenerate case. The initializer is constant zero so the
    // length of the string must be zero.
    Str = "";
    return true;
  }
  
  // Must be a Constant Array
  const ConstantDataArray *Array =
    dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (Array == 0 || !Array->isString())
    return false;
  
  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getArrayNumElements();

  // Start out with the entire array in the StringRef.
  Str = Array->getAsString();

  if (Offset > NumElts)
    return false;
  
  // Skip over 'offset' bytes.
  Str = Str.substr(Offset);
  
  if (TrimAtNul) {
    // Trim off the \0 and anything after it.  If the array is not nul
    // terminated, we just return the whole end of string.  The client may know
    // some other way that the string is length-bound.
    Str = Str.substr(0, Str.find('\0'));
  }
  return true;
}

// These next two are very similar to the above, but also look through PHI
// nodes.
// TODO: See if we can integrate these two together.

/// GetStringLengthH - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(Value *V, SmallPtrSet<PHINode*, 32> &PHIs) {
  // Look through noop bitcast instructions.
  V = V->stripPointerCasts();

  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN))
      return ~0ULL;  // already in the set.

    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      uint64_t Len = GetStringLengthH(PN->getIncomingValue(i), PHIs);
      if (Len == 0) return 0; // Unknown length -> unknown.

      if (Len == ~0ULL) continue;

      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0;    // Disagree -> unknown.
      LenSoFar = Len;
    }

    // Success, all agree.
    return LenSoFar;
  }

  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs);
    if (Len1 == 0) return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs);
    if (Len2 == 0) return 0;
    if (Len1 == ~0ULL) return Len2;
    if (Len2 == ~0ULL) return Len1;
    if (Len1 != Len2) return 0;
    return Len1;
  }
  
  // Otherwise, see if we can read the string.
  StringRef StrData;
  if (!getConstantStringInfo(V, StrData))
    return 0;

  return StrData.size()+1;
}

/// GetStringLength - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
uint64_t llvm::GetStringLength(Value *V) {
  if (!V->getType()->isPointerTy()) return 0;

  SmallPtrSet<PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

Value *
llvm::GetUnderlyingObject(Value *V, const TargetData *TD, unsigned MaxLookup) {
  if (!V->getType()->isPointerTy())
    return V;
  for (unsigned Count = 0; MaxLookup == 0 || Count < MaxLookup; ++Count) {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->mayBeOverridden())
        return V;
      V = GA->getAliasee();
    } else {
      // See if InstructionSimplify knows any relevant tricks.
      if (Instruction *I = dyn_cast<Instruction>(V))
        // TODO: Acquire a DominatorTree and use it.
        if (Value *Simplified = SimplifyInstruction(I, TD, 0)) {
          V = Simplified;
          continue;
        }

      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  }
  return V;
}

/// onlyUsedByLifetimeMarkers - Return true if the only users of this pointer
/// are lifetime markers.
///
bool llvm::onlyUsedByLifetimeMarkers(const Value *V) {
  for (Value::const_use_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(*UI);
    if (!II) return false;

    if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
        II->getIntrinsicID() != Intrinsic::lifetime_end)
      return false;
  }
  return true;
}

bool llvm::isSafeToSpeculativelyExecute(const Value *V,
                                        const TargetData *TD) {
  const Operator *Inst = dyn_cast<Operator>(V);
  if (!Inst)
    return false;

  for (unsigned i = 0, e = Inst->getNumOperands(); i != e; ++i)
    if (Constant *C = dyn_cast<Constant>(Inst->getOperand(i)))
      if (C->canTrap())
        return false;

  switch (Inst->getOpcode()) {
  default:
    return true;
  case Instruction::UDiv:
  case Instruction::URem:
    // x / y is undefined if y == 0, but calcuations like x / 3 are safe.
    return isKnownNonZero(Inst->getOperand(1), TD);
  case Instruction::SDiv:
  case Instruction::SRem: {
    Value *Op = Inst->getOperand(1);
    // x / y is undefined if y == 0
    if (!isKnownNonZero(Op, TD))
      return false;
    // x / y might be undefined if y == -1
    unsigned BitWidth = getBitWidth(Op->getType(), TD);
    if (BitWidth == 0)
      return false;
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(Op, KnownZero, KnownOne, TD);
    return !!KnownZero;
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(Inst);
    if (!LI->isUnordered())
      return false;
    return LI->getPointerOperand()->isDereferenceablePointer();
  }
  case Instruction::Call: {
   if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
     switch (II->getIntrinsicID()) {
       // These synthetic intrinsics have no side-effects, and just mark
       // information about their operands.
       // FIXME: There are other no-op synthetic instructions that potentially
       // should be considered at least *safe* to speculate...
       case Intrinsic::dbg_declare:
       case Intrinsic::dbg_value:
         return true;

       case Intrinsic::bswap:
       case Intrinsic::ctlz:
       case Intrinsic::ctpop:
       case Intrinsic::cttz:
       case Intrinsic::objectsize:
       case Intrinsic::sadd_with_overflow:
       case Intrinsic::smul_with_overflow:
       case Intrinsic::ssub_with_overflow:
       case Intrinsic::uadd_with_overflow:
       case Intrinsic::umul_with_overflow:
       case Intrinsic::usub_with_overflow:
         return true;
       // TODO: some fp intrinsics are marked as having the same error handling
       // as libm. They're safe to speculate when they won't error.
       // TODO: are convert_{from,to}_fp16 safe?
       // TODO: can we list target-specific intrinsics here?
       default: break;
     }
   }
    return false; // The called function could have undefined behavior or
                  // side-effects, even if marked readnone nounwind.
  }
  case Instruction::VAArg:
  case Instruction::Alloca:
  case Instruction::Invoke:
  case Instruction::PHI:
  case Instruction::Store:
  case Instruction::Ret:
  case Instruction::Br:
  case Instruction::IndirectBr:
  case Instruction::Switch:
  case Instruction::Unreachable:
  case Instruction::Fence:
  case Instruction::LandingPad:
  case Instruction::AtomicRMW:
  case Instruction::AtomicCmpXchg:
  case Instruction::Resume:
    return false; // Misc instructions which have effects
  }
}
