//===- InstCombineAndOrXor.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitAnd, visitOr, and visitXor functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/Intrinsics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Transforms/Utils/CmpInstAnalysis.h"
#include "llvm/Support/ConstantRange.h"
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

/// isFreeToInvert - Return true if the specified value is free to invert (apply
/// ~ to).  This happens in cases where the ~ can be eliminated.
static inline bool isFreeToInvert(Value *V) {
  // ~(~(X)) -> X.
  if (BinaryOperator::isNot(V))
    return true;
  
  // Constants can be considered to be not'ed values.
  if (isa<ConstantInt>(V))
    return true;
  
  // Compares can be inverted if they have a single use.
  if (CmpInst *CI = dyn_cast<CmpInst>(V))
    return CI->hasOneUse();
  
  return false;
}

static inline Value *dyn_castNotVal(Value *V) {
  // If this is not(not(x)) don't return that this is a not: we want the two
  // not's to be folded first.
  if (BinaryOperator::isNot(V)) {
    Value *Operand = BinaryOperator::getNotArgument(V);
    if (!isFreeToInvert(Operand))
      return Operand;
  }
  
  // Constants can be considered to be not'ed values...
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantInt::get(C->getType(), ~C->getValue());
  return 0;
}

/// getFCmpCode - Similar to getICmpCode but for FCmpInst. This encodes a fcmp
/// predicate into a three bit mask. It also returns whether it is an ordered
/// predicate by reference.
static unsigned getFCmpCode(FCmpInst::Predicate CC, bool &isOrdered) {
  isOrdered = false;
  switch (CC) {
  case FCmpInst::FCMP_ORD: isOrdered = true; return 0;  // 000
  case FCmpInst::FCMP_UNO:                   return 0;  // 000
  case FCmpInst::FCMP_OGT: isOrdered = true; return 1;  // 001
  case FCmpInst::FCMP_UGT:                   return 1;  // 001
  case FCmpInst::FCMP_OEQ: isOrdered = true; return 2;  // 010
  case FCmpInst::FCMP_UEQ:                   return 2;  // 010
  case FCmpInst::FCMP_OGE: isOrdered = true; return 3;  // 011
  case FCmpInst::FCMP_UGE:                   return 3;  // 011
  case FCmpInst::FCMP_OLT: isOrdered = true; return 4;  // 100
  case FCmpInst::FCMP_ULT:                   return 4;  // 100
  case FCmpInst::FCMP_ONE: isOrdered = true; return 5;  // 101
  case FCmpInst::FCMP_UNE:                   return 5;  // 101
  case FCmpInst::FCMP_OLE: isOrdered = true; return 6;  // 110
  case FCmpInst::FCMP_ULE:                   return 6;  // 110
    // True -> 7
  default:
    // Not expecting FCMP_FALSE and FCMP_TRUE;
    llvm_unreachable("Unexpected FCmp predicate!");
  }
}

/// getNewICmpValue - This is the complement of getICmpCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand 
/// new ICmp instruction. The sign is passed in to determine which kind
/// of predicate to use in the new icmp instruction.
static Value *getNewICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                              InstCombiner::BuilderTy *Builder) {
  ICmpInst::Predicate NewPred;
  if (Value *NewConstant = getICmpValue(Sign, Code, LHS, RHS, NewPred))
    return NewConstant;
  return Builder->CreateICmp(NewPred, LHS, RHS);
}

/// getFCmpValue - This is the complement of getFCmpCode, which turns an
/// opcode and two operands into either a FCmp instruction. isordered is passed
/// in to determine which kind of predicate to use in the new fcmp instruction.
static Value *getFCmpValue(bool isordered, unsigned code,
                           Value *LHS, Value *RHS,
                           InstCombiner::BuilderTy *Builder) {
  CmpInst::Predicate Pred;
  switch (code) {
  default: llvm_unreachable("Illegal FCmp code!");
  case 0: Pred = isordered ? FCmpInst::FCMP_ORD : FCmpInst::FCMP_UNO; break;
  case 1: Pred = isordered ? FCmpInst::FCMP_OGT : FCmpInst::FCMP_UGT; break;
  case 2: Pred = isordered ? FCmpInst::FCMP_OEQ : FCmpInst::FCMP_UEQ; break;
  case 3: Pred = isordered ? FCmpInst::FCMP_OGE : FCmpInst::FCMP_UGE; break;
  case 4: Pred = isordered ? FCmpInst::FCMP_OLT : FCmpInst::FCMP_ULT; break;
  case 5: Pred = isordered ? FCmpInst::FCMP_ONE : FCmpInst::FCMP_UNE; break;
  case 6: Pred = isordered ? FCmpInst::FCMP_OLE : FCmpInst::FCMP_ULE; break;
  case 7: 
    if (!isordered) return ConstantInt::getTrue(LHS->getContext());
    Pred = FCmpInst::FCMP_ORD; break;
  }
  return Builder->CreateFCmp(Pred, LHS, RHS);
}

// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantInt *OpRHS,
                                    ConstantInt *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!Op->isShift())
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      Value *And = Builder->CreateAnd(X, AndRHS);
      And->takeName(Op);
      return BinaryOperator::CreateXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Op->hasOneUse()){
      if (Together != OpRHS) {
        // (X | C1) & C2 --> (X | (C1&C2)) & C2
        Value *Or = Builder->CreateOr(X, Together);
        Or->takeName(Op);
        return BinaryOperator::CreateAnd(Or, AndRHS);
      }
      
      ConstantInt *TogetherCI = dyn_cast<ConstantInt>(Together);
      if (TogetherCI && !TogetherCI->isZero()){
        // (X | C1) & C2 --> (X & (C2^(C1&C2))) | C1
        // NOTE: This reduces the number of bits set in the & mask, which
        // can expose opportunities for store narrowing.
        Together = ConstantExpr::getXor(AndRHS, Together);
        Value *And = Builder->CreateAnd(X, Together);
        And->takeName(Op);
        return BinaryOperator::CreateOr(And, OpRHS);
      }
    }
    
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      const APInt &AndRHSV = cast<ConstantInt>(AndRHS)->getValue();

      // If there is only one bit set.
      if (AndRHSV.isPowerOf2()) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        const APInt& AddRHS = cast<ConstantInt>(OpRHS)->getValue();

        // Check to see if any bits below the one bit set in AndRHSV are set.
        if ((AddRHS & (AndRHSV-1)) == 0) {
          // If not, the only thing that can effect the output of the AND is
          // the bit specified by AndRHSV.  If that bit is set, the effect of
          // the XOR is to toggle the bit.  If it is clear, then the ADD has
          // no effect.
          if ((AddRHS & AndRHSV) == 0) { // Bit is not set, noop
            TheAnd.setOperand(0, X);
            return &TheAnd;
          } else {
            // Pull the XOR out of the AND.
            Value *NewAnd = Builder->CreateAnd(X, AndRHS);
            NewAnd->takeName(Op);
            return BinaryOperator::CreateXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShlMask(APInt::getHighBitsSet(BitWidth, BitWidth-OpRHSVal));
    ConstantInt *CI = ConstantInt::get(AndRHS->getContext(),
                                       AndRHS->getValue() & ShlMask);

    if (CI->getValue() == ShlMask)
      // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.
    
    if (CI != AndRHS) {                  // Reducing bits set in and.
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  }
  case Instruction::LShr: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
    ConstantInt *CI = ConstantInt::get(Op->getContext(),
                                       AndRHS->getValue() & ShrMask);

    if (CI->getValue() == ShrMask)
      // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);
    
    if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
      return &TheAnd;
    }
    break;
  }
  case Instruction::AShr:
    // Signed shr.
    // See if this is shifting in some sign extension, then masking it out
    // with an and.
    if (Op->hasOneUse()) {
      uint32_t BitWidth = AndRHS->getType()->getBitWidth();
      uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
      APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
      Constant *C = ConstantInt::get(Op->getContext(),
                                     AndRHS->getValue() & ShrMask);
      if (C == AndRHS) {          // Masking out bits shifted in.
        // (Val ashr C1) & C2 -> (Val lshr C1) & C2
        // Make the argument unsigned.
        Value *ShVal = Op->getOperand(0);
        ShVal = Builder->CreateLShr(ShVal, OpRHS, Op->getName());
        return BinaryOperator::CreateAnd(ShVal, AndRHS, TheAnd.getName());
      }
    }
    break;
  }
  return 0;
}


/// InsertRangeTest - Emit a computation of: (V >= Lo && V < Hi) if Inside is
/// true, otherwise (V < Lo || V >= Hi).  In practice, we emit the more efficient
/// (V-Lo) <u Hi-Lo.  This method expects that Lo <= Hi. isSigned indicates
/// whether to treat the V, Lo and HI as signed or not. IB is the location to
/// insert new instructions.
Value *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                     bool isSigned, bool Inside) {
  assert(cast<ConstantInt>(ConstantExpr::getICmp((isSigned ? 
            ICmpInst::ICMP_SLE:ICmpInst::ICMP_ULE), Lo, Hi))->getZExtValue() &&
         "Lo is not <= Hi in range emission code!");
    
  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return ConstantInt::getFalse(V->getContext());

    // V >= Min && V < Hi --> V < Hi
    if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
      ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT);
      return Builder->CreateICmp(pred, V, Hi);
    }

    // Emit V-Lo <u Hi-Lo
    Constant *NegLo = ConstantExpr::getNeg(Lo);
    Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
    Constant *UpperBound = ConstantExpr::getAdd(NegLo, Hi);
    return Builder->CreateICmpULT(Add, UpperBound);
  }

  if (Lo == Hi)  // Trivially true.
    return ConstantInt::getTrue(V->getContext());

  // V < Min || V >= Hi -> V > Hi-1
  Hi = SubOne(cast<ConstantInt>(Hi));
  if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
    ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT);
    return Builder->CreateICmp(pred, V, Hi);
  }

  // Emit V-Lo >u Hi-1-Lo
  // Note that Hi has already had one subtracted from it, above.
  ConstantInt *NegLo = cast<ConstantInt>(ConstantExpr::getNeg(Lo));
  Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
  Constant *LowerBound = ConstantExpr::getAdd(NegLo, Hi);
  return Builder->CreateICmpUGT(Add, LowerBound);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantInt *Val, uint32_t &MB, uint32_t &ME) {
  const APInt& V = Val->getValue();
  uint32_t BitWidth = Val->getType()->getBitWidth();
  if (!APIntOps::isShiftedMask(BitWidth, V)) return false;

  // look for the first zero bit after the run of ones
  MB = BitWidth - ((V - 1) ^ V).countLeadingZeros();
  // look for the first non-zero bit
  ME = V.getActiveBits(); 
  return true;
}

/// FoldLogicalPlusAnd - This is part of an expression (LHS +/- RHS) & Mask,
/// where isSub determines whether the operator is a sub.  If we can fold one of
/// the following xforms:
/// 
/// ((A & N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == Mask
/// ((A | N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
/// ((A ^ N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
///
/// return (A +/- B).
///
Value *InstCombiner::FoldLogicalPlusAnd(Value *LHS, Value *RHS,
                                        ConstantInt *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return 0;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return 0;
  case Instruction::And:
    if (ConstantExpr::getAnd(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getValue().countLeadingZeros() + 
           Mask->getValue().countPopulation()) == 
          Mask->getValue().getBitWidth())
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      uint32_t MB = 0, ME = 0;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        uint32_t BitWidth = cast<IntegerType>(RHS->getType())->getBitWidth();
        APInt Mask(APInt::getLowBitsSet(BitWidth, MB-1));
        if (MaskedValueIsZero(RHS, Mask))
          break;
      }
    }
    return 0;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getValue().countLeadingZeros() + 
         Mask->getValue().countPopulation()) == Mask->getValue().getBitWidth()
        && ConstantExpr::getAnd(N, Mask)->isNullValue())
      break;
    return 0;
  }
  
  if (isSub)
    return Builder->CreateSub(LHSI->getOperand(0), RHS, "fold");
  return Builder->CreateAdd(LHSI->getOperand(0), RHS, "fold");
}

/// enum for classifying (icmp eq (A & B), C) and (icmp ne (A & B), C)
/// One of A and B is considered the mask, the other the value. This is 
/// described as the "AMask" or "BMask" part of the enum. If the enum 
/// contains only "Mask", then both A and B can be considered masks.
/// If A is the mask, then it was proven, that (A & C) == C. This
/// is trivial if C == A, or C == 0. If both A and C are constants, this
/// proof is also easy.
/// For the following explanations we assume that A is the mask.
/// The part "AllOnes" declares, that the comparison is true only 
/// if (A & B) == A, or all bits of A are set in B.
///   Example: (icmp eq (A & 3), 3) -> FoldMskICmp_AMask_AllOnes
/// The part "AllZeroes" declares, that the comparison is true only 
/// if (A & B) == 0, or all bits of A are cleared in B.
///   Example: (icmp eq (A & 3), 0) -> FoldMskICmp_Mask_AllZeroes
/// The part "Mixed" declares, that (A & B) == C and C might or might not 
/// contain any number of one bits and zero bits.
///   Example: (icmp eq (A & 3), 1) -> FoldMskICmp_AMask_Mixed
/// The Part "Not" means, that in above descriptions "==" should be replaced
/// by "!=".
///   Example: (icmp ne (A & 3), 3) -> FoldMskICmp_AMask_NotAllOnes
/// If the mask A contains a single bit, then the following is equivalent:
///    (icmp eq (A & B), A) equals (icmp ne (A & B), 0)
///    (icmp ne (A & B), A) equals (icmp eq (A & B), 0)
enum MaskedICmpType {
  FoldMskICmp_AMask_AllOnes           =     1,
  FoldMskICmp_AMask_NotAllOnes        =     2,
  FoldMskICmp_BMask_AllOnes           =     4,
  FoldMskICmp_BMask_NotAllOnes        =     8,
  FoldMskICmp_Mask_AllZeroes          =    16,
  FoldMskICmp_Mask_NotAllZeroes       =    32,
  FoldMskICmp_AMask_Mixed             =    64,
  FoldMskICmp_AMask_NotMixed          =   128,
  FoldMskICmp_BMask_Mixed             =   256,
  FoldMskICmp_BMask_NotMixed          =   512
};

/// return the set of pattern classes (from MaskedICmpType)
/// that (icmp SCC (A & B), C) satisfies
static unsigned getTypeOfMaskedICmp(Value* A, Value* B, Value* C, 
                                    ICmpInst::Predicate SCC)
{
  ConstantInt *ACst = dyn_cast<ConstantInt>(A);
  ConstantInt *BCst = dyn_cast<ConstantInt>(B);
  ConstantInt *CCst = dyn_cast<ConstantInt>(C);
  bool icmp_eq = (SCC == ICmpInst::ICMP_EQ);
  bool icmp_abit = (ACst != 0 && !ACst->isZero() && 
                    ACst->getValue().isPowerOf2());
  bool icmp_bbit = (BCst != 0 && !BCst->isZero() && 
                    BCst->getValue().isPowerOf2());
  unsigned result = 0;
  if (CCst != 0 && CCst->isZero()) {
    // if C is zero, then both A and B qualify as mask
    result |= (icmp_eq ? (FoldMskICmp_Mask_AllZeroes |
                          FoldMskICmp_Mask_AllZeroes |
                          FoldMskICmp_AMask_Mixed |
                          FoldMskICmp_BMask_Mixed)
                       : (FoldMskICmp_Mask_NotAllZeroes |
                          FoldMskICmp_Mask_NotAllZeroes |
                          FoldMskICmp_AMask_NotMixed |
                          FoldMskICmp_BMask_NotMixed));
    if (icmp_abit)
      result |= (icmp_eq ? (FoldMskICmp_AMask_NotAllOnes |
                            FoldMskICmp_AMask_NotMixed) 
                         : (FoldMskICmp_AMask_AllOnes |
                            FoldMskICmp_AMask_Mixed));
    if (icmp_bbit)
      result |= (icmp_eq ? (FoldMskICmp_BMask_NotAllOnes |
                            FoldMskICmp_BMask_NotMixed) 
                         : (FoldMskICmp_BMask_AllOnes |
                            FoldMskICmp_BMask_Mixed));
    return result;
  }
  if (A == C) {
    result |= (icmp_eq ? (FoldMskICmp_AMask_AllOnes |
                          FoldMskICmp_AMask_Mixed)
                       : (FoldMskICmp_AMask_NotAllOnes |
                          FoldMskICmp_AMask_NotMixed));
    if (icmp_abit)
      result |= (icmp_eq ? (FoldMskICmp_Mask_NotAllZeroes |
                            FoldMskICmp_AMask_NotMixed)
                         : (FoldMskICmp_Mask_AllZeroes |
                            FoldMskICmp_AMask_Mixed));
  }
  else if (ACst != 0 && CCst != 0 &&
        ConstantExpr::getAnd(ACst, CCst) == CCst) {
    result |= (icmp_eq ? FoldMskICmp_AMask_Mixed
                       : FoldMskICmp_AMask_NotMixed);
  }
  if (B == C) 
  {
    result |= (icmp_eq ? (FoldMskICmp_BMask_AllOnes |
                          FoldMskICmp_BMask_Mixed)
                       : (FoldMskICmp_BMask_NotAllOnes |
                          FoldMskICmp_BMask_NotMixed));
    if (icmp_bbit)
      result |= (icmp_eq ? (FoldMskICmp_Mask_NotAllZeroes |
                            FoldMskICmp_BMask_NotMixed) 
                         : (FoldMskICmp_Mask_AllZeroes |
                            FoldMskICmp_BMask_Mixed));
  }
  else if (BCst != 0 && CCst != 0 &&
        ConstantExpr::getAnd(BCst, CCst) == CCst) {
    result |= (icmp_eq ? FoldMskICmp_BMask_Mixed
                       : FoldMskICmp_BMask_NotMixed);
  }
  return result;
}

/// decomposeBitTestICmp - Decompose an icmp into the form ((X & Y) pred Z)
/// if possible. The returned predicate is either == or !=. Returns false if
/// decomposition fails.
static bool decomposeBitTestICmp(const ICmpInst *I, ICmpInst::Predicate &Pred,
                                 Value *&X, Value *&Y, Value *&Z) {
  // X < 0 is equivalent to (X & SignBit) != 0.
  if (I->getPredicate() == ICmpInst::ICMP_SLT)
    if (ConstantInt *C = dyn_cast<ConstantInt>(I->getOperand(1)))
      if (C->isZero()) {
        X = I->getOperand(0);
        Y = ConstantInt::get(I->getContext(),
                             APInt::getSignBit(C->getBitWidth()));
        Pred = ICmpInst::ICMP_NE;
        Z = C;
        return true;
      }

  // X > -1 is equivalent to (X & SignBit) == 0.
  if (I->getPredicate() == ICmpInst::ICMP_SGT)
    if (ConstantInt *C = dyn_cast<ConstantInt>(I->getOperand(1)))
      if (C->isAllOnesValue()) {
        X = I->getOperand(0);
        Y = ConstantInt::get(I->getContext(),
                             APInt::getSignBit(C->getBitWidth()));
        Pred = ICmpInst::ICMP_EQ;
        Z = ConstantInt::getNullValue(C->getType());
        return true;
      }

  return false;
}

/// foldLogOpOfMaskedICmpsHelper:
/// handle (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// return the set of pattern classes (from MaskedICmpType)
/// that both LHS and RHS satisfy
static unsigned foldLogOpOfMaskedICmpsHelper(Value*& A, 
                                             Value*& B, Value*& C,
                                             Value*& D, Value*& E,
                                             ICmpInst *LHS, ICmpInst *RHS,
                                             ICmpInst::Predicate &LHSCC,
                                             ICmpInst::Predicate &RHSCC) {
  if (LHS->getOperand(0)->getType() != RHS->getOperand(0)->getType()) return 0;
  // vectors are not (yet?) supported
  if (LHS->getOperand(0)->getType()->isVectorTy()) return 0;

  // Here comes the tricky part:
  // LHS might be of the form L11 & L12 == X, X == L21 & L22, 
  // and L11 & L12 == L21 & L22. The same goes for RHS.
  // Now we must find those components L** and R**, that are equal, so
  // that we can extract the parameters A, B, C, D, and E for the canonical 
  // above.
  Value *L1 = LHS->getOperand(0);
  Value *L2 = LHS->getOperand(1);
  Value *L11,*L12,*L21,*L22;
  // Check whether the icmp can be decomposed into a bit test.
  if (decomposeBitTestICmp(LHS, LHSCC, L11, L12, L2)) {
    L21 = L22 = L1 = 0;
  } else {
    // Look for ANDs in the LHS icmp.
    if (match(L1, m_And(m_Value(L11), m_Value(L12)))) {
      if (!match(L2, m_And(m_Value(L21), m_Value(L22))))
        L21 = L22 = 0;
    } else {
      if (!match(L2, m_And(m_Value(L11), m_Value(L12))))
        return 0;
      std::swap(L1, L2);
      L21 = L22 = 0;
    }
  }

  // Bail if LHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(LHSCC))
    return 0;

  Value *R1 = RHS->getOperand(0);
  Value *R2 = RHS->getOperand(1);
  Value *R11,*R12;
  bool ok = false;
  if (decomposeBitTestICmp(RHS, RHSCC, R11, R12, R2)) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11;
    } else {
      return 0;
    }
    E = R2; R1 = 0; ok = true;
  } else if (match(R1, m_And(m_Value(R11), m_Value(R12)))) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12; E = R2; ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11; E = R2; ok = true;
    }
  }

  // Bail if RHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(RHSCC))
    return 0;

  // Look for ANDs in on the right side of the RHS icmp.
  if (!ok && match(R2, m_And(m_Value(R11), m_Value(R12)))) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12; E = R1; ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11; E = R1; ok = true;
    } else {
      return 0;
    }
  }
  if (!ok)
    return 0;

  if (L11 == A) {
    B = L12; C = L2;
  }
  else if (L12 == A) {
    B = L11; C = L2;
  }
  else if (L21 == A) {
    B = L22; C = L1;
  }
  else if (L22 == A) {
    B = L21; C = L1;
  }

  unsigned left_type = getTypeOfMaskedICmp(A, B, C, LHSCC);
  unsigned right_type = getTypeOfMaskedICmp(A, D, E, RHSCC);
  return left_type & right_type;
}
/// foldLogOpOfMaskedICmps:
/// try to fold (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// into a single (icmp(A & X) ==/!= Y)
static Value* foldLogOpOfMaskedICmps(ICmpInst *LHS, ICmpInst *RHS,
                                     ICmpInst::Predicate NEWCC,
                                     llvm::InstCombiner::BuilderTy* Builder) {
  Value *A = 0, *B = 0, *C = 0, *D = 0, *E = 0;
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();
  unsigned mask = foldLogOpOfMaskedICmpsHelper(A, B, C, D, E, LHS, RHS,
                                               LHSCC, RHSCC);
  if (mask == 0) return 0;
  assert(ICmpInst::isEquality(LHSCC) && ICmpInst::isEquality(RHSCC) &&
         "foldLogOpOfMaskedICmpsHelper must return an equality predicate.");

  if (NEWCC == ICmpInst::ICMP_NE)
    mask >>= 1; // treat "Not"-states as normal states

  if (mask & FoldMskICmp_Mask_AllZeroes) {
    // (icmp eq (A & B), 0) & (icmp eq (A & D), 0) 
    // -> (icmp eq (A & (B|D)), 0)
    Value* newOr = Builder->CreateOr(B, D);
    Value* newAnd = Builder->CreateAnd(A, newOr);
    // we can't use C as zero, because we might actually handle
    //   (icmp ne (A & B), B) & (icmp ne (A & D), D) 
    // with B and D, having a single bit set
    Value* zero = Constant::getNullValue(A->getType());
    return Builder->CreateICmp(NEWCC, newAnd, zero);
  }
  else if (mask & FoldMskICmp_BMask_AllOnes) {
    // (icmp eq (A & B), B) & (icmp eq (A & D), D) 
    // -> (icmp eq (A & (B|D)), (B|D))
    Value* newOr = Builder->CreateOr(B, D);
    Value* newAnd = Builder->CreateAnd(A, newOr);
    return Builder->CreateICmp(NEWCC, newAnd, newOr);
  }     
  else if (mask & FoldMskICmp_AMask_AllOnes) {
    // (icmp eq (A & B), A) & (icmp eq (A & D), A) 
    // -> (icmp eq (A & (B&D)), A)
    Value* newAnd1 = Builder->CreateAnd(B, D);
    Value* newAnd = Builder->CreateAnd(A, newAnd1);
    return Builder->CreateICmp(NEWCC, newAnd, A);
  }
  else if (mask & FoldMskICmp_BMask_Mixed) {
    // (icmp eq (A & B), C) & (icmp eq (A & D), E) 
    // We already know that B & C == C && D & E == E.
    // If we can prove that (B & D) & (C ^ E) == 0, that is, the bits of
    // C and E, which are shared by both the mask B and the mask D, don't
    // contradict, then we can transform to
    // -> (icmp eq (A & (B|D)), (C|E))
    // Currently, we only handle the case of B, C, D, and E being constant.
    ConstantInt *BCst = dyn_cast<ConstantInt>(B);
    if (BCst == 0) return 0;
    ConstantInt *DCst = dyn_cast<ConstantInt>(D);
    if (DCst == 0) return 0;
    // we can't simply use C and E, because we might actually handle
    //   (icmp ne (A & B), B) & (icmp eq (A & D), D) 
    // with B and D, having a single bit set

    ConstantInt *CCst = dyn_cast<ConstantInt>(C);
    if (CCst == 0) return 0;
    if (LHSCC != NEWCC)
      CCst = dyn_cast<ConstantInt>( ConstantExpr::getXor(BCst, CCst) );
    ConstantInt *ECst = dyn_cast<ConstantInt>(E);
    if (ECst == 0) return 0;
    if (RHSCC != NEWCC)
      ECst = dyn_cast<ConstantInt>( ConstantExpr::getXor(DCst, ECst) );
    ConstantInt* MCst = dyn_cast<ConstantInt>(
      ConstantExpr::getAnd(ConstantExpr::getAnd(BCst, DCst),
                           ConstantExpr::getXor(CCst, ECst)) );
    // if there is a conflict we should actually return a false for the
    // whole construct
    if (!MCst->isZero())
      return 0;
    Value *newOr1 = Builder->CreateOr(B, D);
    Value *newOr2 = ConstantExpr::getOr(CCst, ECst);
    Value *newAnd = Builder->CreateAnd(A, newOr1);
    return Builder->CreateICmp(NEWCC, newAnd, newOr2);
  }
  return 0;
}

/// FoldAndOfICmps - Fold (icmp)&(icmp) if possible.
Value *InstCombiner::FoldAndOfICmps(ICmpInst *LHS, ICmpInst *RHS) {
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) & getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(isSigned, Code, Op0, Op1, Builder);
    }
  }

  // handle (roughly):  (icmp eq (A & B), C) & (icmp eq (A & D), E)
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, ICmpInst::ICMP_EQ, Builder))
    return V;
  
  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (LHSCst == 0 || RHSCst == 0) return 0;
  
  if (LHSCst == RHSCst && LHSCC == RHSCC) {
    // (icmp ult A, C) & (icmp ult B, C) --> (icmp ult (A|B), C)
    // where C is a power of 2
    if (LHSCC == ICmpInst::ICMP_ULT &&
        LHSCst->getValue().isPowerOf2()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }
    
    // (icmp eq A, 0) & (icmp eq B, 0) --> (icmp eq (A|B), 0)
    if (LHSCC == ICmpInst::ICMP_EQ && LHSCst->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }
  }

  // (trunc x) == C1 & (and x, CA) == C2 -> (and x, CA|CMAX) == C1|C2
  // where CMAX is the all ones value for the truncated type,
  // iff the lower bits of C2 and CA are zero.
  if (LHSCC == ICmpInst::ICMP_EQ && LHSCC == RHSCC &&
      LHS->hasOneUse() && RHS->hasOneUse()) {
    Value *V;
    ConstantInt *AndCst, *SmallCst = 0, *BigCst = 0;

    // (trunc x) == C1 & (and x, CA) == C2
    if (match(Val2, m_Trunc(m_Value(V))) &&
        match(Val, m_And(m_Specific(V), m_ConstantInt(AndCst)))) {
      SmallCst = RHSCst;
      BigCst = LHSCst;
    }
    // (and x, CA) == C2 & (trunc x) == C1
    else if (match(Val, m_Trunc(m_Value(V))) &&
             match(Val2, m_And(m_Specific(V), m_ConstantInt(AndCst)))) {
      SmallCst = LHSCst;
      BigCst = RHSCst;
    }

    if (SmallCst && BigCst) {
      unsigned BigBitSize = BigCst->getType()->getBitWidth();
      unsigned SmallBitSize = SmallCst->getType()->getBitWidth();

      // Check that the low bits are zero.
      APInt Low = APInt::getLowBitsSet(BigBitSize, SmallBitSize);
      if ((Low & AndCst->getValue()) == 0 && (Low & BigCst->getValue()) == 0) {
        Value *NewAnd = Builder->CreateAnd(V, Low | AndCst->getValue());
        APInt N = SmallCst->getValue().zext(BigBitSize) | BigCst->getValue();
        Value *NewVal = ConstantInt::get(AndCst->getType()->getContext(), N);
        return Builder->CreateICmp(LHSCC, NewAnd, NewVal);
      }
    }
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) & (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;

  // Make a constant range that's the intersection of the two icmp ranges.
  // If the intersection is empty, we know that the result is false.
  ConstantRange LHSRange = 
    ConstantRange::makeICmpRegion(LHSCC, LHSCst->getValue());
  ConstantRange RHSRange = 
    ConstantRange::makeICmpRegion(RHSCC, RHSCst->getValue());

  if (LHSRange.intersectWith(RHSRange).isEmptySet())
    return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);

  // We can't fold (ugt x, C) & (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
    
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
    
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }

  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and and'ing the result
  // together.  Because of the above check, we know that we only have
  // icmp eq, icmp ne, icmp [su]lt, and icmp [SU]gt here. We also know 
  // (from the icmp folding check above), that the two constants 
  // are not equal and that the larger constant is on the RHS
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_NE:         // (X == 13 & X != 15) -> X == 13
    case ICmpInst::ICMP_ULT:        // (X == 13 & X <  15) -> X == 13
    case ICmpInst::ICMP_SLT:        // (X == 13 & X <  15) -> X == 13
      return LHS;
    }
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_ULT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X u< 14) -> X < 13
        return Builder->CreateICmpULT(Val, LHSCst);
      break;                        // (X != 13 & X u< 15) -> no change
    case ICmpInst::ICMP_SLT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X s< 14) -> X < 13
        return Builder->CreateICmpSLT(Val, LHSCst);
      break;                        // (X != 13 & X s< 15) -> no change
    case ICmpInst::ICMP_EQ:         // (X != 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X != 13 & X u> 15) -> X u> 15
    case ICmpInst::ICMP_SGT:        // (X != 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_NE:
      if (LHSCst == SubOne(RHSCst)){// (X != 13 & X != 14) -> X-13 >u 1
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        return Builder->CreateICmpUGT(Add, ConstantInt::get(Add->getType(), 1));
      }
      break;                        // (X != 13 & X != 15) -> no change
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X u< 13 & X u> 15) -> false
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    case ICmpInst::ICMP_SGT:        // (X u< 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 & X != 15) -> X u< 13
    case ICmpInst::ICMP_ULT:        // (X u< 13 & X u< 15) -> X u< 13
      return LHS;
    case ICmpInst::ICMP_SLT:        // (X u< 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_UGT:        // (X s< 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 & X != 15) -> X < 13
    case ICmpInst::ICMP_SLT:        // (X s< 13 & X s< 15) -> X < 13
      return LHS;
    case ICmpInst::ICMP_ULT:        // (X s< 13 & X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X u> 13 & X u> 15) -> X u> 15
      return RHS;
    case ICmpInst::ICMP_SGT:        // (X u> 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X u> 13 & X != 14) -> X u> 14
        return Builder->CreateICmp(LHSCC, Val, RHSCst);
      break;                        // (X u> 13 & X != 15) -> no change
    case ICmpInst::ICMP_ULT:        // (X u> 13 & X u< 15) -> (X-14) <u 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, false, true);
    case ICmpInst::ICMP_SLT:        // (X u> 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_SGT:        // (X s> 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_UGT:        // (X s> 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X s> 13 & X != 14) -> X s> 14
        return Builder->CreateICmp(LHSCC, Val, RHSCst);
      break;                        // (X s> 13 & X != 15) -> no change
    case ICmpInst::ICMP_SLT:        // (X s> 13 & X s< 15) -> (X-14) s< 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, true, true);
    case ICmpInst::ICMP_ULT:        // (X s> 13 & X u< 15) -> no change
      break;
    }
    break;
  }
 
  return 0;
}

/// FoldAndOfFCmps - Optimize (fcmp)&(fcmp).  NOTE: Unlike the rest of
/// instcombine, this returns a Value which should already be inserted into the
/// function.
Value *InstCombiner::FoldAndOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  if (LHS->getPredicate() == FCmpInst::FCMP_ORD &&
      RHS->getPredicate() == FCmpInst::FCMP_ORD) {
    // (fcmp ord x, c) & (fcmp ord y, c)  -> (fcmp ord x, y)
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // false.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return ConstantInt::getFalse(LHS->getContext());
        return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
      }
    
    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp ord x,x" is "fcmp ord x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
    return 0;
  }
  
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();
  
  
  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }
  
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) & (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return Builder->CreateFCmp((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
    if (Op0CC == FCmpInst::FCMP_FALSE || Op1CC == FCmpInst::FCMP_FALSE)
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    if (Op0CC == FCmpInst::FCMP_TRUE)
      return RHS;
    if (Op1CC == FCmpInst::FCMP_TRUE)
      return LHS;
    
    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    // uno && ord -> false
    if (Op0Pred == 0 && Op1Pred == 0 && Op0Ordered != Op1Ordered)
        return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    if (Op1Pred == 0) {
      std::swap(LHS, RHS);
      std::swap(Op0Pred, Op1Pred);
      std::swap(Op0Ordered, Op1Ordered);
    }
    if (Op0Pred == 0) {
      // uno && ueq -> uno && (uno || eq) -> uno
      // ord && olt -> ord && (ord && lt) -> olt
      if (!Op0Ordered && (Op0Ordered == Op1Ordered))
        return LHS;
      if (Op0Ordered && (Op0Ordered == Op1Ordered))
        return RHS;
      
      // uno && oeq -> uno && (ord && eq) -> false
      if (!Op0Ordered)
        return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
      // ord && ueq -> ord && (uno || eq) -> oeq
      return getFCmpValue(true, Op1Pred, Op0LHS, Op0RHS, Builder);
    }
  }

  return 0;
}


Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyAndInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // (A|B)&(A|C) -> A|(B&C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;  

  if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(Op1)) {
    const APInt &AndRHSMask = AndRHS->getValue();

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      Value *Op0LHS = Op0I->getOperand(0);
      Value *Op0RHS = Op0I->getOperand(1);
      switch (Op0I->getOpcode()) {
      default: break;
      case Instruction::Xor:
      case Instruction::Or: {
        // If the mask is only needed on one incoming arm, push it up.
        if (!Op0I->hasOneUse()) break;
          
        APInt NotAndRHS(~AndRHSMask);
        if (MaskedValueIsZero(Op0LHS, NotAndRHS)) {
          // Not masking anything out for the LHS, move to RHS.
          Value *NewRHS = Builder->CreateAnd(Op0RHS, AndRHS,
                                             Op0RHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), Op0LHS, NewRHS);
        }
        if (!isa<Constant>(Op0RHS) &&
            MaskedValueIsZero(Op0RHS, NotAndRHS)) {
          // Not masking anything out for the RHS, move to LHS.
          Value *NewLHS = Builder->CreateAnd(Op0LHS, AndRHS,
                                             Op0LHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), NewLHS, Op0RHS);
        }

        break;
      }
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::CreateAnd(V, AndRHS);

        // (A - N) & AndRHS -> -N & AndRHS iff A&AndRHS==0 and AndRHS
        // has 1's for all bits that the subtraction with A might affect.
        if (Op0I->hasOneUse() && !match(Op0LHS, m_Zero())) {
          uint32_t BitWidth = AndRHSMask.getBitWidth();
          uint32_t Zeros = AndRHSMask.countLeadingZeros();
          APInt Mask = APInt::getLowBitsSet(BitWidth, BitWidth - Zeros);

          if (MaskedValueIsZero(Op0LHS, Mask)) {
            Value *NewNeg = Builder->CreateNeg(Op0RHS);
            return BinaryOperator::CreateAnd(NewNeg, AndRHS);
          }
        }
        break;

      case Instruction::Shl:
      case Instruction::LShr:
        // (1 << x) & 1 --> zext(x == 0)
        // (1 >> x) & 1 --> zext(x == 0)
        if (AndRHSMask == 1 && Op0LHS == AndRHS) {
          Value *NewICmp =
            Builder->CreateICmpEQ(Op0RHS, Constant::getNullValue(I.getType()));
          return new ZExtInst(NewICmp, I.getType());
        }
        break;
      }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    }
    
    // If this is an integer truncation, and if the source is an 'and' with
    // immediate, transform it.  This frequently occurs for bitfield accesses.
    {
      Value *X = 0; ConstantInt *YC = 0;
      if (match(Op0, m_Trunc(m_And(m_Value(X), m_ConstantInt(YC))))) {
        // Change: and (trunc (and X, YC) to T), C2
        // into  : and (trunc X to T), trunc(YC) & C2
        // This will fold the two constants together, which may allow 
        // other simplifications.
        Value *NewCast = Builder->CreateTrunc(X, I.getType(), "and.shrunk");
        Constant *C3 = ConstantExpr::getTrunc(YC, I.getType());
        C3 = ConstantExpr::getAnd(C3, AndRHS);
        return BinaryOperator::CreateAnd(NewCast, C3);
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }


  // (~A & ~B) == (~(A | B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *Or = Builder->CreateOr(Op0NotVal, Op1NotVal,
                                      I.getName()+".demorgan");
        return BinaryOperator::CreateNot(Or);
      }
  
  {
    Value *A = 0, *B = 0, *C = 0, *D = 0;
    // (A|B) & ~(A&B) -> A^B
    if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);
    
    // ~(A&B) & (A|B) -> A^B
    if (match(Op1, m_Or(m_Value(A), m_Value(B))) &&
        match(Op0, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);
    
    // A&(A^B) => A & ~B
    {
      Value *tmpOp0 = Op0;
      Value *tmpOp1 = Op1;
      if (Op0->hasOneUse() &&
          match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
        if (A == Op1 || B == Op1 ) {
          tmpOp1 = Op0;
          tmpOp0 = Op1;
          // Simplify below
        }
      }

      if (tmpOp1->hasOneUse() &&
          match(tmpOp1, m_Xor(m_Value(A), m_Value(B)))) {
        if (B == tmpOp0) {
          std::swap(A, B);
        }
        // Notice that the patten (A&(~B)) is actually (A&(-1^B)), so if
        // A is originally -1 (or a vector of -1 and undefs), then we enter
        // an endless loop. By checking that A is non-constant we ensure that
        // we will never get to the loop.
        if (A == tmpOp0 && !isa<Constant>(A)) // A&(A^B) -> A & ~B
          return BinaryOperator::CreateAnd(A, Builder->CreateNot(B));
      }
    }

    // (A&((~A)|B)) -> A&B
    if (match(Op0, m_Or(m_Not(m_Specific(Op1)), m_Value(A))) ||
        match(Op0, m_Or(m_Value(A), m_Not(m_Specific(Op1)))))
      return BinaryOperator::CreateAnd(A, Op1);
    if (match(Op1, m_Or(m_Not(m_Specific(Op0)), m_Value(A))) ||
        match(Op1, m_Or(m_Value(A), m_Not(m_Specific(Op0)))))
      return BinaryOperator::CreateAnd(A, Op0);
  }
  
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0))
      if (Value *Res = FoldAndOfICmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);
  
  // If and'ing two fcmp, try combine them into one.
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldAndOfFCmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);
  
  
  // fold (and (cast A), (cast B)) -> (cast (and A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0))
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1)) {
      Type *SrcTy = Op0C->getOperand(0)->getType();
      if (Op0C->getOpcode() == Op1C->getOpcode() && // same cast kind ?
          SrcTy == Op1C->getOperand(0)->getType() &&
          SrcTy->isIntOrIntVectorTy()) {
        Value *Op0COp = Op0C->getOperand(0), *Op1COp = Op1C->getOperand(0);
        
        // Only do this if the casts both really cause code to be generated.
        if (ShouldOptimizeCast(Op0C->getOpcode(), Op0COp, I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1COp, I.getType())) {
          Value *NewOp = Builder->CreateAnd(Op0COp, Op1COp, I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
        
        // If this is and(cast(icmp), cast(icmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1COp))
          if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0COp))
            if (Value *Res = FoldAndOfICmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
        
        // If this is and(cast(fcmp), cast(fcmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (FCmpInst *RHS = dyn_cast<FCmpInst>(Op1COp))
          if (FCmpInst *LHS = dyn_cast<FCmpInst>(Op0COp))
            if (Value *Res = FoldAndOfFCmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
      }
    }
    
  // (X >> Z) & (Y >> Z)  -> (X&Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Value *NewOp =
          Builder->CreateAnd(SI0->getOperand(0), SI1->getOperand(0),
                             SI0->getName());
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  return Changed ? &I : 0;
}

/// CollectBSwapParts - Analyze the specified subexpression and see if it is
/// capable of providing pieces of a bswap.  The subexpression provides pieces
/// of a bswap if it is proven that each of the non-zero bytes in the output of
/// the expression came from the corresponding "byte swapped" byte in some other
/// value.  For example, if the current subexpression is "(shl i32 %X, 24)" then
/// we know that the expression deposits the low byte of %X into the high byte
/// of the bswap result and that all other bytes are zero.  This expression is
/// accepted, the high byte of ByteValues is set to X to indicate a correct
/// match.
///
/// This function returns true if the match was unsuccessful and false if so.
/// On entry to the function the "OverallLeftShift" is a signed integer value
/// indicating the number of bytes that the subexpression is later shifted.  For
/// example, if the expression is later right shifted by 16 bits, the
/// OverallLeftShift value would be -2 on entry.  This is used to specify which
/// byte of ByteValues is actually being set.
///
/// Similarly, ByteMask is a bitmask where a bit is clear if its corresponding
/// byte is masked to zero by a user.  For example, in (X & 255), X will be
/// processed with a bytemask of 1.  Because bytemask is 32-bits, this limits
/// this function to working on up to 32-byte (256 bit) values.  ByteMask is
/// always in the local (OverallLeftShift) coordinate space.
///
static bool CollectBSwapParts(Value *V, int OverallLeftShift, uint32_t ByteMask,
                              SmallVector<Value*, 8> &ByteValues) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an or instruction, it may be an inner node of the bswap.
    if (I->getOpcode() == Instruction::Or) {
      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues) ||
             CollectBSwapParts(I->getOperand(1), OverallLeftShift, ByteMask,
                               ByteValues);
    }
  
    // If this is a logical shift by a constant multiple of 8, recurse with
    // OverallLeftShift and ByteMask adjusted.
    if (I->isLogicalShift() && isa<ConstantInt>(I->getOperand(1))) {
      unsigned ShAmt = 
        cast<ConstantInt>(I->getOperand(1))->getLimitedValue(~0U);
      // Ensure the shift amount is defined and of a byte value.
      if ((ShAmt & 7) || (ShAmt > 8*ByteValues.size()))
        return true;

      unsigned ByteShift = ShAmt >> 3;
      if (I->getOpcode() == Instruction::Shl) {
        // X << 2 -> collect(X, +2)
        OverallLeftShift += ByteShift;
        ByteMask >>= ByteShift;
      } else {
        // X >>u 2 -> collect(X, -2)
        OverallLeftShift -= ByteShift;
        ByteMask <<= ByteShift;
        ByteMask &= (~0U >> (32-ByteValues.size()));
      }

      if (OverallLeftShift >= (int)ByteValues.size()) return true;
      if (OverallLeftShift <= -(int)ByteValues.size()) return true;

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }

    // If this is a logical 'and' with a mask that clears bytes, clear the
    // corresponding bytes in ByteMask.
    if (I->getOpcode() == Instruction::And &&
        isa<ConstantInt>(I->getOperand(1))) {
      // Scan every byte of the and mask, seeing if the byte is either 0 or 255.
      unsigned NumBytes = ByteValues.size();
      APInt Byte(I->getType()->getPrimitiveSizeInBits(), 255);
      const APInt &AndMask = cast<ConstantInt>(I->getOperand(1))->getValue();
      
      for (unsigned i = 0; i != NumBytes; ++i, Byte <<= 8) {
        // If this byte is masked out by a later operation, we don't care what
        // the and mask is.
        if ((ByteMask & (1 << i)) == 0)
          continue;
        
        // If the AndMask is all zeros for this byte, clear the bit.
        APInt MaskB = AndMask & Byte;
        if (MaskB == 0) {
          ByteMask &= ~(1U << i);
          continue;
        }
        
        // If the AndMask is not all ones for this byte, it's not a bytezap.
        if (MaskB != Byte)
          return true;

        // Otherwise, this byte is kept.
      }

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }
  }
  
  // Okay, we got to something that isn't a shift, 'or' or 'and'.  This must be
  // the input value to the bswap.  Some observations: 1) if more than one byte
  // is demanded from this input, then it could not be successfully assembled
  // into a byteswap.  At least one of the two bytes would not be aligned with
  // their ultimate destination.
  if (!isPowerOf2_32(ByteMask)) return true;
  unsigned InputByteNo = CountTrailingZeros_32(ByteMask);
  
  // 2) The input and ultimate destinations must line up: if byte 3 of an i32
  // is demanded, it needs to go into byte 0 of the result.  This means that the
  // byte needs to be shifted until it lands in the right byte bucket.  The
  // shift amount depends on the position: if the byte is coming from the high
  // part of the value (e.g. byte 3) then it must be shifted right.  If from the
  // low part, it must be shifted left.
  unsigned DestByteNo = InputByteNo + OverallLeftShift;
  if (ByteValues.size()-1-DestByteNo != InputByteNo)
    return true;
  
  // If the destination byte value is already defined, the values are or'd
  // together, which isn't a bswap (unless it's an or of the same bits).
  if (ByteValues[DestByteNo] && ByteValues[DestByteNo] != V)
    return true;
  ByteValues[DestByteNo] = V;
  return false;
}

/// MatchBSwap - Given an OR instruction, check to see if this is a bswap idiom.
/// If so, insert the new bswap intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  IntegerType *ITy = dyn_cast<IntegerType>(I.getType());
  if (!ITy || ITy->getBitWidth() % 16 || 
      // ByteMask only allows up to 32-byte values.
      ITy->getBitWidth() > 32*8) 
    return 0;   // Can only bswap pairs of bytes.  Can't do vectors.
  
  /// ByteValues - For each byte of the result, we keep track of which value
  /// defines each byte.
  SmallVector<Value*, 8> ByteValues;
  ByteValues.resize(ITy->getBitWidth()/8);
    
  // Try to find all the pieces corresponding to the bswap.
  uint32_t ByteMask = ~0U >> (32-ByteValues.size());
  if (CollectBSwapParts(&I, 0, ByteMask, ByteValues))
    return 0;
  
  // Check to see if all of the bytes come from the same value.
  Value *V = ByteValues[0];
  if (V == 0) return 0;  // Didn't find a byte?  Must be zero.
  
  // Check to make sure that all of the bytes come from the same value.
  for (unsigned i = 1, e = ByteValues.size(); i != e; ++i)
    if (ByteValues[i] != V)
      return 0;
  Module *M = I.getParent()->getParent()->getParent();
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, ITy);
  return CallInst::Create(F, V);
}

/// MatchSelectFromAndOr - We have an expression of the form (A&C)|(B&D).  Check
/// If A is (cond?-1:0) and either B or D is ~(cond?-1,0) or (cond?0,-1), then
/// we can simplify this expression to "cond ? C : D or B".
static Instruction *MatchSelectFromAndOr(Value *A, Value *B,
                                         Value *C, Value *D) {
  // If A is not a select of -1/0, this cannot match.
  Value *Cond = 0;
  if (!match(A, m_SExt(m_Value(Cond))) ||
      !Cond->getType()->isIntegerTy(1))
    return 0;

  // ((cond?-1:0)&C) | (B&(cond?0:-1)) -> cond ? C : B.
  if (match(D, m_Not(m_SExt(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);
  if (match(D, m_SExt(m_Not(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);
  
  // ((cond?-1:0)&C) | ((cond?0:-1)&D) -> cond ? C : D.
  if (match(B, m_Not(m_SExt(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  if (match(B, m_SExt(m_Not(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  return 0;
}

/// FoldOrOfICmps - Fold (icmp)|(icmp) if possible.
Value *InstCombiner::FoldOrOfICmps(ICmpInst *LHS, ICmpInst *RHS) {
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) | getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(isSigned, Code, Op0, Op1, Builder);
    }
  }

  // handle (roughly):
  // (icmp ne (A & B), C) | (icmp ne (A & D), E)
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, ICmpInst::ICMP_NE, Builder))
    return V;

  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (LHSCst == 0 || RHSCst == 0) return 0;

  if (LHSCst == RHSCst && LHSCC == RHSCC) {
    // (icmp ne A, 0) | (icmp ne B, 0) --> (icmp ne (A|B), 0)
    if (LHSCC == ICmpInst::ICMP_NE && LHSCst->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }
  }

  // (icmp ult (X + CA), C1) | (icmp eq X, C2) -> (icmp ule (X + CA), C1)
  //   iff C2 + CA == C1.
  if (LHSCC == ICmpInst::ICMP_ULT && RHSCC == ICmpInst::ICMP_EQ) {
    ConstantInt *AddCst;
    if (match(Val, m_Add(m_Specific(Val2), m_ConstantInt(AddCst))))
      if (RHSCst->getValue() + AddCst->getValue() == LHSCst->getValue())
        return Builder->CreateICmpULE(Val, LHSCst);
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) | (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;
  
  // We can't fold (ugt x, C) | (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
  
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
  
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }
  
  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and or'ing the result
  // together.  Because of the above check, we know that we only have
  // ICMP_EQ, ICMP_NE, ICMP_LT, and ICMP_GT here. We also know (from the
  // icmp folding check above), that the two constants are not
  // equal.
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:
      if (LHSCst == SubOne(RHSCst)) {
        // (X == 13 | X == 14) -> X-13 <u 2
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        AddCST = ConstantExpr::getSub(AddOne(RHSCst), LHSCst);
        return Builder->CreateICmpULT(Add, AddCST);
      }
      break;                         // (X == 13 | X == 15) -> no change
    case ICmpInst::ICMP_UGT:         // (X == 13 | X u> 14) -> no change
    case ICmpInst::ICMP_SGT:         // (X == 13 | X s> 14) -> no change
      break;
    case ICmpInst::ICMP_NE:          // (X == 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:         // (X == 13 | X u< 15) -> X u< 15
    case ICmpInst::ICMP_SLT:         // (X == 13 | X s< 15) -> X s< 15
      return RHS;
    }
    break;
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:          // (X != 13 | X == 15) -> X != 13
    case ICmpInst::ICMP_UGT:         // (X != 13 | X u> 15) -> X != 13
    case ICmpInst::ICMP_SGT:         // (X != 13 | X s> 15) -> X != 13
      return LHS;
    case ICmpInst::ICMP_NE:          // (X != 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:         // (X != 13 | X u< 15) -> true
    case ICmpInst::ICMP_SLT:         // (X != 13 | X s< 15) -> true
      return ConstantInt::getTrue(LHS->getContext());
    }
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_UGT:        // (X u< 13 | X u> 15) -> (X-13) u> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(false))
        return LHS;
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), false, false);
    case ICmpInst::ICMP_SGT:        // (X u< 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:        // (X u< 13 | X u< 15) -> X u< 15
      return RHS;
    case ICmpInst::ICMP_SLT:        // (X u< 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_SGT:        // (X s< 13 | X s> 15) -> (X-13) s> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(true))
        return LHS;
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), true, false);
    case ICmpInst::ICMP_UGT:        // (X s< 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_SLT:        // (X s< 13 | X s< 15) -> X s< 15
      return RHS;
    case ICmpInst::ICMP_ULT:        // (X s< 13 | X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 | X == 15) -> X u> 13
    case ICmpInst::ICMP_UGT:        // (X u> 13 | X u> 15) -> X u> 13
      return LHS;
    case ICmpInst::ICMP_SGT:        // (X u> 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u> 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:        // (X u> 13 | X u< 15) -> true
      return ConstantInt::getTrue(LHS->getContext());
    case ICmpInst::ICMP_SLT:        // (X u> 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 | X == 15) -> X > 13
    case ICmpInst::ICMP_SGT:        // (X s> 13 | X s> 15) -> X > 13
      return LHS;
    case ICmpInst::ICMP_UGT:        // (X s> 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s> 13 | X != 15) -> true
    case ICmpInst::ICMP_SLT:        // (X s> 13 | X s< 15) -> true
      return ConstantInt::getTrue(LHS->getContext());
    case ICmpInst::ICMP_ULT:        // (X s> 13 | X u< 15) -> no change
      break;
    }
    break;
  }
  return 0;
}

/// FoldOrOfFCmps - Optimize (fcmp)|(fcmp).  NOTE: Unlike the rest of
/// instcombine, this returns a Value which should already be inserted into the
/// function.
Value *InstCombiner::FoldOrOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  if (LHS->getPredicate() == FCmpInst::FCMP_UNO &&
      RHS->getPredicate() == FCmpInst::FCMP_UNO && 
      LHS->getOperand(0)->getType() == RHS->getOperand(0)->getType()) {
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // true.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return ConstantInt::getTrue(LHS->getContext());
        
        // Otherwise, no need to compare the two constants, compare the
        // rest.
        return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));
      }
    
    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp uno x,x" is "fcmp uno x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));
    
    return 0;
  }
  
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();
  
  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) | (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return Builder->CreateFCmp((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
    if (Op0CC == FCmpInst::FCMP_TRUE || Op1CC == FCmpInst::FCMP_TRUE)
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
    if (Op0CC == FCmpInst::FCMP_FALSE)
      return RHS;
    if (Op1CC == FCmpInst::FCMP_FALSE)
      return LHS;
    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    if (Op0Ordered == Op1Ordered) {
      // If both are ordered or unordered, return a new fcmp with
      // or'ed predicates.
      return getFCmpValue(Op0Ordered, Op0Pred|Op1Pred, Op0LHS, Op0RHS, Builder);
    }
  }
  return 0;
}

/// FoldOrWithConstants - This helper function folds:
///
///     ((A | B) & C1) | (B & C2)
///
/// into:
/// 
///     (A & C1) | B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                               Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1) return 0;

  Value *V1 = 0;
  ConstantInt *CI2 = 0;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2)))) return 0;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue()) return 0;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd((V1 == A) ? B : A, CI1);
    return BinaryOperator::CreateOr(NewOp, V1);
  }

  return 0;
}

Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyOrInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // (A&B)|(A&C) -> A&(B|C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    ConstantInt *C1 = 0; Value *X = 0;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    // iff (C1 & C2) == 0.
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) &&
        (RHS->getValue() & C1->getValue()) != 0 &&
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateAnd(Or, 
                         ConstantInt::get(I.getContext(),
                                          RHS->getValue() | C1->getValue()));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) &&
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateXor(Or,
                 ConstantInt::get(I.getContext(),
                                  C1->getValue() & ~RHS->getValue()));
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *A = 0, *B = 0;
  ConstantInt *C1 = 0, *C2 = 0;

  // (A | B) | C  and  A | (B | C)                  -> bswap if possible.
  // (A >> B) | (C << D)  and  (A << B) | (B >> C)  -> bswap if possible.
  if (match(Op0, m_Or(m_Value(), m_Value())) ||
      match(Op1, m_Or(m_Value(), m_Value())) ||
      (match(Op0, m_LogicalShift(m_Value(), m_Value())) &&
       match(Op1, m_LogicalShift(m_Value(), m_Value())))) {
    if (Instruction *BSwap = MatchBSwap(I))
      return BSwap;
  }
  
  // (X^C)|Y -> (X|Y)^C iff Y&C == 0
  if (Op0->hasOneUse() &&
      match(Op0, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op1, C1->getValue())) {
    Value *NOr = Builder->CreateOr(A, Op1);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() &&
      match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1->getValue())) {
    Value *NOr = Builder->CreateOr(A, Op0);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // (A & C)|(B & D)
  Value *C = 0, *D = 0;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    Value *V1 = 0, *V2 = 0;
    C1 = dyn_cast<ConstantInt>(C);
    C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2) {  // (A & C1)|(B & C2)
      // If we have: ((V + N) & C1) | (V & C2)
      // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
      // replace with V+N.
      if (C1->getValue() == ~C2->getValue()) {
        if ((C2->getValue() & (C2->getValue()+1)) == 0 && // C2 == 0+1+
            match(A, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == B && MaskedValueIsZero(V2, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
          if (V2 == B && MaskedValueIsZero(V1, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
        }
        // Or commutes, try both ways.
        if ((C1->getValue() & (C1->getValue()+1)) == 0 &&
            match(B, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == A && MaskedValueIsZero(V2, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
          if (V2 == A && MaskedValueIsZero(V1, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
        }
      }
      
      if ((C1->getValue() & C2->getValue()) == 0) {
        // ((V | N) & C1) | (V & C2) --> (V|N) & (C1|C2)
        // iff (C1&C2) == 0 and (N&~C1) == 0
        if (match(A, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == B && MaskedValueIsZero(V2, ~C1->getValue())) ||  // (V|N)
             (V2 == B && MaskedValueIsZero(V1, ~C1->getValue()))))   // (N|V)
          return BinaryOperator::CreateAnd(A,
                               ConstantInt::get(A->getContext(),
                                                C1->getValue()|C2->getValue()));
        // Or commutes, try both ways.
        if (match(B, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == A && MaskedValueIsZero(V2, ~C2->getValue())) ||  // (V|N)
             (V2 == A && MaskedValueIsZero(V1, ~C2->getValue()))))   // (N|V)
          return BinaryOperator::CreateAnd(B,
                               ConstantInt::get(B->getContext(),
                                                C1->getValue()|C2->getValue()));
        
        // ((V|C3)&C1) | ((V|C4)&C2) --> (V|C3|C4)&(C1|C2)
        // iff (C1&C2) == 0 and (C3&~C1) == 0 and (C4&~C2) == 0.
        ConstantInt *C3 = 0, *C4 = 0;
        if (match(A, m_Or(m_Value(V1), m_ConstantInt(C3))) &&
            (C3->getValue() & ~C1->getValue()) == 0 &&
            match(B, m_Or(m_Specific(V1), m_ConstantInt(C4))) &&
            (C4->getValue() & ~C2->getValue()) == 0) {
          V2 = Builder->CreateOr(V1, ConstantExpr::getOr(C3, C4), "bitfield");
          return BinaryOperator::CreateAnd(V2,
                               ConstantInt::get(B->getContext(),
                                                C1->getValue()|C2->getValue()));
        }
      }
    }

    // (A & (C0?-1:0)) | (B & ~(C0?-1:0)) ->  C0 ? A : B, and commuted variants.
    // Don't do this for vector select idioms, the code generator doesn't handle
    // them well yet.
    if (!I.getType()->isVectorTy()) {
      if (Instruction *Match = MatchSelectFromAndOr(A, B, C, D))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(B, A, D, C))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(C, B, A, D))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(D, A, B, C))
        return Match;
    }

    // ((A&~B)|(~A&B)) -> A^B
    if ((match(C, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, D);
    // ((~B&A)|(~A&B)) -> A^B
    if ((match(A, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, D);
    // ((A&~B)|(B&~A)) -> A^B
    if ((match(C, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, B);
    // ((~B&A)|(B&~A)) -> A^B
    if ((match(A, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, B);

    // ((A|B)&1)|(B&-2) -> (A&1) | B
    if (match(A, m_Or(m_Value(V1), m_Specific(B))) ||
        match(A, m_Or(m_Specific(B), m_Value(V1)))) {
      Instruction *Ret = FoldOrWithConstants(I, Op1, V1, B, C);
      if (Ret) return Ret;
    }
    // (B&-2)|((A|B)&1) -> (A&1) | B
    if (match(B, m_Or(m_Specific(A), m_Value(V1))) ||
        match(B, m_Or(m_Value(V1), m_Specific(A)))) {
      Instruction *Ret = FoldOrWithConstants(I, Op0, A, V1, D);
      if (Ret) return Ret;
    }
  }
  
  // (X >> Z) | (Y >> Z)  -> (X|Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Value *NewOp = Builder->CreateOr(SI0->getOperand(0), SI1->getOperand(0),
                                         SI0->getName());
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  // (~A | ~B) == (~(A & B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *And = Builder->CreateAnd(Op0NotVal, Op1NotVal,
                                        I.getName()+".demorgan");
        return BinaryOperator::CreateNot(And);
      }

  // Canonicalize xor to the RHS.
  bool SwappedForXor = false;
  if (match(Op0, m_Xor(m_Value(), m_Value()))) {
    std::swap(Op0, Op1);
    SwappedForXor = true;
  }

  // A | ( A ^ B) -> A |  B
  // A | (~A ^ B) -> A | ~B
  // (A & B) | (A ^ B)
  if (match(Op1, m_Xor(m_Value(A), m_Value(B)))) {
    if (Op0 == A || Op0 == B)
      return BinaryOperator::CreateOr(A, B);

    if (match(Op0, m_And(m_Specific(A), m_Specific(B))) ||
        match(Op0, m_And(m_Specific(B), m_Specific(A))))
      return BinaryOperator::CreateOr(A, B);

    if (Op1->hasOneUse() && match(A, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder->CreateNot(B, B->getName()+".not");
      return BinaryOperator::CreateOr(Not, Op0);
    }
    if (Op1->hasOneUse() && match(B, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder->CreateNot(A, A->getName()+".not");
      return BinaryOperator::CreateOr(Not, Op0);
    }
  }

  // A | ~(A | B) -> A | ~B
  // A | ~(A ^ B) -> A | ~B
  if (match(Op1, m_Not(m_Value(A))))
    if (BinaryOperator *B = dyn_cast<BinaryOperator>(A))
      if ((Op0 == B->getOperand(0) || Op0 == B->getOperand(1)) &&
          Op1->hasOneUse() && (B->getOpcode() == Instruction::Or ||
                               B->getOpcode() == Instruction::Xor)) {
        Value *NotOp = Op0 == B->getOperand(0) ? B->getOperand(1) :
                                                 B->getOperand(0);
        Value *Not = Builder->CreateNot(NotOp, NotOp->getName()+".not");
        return BinaryOperator::CreateOr(Not, Op0);
      }

  if (SwappedForXor)
    std::swap(Op0, Op1);

  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (Value *Res = FoldOrOfICmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);
    
  // (fcmp uno x, c) | (fcmp uno y, c)  -> (fcmp uno x, y)
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldOrOfFCmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);
  
  // fold (or (cast A), (cast B)) -> (cast (or A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    CastInst *Op1C = dyn_cast<CastInst>(Op1);
    if (Op1C && Op0C->getOpcode() == Op1C->getOpcode()) {// same cast kind ?
      Type *SrcTy = Op0C->getOperand(0)->getType();
      if (SrcTy == Op1C->getOperand(0)->getType() &&
          SrcTy->isIntOrIntVectorTy()) {
        Value *Op0COp = Op0C->getOperand(0), *Op1COp = Op1C->getOperand(0);

        if ((!isa<ICmpInst>(Op0COp) || !isa<ICmpInst>(Op1COp)) &&
            // Only do this if the casts both really cause code to be
            // generated.
            ShouldOptimizeCast(Op0C->getOpcode(), Op0COp, I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1COp, I.getType())) {
          Value *NewOp = Builder->CreateOr(Op0COp, Op1COp, I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
        
        // If this is or(cast(icmp), cast(icmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1COp))
          if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0COp))
            if (Value *Res = FoldOrOfICmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
        
        // If this is or(cast(fcmp), cast(fcmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (FCmpInst *RHS = dyn_cast<FCmpInst>(Op1COp))
          if (FCmpInst *LHS = dyn_cast<FCmpInst>(Op0COp))
            if (Value *Res = FoldOrOfFCmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
      }
    }
  }

  // or(sext(A), B) -> A ? -1 : B where A is an i1
  // or(A, sext(B)) -> B ? -1 : A where B is an i1
  if (match(Op0, m_SExt(m_Value(A))) && A->getType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op1);
  if (match(Op1, m_SExt(m_Value(A))) && A->getType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op0);

  // Note: If we've gotten to the point of visiting the outer OR, then the
  // inner one couldn't be simplified.  If it was a constant, then it won't
  // be simplified by a later pass either, so we try swapping the inner/outer
  // ORs in the hopes that we'll be able to simplify it this way.
  // (X|C) | V --> (X|V) | C
  if (Op0->hasOneUse() && !isa<ConstantInt>(Op1) &&
      match(Op0, m_Or(m_Value(A), m_ConstantInt(C1)))) {
    Value *Inner = Builder->CreateOr(A, Op1);
    Inner->takeName(Op0);
    return BinaryOperator::CreateOr(Inner, C1);
  }
  
  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyXorInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // (A&B)^(A&C) -> A&(B^C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Is this a ~ operation?
  if (Value *NotOp = dyn_castNotVal(&I)) {
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(NotOp)) {
      if (Op0I->getOpcode() == Instruction::And || 
          Op0I->getOpcode() == Instruction::Or) {
        // ~(~X & Y) --> (X | ~Y) - De Morgan's Law
        // ~(~X | Y) === (X & ~Y) - De Morgan's Law
        if (dyn_castNotVal(Op0I->getOperand(1)))
          Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1),
                               Op0I->getOperand(1)->getName()+".not");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(Op0NotVal, NotY);
          return BinaryOperator::CreateAnd(Op0NotVal, NotY);
        }
        
        // ~(X & Y) --> (~X | ~Y) - De Morgan's Law
        // ~(X | Y) === (~X & ~Y) - De Morgan's Law
        if (isFreeToInvert(Op0I->getOperand(0)) && 
            isFreeToInvert(Op0I->getOperand(1))) {
          Value *NotX =
            Builder->CreateNot(Op0I->getOperand(0), "notlhs");
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1), "notrhs");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(NotX, NotY);
          return BinaryOperator::CreateAnd(NotX, NotY);
        }

      } else if (Op0I->getOpcode() == Instruction::AShr) {
        // ~(~X >>s Y) --> (X >>s Y)
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0)))
          return BinaryOperator::CreateAShr(Op0NotVal, Op0I->getOperand(1));
      }
    }
  }
  
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    if (RHS->isOne() && Op0->hasOneUse())
      // xor (cmp A, B), true = not (cmp A, B) = !cmp A, B
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0))
        return CmpInst::Create(CI->getOpcode(),
                               CI->getInversePredicate(),
                               CI->getOperand(0), CI->getOperand(1));

    // fold (xor(zext(cmp)), 1) and (xor(sext(cmp)), -1) to ext(!cmp).
    if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0C->getOperand(0))) {
        if (CI->hasOneUse() && Op0C->hasOneUse()) {
          Instruction::CastOps Opcode = Op0C->getOpcode();
          if ((Opcode == Instruction::ZExt || Opcode == Instruction::SExt) &&
              (RHS == ConstantExpr::getCast(Opcode, 
                                           ConstantInt::getTrue(I.getContext()),
                                            Op0C->getDestTy()))) {
            CI->setPredicate(CI->getInversePredicate());
            return CastInst::Create(Opcode, CI, Op0C->getType());
          }
        }
      }
    }

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                      ConstantInt::get(I.getType(), 1));
          return BinaryOperator::CreateAdd(Op0I->getOperand(1), ConstantRHS);
        }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::CreateSub(
                           ConstantExpr::getSub(NegOp0CI,
                                      ConstantInt::get(I.getType(), 1)),
                                      Op0I->getOperand(0));
          } else if (RHS->getValue().isSignBit()) {
            // (X + C) ^ signbit -> (X + C + signbit)
            Constant *C = ConstantInt::get(I.getContext(),
                                           RHS->getValue() + Op0CI->getValue());
            return BinaryOperator::CreateAdd(Op0I->getOperand(0), C);

          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getValue())) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHS);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = ConstantExpr::getAnd(Op0CI, RHS);
            NewRHS = ConstantExpr::getAnd(NewRHS, 
                                       ConstantExpr::getNot(CommonBits));
            Worklist.Add(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
        }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1);
  if (Op1I) {
    Value *A, *B;
    if (match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op0) {              // B^(B|A) == (A|B)^B
        Op1I->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (B == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    } else if (match(Op1I, m_And(m_Value(A), m_Value(B))) && 
               Op1I->hasOneUse()){
      if (A == Op0) {                                      // A^(A&B) -> A^(B&A)
        Op1I->swapOperands();
        std::swap(A, B);
      }
      if (B == Op0) {                                      // A^(B&A) -> (B&A)^A
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    }
  }
  
  BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0);
  if (Op0I) {
    Value *A, *B;
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        Op0I->hasOneUse()) {
      if (A == Op1)                                  // (B|A)^B == (A|B)^B
        std::swap(A, B);
      if (B == Op1)                                  // (A|B)^B == A & ~B
        return BinaryOperator::CreateAnd(A, Builder->CreateNot(Op1));
    } else if (match(Op0I, m_And(m_Value(A), m_Value(B))) && 
               Op0I->hasOneUse()){
      if (A == Op1)                                        // (A&B)^A -> (B&A)^A
        std::swap(A, B);
      if (B == Op1 &&                                      // (B&A)^A == ~B & A
          !isa<ConstantInt>(Op1)) {  // Canonical form is (B&C)^C
        return BinaryOperator::CreateAnd(Builder->CreateNot(A), Op1);
      }
    }
  }
  
  // (X >> Z) ^ (Y >> Z)  -> (X^Y) >> Z  for all shifts.
  if (Op0I && Op1I && Op0I->isShift() && 
      Op0I->getOpcode() == Op1I->getOpcode() && 
      Op0I->getOperand(1) == Op1I->getOperand(1) &&
      (Op0I->hasOneUse() || Op1I->hasOneUse())) {
    Value *NewOp =
      Builder->CreateXor(Op0I->getOperand(0), Op1I->getOperand(0),
                         Op0I->getName());
    return BinaryOperator::Create(Op1I->getOpcode(), NewOp, 
                                  Op1I->getOperand(1));
  }
    
  if (Op0I && Op1I) {
    Value *A, *B, *C, *D;
    // (A & B)^(A | B) -> A ^ B
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Or(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
    // (A | B)^(A & B) -> A ^ B
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
  }

  // (icmp1 A, B) ^ (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (PredicatesFoldable(LHS->getPredicate(), RHS->getPredicate())) {
        if (LHS->getOperand(0) == RHS->getOperand(1) &&
            LHS->getOperand(1) == RHS->getOperand(0))
          LHS->swapOperands();
        if (LHS->getOperand(0) == RHS->getOperand(0) &&
            LHS->getOperand(1) == RHS->getOperand(1)) {
          Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
          unsigned Code = getICmpCode(LHS) ^ getICmpCode(RHS);
          bool isSigned = LHS->isSigned() || RHS->isSigned();
          return ReplaceInstUsesWith(I, 
                               getNewICmpValue(isSigned, Code, Op0, Op1,
                                               Builder));
        }
      }

  // fold (xor (cast A), (cast B)) -> (cast (xor A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind?
        Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isIntegerTy() &&
            // Only do this if the casts both really cause code to be generated.
            ShouldOptimizeCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                               I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                               I.getType())) {
          Value *NewOp = Builder->CreateXor(Op0C->getOperand(0),
                                            Op1C->getOperand(0), I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
  }

  return Changed ? &I : 0;
}
