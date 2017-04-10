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

#include "InstCombineInternal.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/CmpInstAnalysis.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

static inline Value *dyn_castNotVal(Value *V) {
  // If this is not(not(x)) don't return that this is a not: we want the two
  // not's to be folded first.
  if (BinaryOperator::isNot(V)) {
    Value *Operand = BinaryOperator::getNotArgument(V);
    if (!IsFreeToInvert(Operand, Operand->hasOneUse()))
      return Operand;
  }

  // Constants can be considered to be not'ed values...
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantInt::get(C->getType(), ~C->getValue());
  return nullptr;
}

/// Similar to getICmpCode but for FCmpInst. This encodes a fcmp predicate into
/// a four bit mask.
static unsigned getFCmpCode(FCmpInst::Predicate CC) {
  assert(FCmpInst::FCMP_FALSE <= CC && CC <= FCmpInst::FCMP_TRUE &&
         "Unexpected FCmp predicate!");
  // Take advantage of the bit pattern of FCmpInst::Predicate here.
  //                                                 U L G E
  static_assert(FCmpInst::FCMP_FALSE ==  0, "");  // 0 0 0 0
  static_assert(FCmpInst::FCMP_OEQ   ==  1, "");  // 0 0 0 1
  static_assert(FCmpInst::FCMP_OGT   ==  2, "");  // 0 0 1 0
  static_assert(FCmpInst::FCMP_OGE   ==  3, "");  // 0 0 1 1
  static_assert(FCmpInst::FCMP_OLT   ==  4, "");  // 0 1 0 0
  static_assert(FCmpInst::FCMP_OLE   ==  5, "");  // 0 1 0 1
  static_assert(FCmpInst::FCMP_ONE   ==  6, "");  // 0 1 1 0
  static_assert(FCmpInst::FCMP_ORD   ==  7, "");  // 0 1 1 1
  static_assert(FCmpInst::FCMP_UNO   ==  8, "");  // 1 0 0 0
  static_assert(FCmpInst::FCMP_UEQ   ==  9, "");  // 1 0 0 1
  static_assert(FCmpInst::FCMP_UGT   == 10, "");  // 1 0 1 0
  static_assert(FCmpInst::FCMP_UGE   == 11, "");  // 1 0 1 1
  static_assert(FCmpInst::FCMP_ULT   == 12, "");  // 1 1 0 0
  static_assert(FCmpInst::FCMP_ULE   == 13, "");  // 1 1 0 1
  static_assert(FCmpInst::FCMP_UNE   == 14, "");  // 1 1 1 0
  static_assert(FCmpInst::FCMP_TRUE  == 15, "");  // 1 1 1 1
  return CC;
}

/// This is the complement of getICmpCode, which turns an opcode and two
/// operands into either a constant true or false, or a brand new ICmp
/// instruction. The sign is passed in to determine which kind of predicate to
/// use in the new icmp instruction.
static Value *getNewICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                              InstCombiner::BuilderTy *Builder) {
  ICmpInst::Predicate NewPred;
  if (Value *NewConstant = getICmpValue(Sign, Code, LHS, RHS, NewPred))
    return NewConstant;
  return Builder->CreateICmp(NewPred, LHS, RHS);
}

/// This is the complement of getFCmpCode, which turns an opcode and two
/// operands into either a FCmp instruction, or a true/false constant.
static Value *getFCmpValue(unsigned Code, Value *LHS, Value *RHS,
                           InstCombiner::BuilderTy *Builder) {
  const auto Pred = static_cast<FCmpInst::Predicate>(Code);
  assert(FCmpInst::FCMP_FALSE <= Pred && Pred <= FCmpInst::FCMP_TRUE &&
         "Unexpected FCmp predicate!");
  if (Pred == FCmpInst::FCMP_FALSE)
    return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
  if (Pred == FCmpInst::FCMP_TRUE)
    return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
  return Builder->CreateFCmp(Pred, LHS, RHS);
}

/// \brief Transform BITWISE_OP(BSWAP(A),BSWAP(B)) to BSWAP(BITWISE_OP(A, B))
/// \param I Binary operator to transform.
/// \return Pointer to node that must replace the original binary operator, or
///         null pointer if no transformation was made.
Value *InstCombiner::SimplifyBSwap(BinaryOperator &I) {
  IntegerType *ITy = dyn_cast<IntegerType>(I.getType());

  // Can't do vectors.
  if (I.getType()->isVectorTy())
    return nullptr;

  // Can only do bitwise ops.
  if (!I.isBitwiseLogicOp())
    return nullptr;

  Value *OldLHS = I.getOperand(0);
  Value *OldRHS = I.getOperand(1);
  ConstantInt *ConstLHS = dyn_cast<ConstantInt>(OldLHS);
  ConstantInt *ConstRHS = dyn_cast<ConstantInt>(OldRHS);
  IntrinsicInst *IntrLHS = dyn_cast<IntrinsicInst>(OldLHS);
  IntrinsicInst *IntrRHS = dyn_cast<IntrinsicInst>(OldRHS);
  bool IsBswapLHS = (IntrLHS && IntrLHS->getIntrinsicID() == Intrinsic::bswap);
  bool IsBswapRHS = (IntrRHS && IntrRHS->getIntrinsicID() == Intrinsic::bswap);

  if (!IsBswapLHS && !IsBswapRHS)
    return nullptr;

  if (!IsBswapLHS && !ConstLHS)
    return nullptr;

  if (!IsBswapRHS && !ConstRHS)
    return nullptr;

  /// OP( BSWAP(x), BSWAP(y) ) -> BSWAP( OP(x, y) )
  /// OP( BSWAP(x), CONSTANT ) -> BSWAP( OP(x, BSWAP(CONSTANT) ) )
  Value *NewLHS = IsBswapLHS ? IntrLHS->getOperand(0) :
                  Builder->getInt(ConstLHS->getValue().byteSwap());

  Value *NewRHS = IsBswapRHS ? IntrRHS->getOperand(0) :
                  Builder->getInt(ConstRHS->getValue().byteSwap());

  Value *BinOp = Builder->CreateBinOp(I.getOpcode(), NewLHS, NewRHS);
  Function *F = Intrinsic::getDeclaration(I.getModule(), Intrinsic::bswap, ITy);
  return Builder->CreateCall(F, BinOp);
}

/// This handles expressions of the form ((val OP C1) & C2).  Where
/// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.
Instruction *InstCombiner::OptAndOp(BinaryOperator *Op,
                                    ConstantInt *OpRHS,
                                    ConstantInt *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = nullptr;
  if (!Op->isShift())
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  default: break;
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
      const APInt &AndRHSV = AndRHS->getValue();

      // If there is only one bit set.
      if (AndRHSV.isPowerOf2()) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        const APInt& AddRHS = OpRHS->getValue();

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
    ConstantInt *CI = Builder->getInt(AndRHS->getValue() & ShlMask);

    if (CI->getValue() == ShlMask)
      // Masking out bits that the shift already masks.
      return replaceInstUsesWith(TheAnd, Op);   // No need for the and.

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
    ConstantInt *CI = Builder->getInt(AndRHS->getValue() & ShrMask);

    if (CI->getValue() == ShrMask)
      // Masking out bits that the shift already masks.
      return replaceInstUsesWith(TheAnd, Op);

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
      Constant *C = Builder->getInt(AndRHS->getValue() & ShrMask);
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
  return nullptr;
}

/// Emit a computation of: (V >= Lo && V < Hi) if Inside is true, otherwise
/// (V < Lo || V >= Hi). This method expects that Lo <= Hi. IsSigned indicates
/// whether to treat V, Lo, and Hi as signed or not.
Value *InstCombiner::insertRangeTest(Value *V, const APInt &Lo, const APInt &Hi,
                                     bool isSigned, bool Inside) {
  assert((isSigned ? Lo.sle(Hi) : Lo.ule(Hi)) &&
         "Lo is not <= Hi in range emission code!");

  Type *Ty = V->getType();
  if (Lo == Hi)
    return Inside ? ConstantInt::getFalse(Ty) : ConstantInt::getTrue(Ty);

  // V >= Min && V <  Hi --> V <  Hi
  // V <  Min || V >= Hi --> V >= Hi
  ICmpInst::Predicate Pred = Inside ? ICmpInst::ICMP_ULT : ICmpInst::ICMP_UGE;
  if (isSigned ? Lo.isMinSignedValue() : Lo.isMinValue()) {
    Pred = isSigned ? ICmpInst::getSignedPredicate(Pred) : Pred;
    return Builder->CreateICmp(Pred, V, ConstantInt::get(Ty, Hi));
  }

  // V >= Lo && V <  Hi --> V - Lo u<  Hi - Lo
  // V <  Lo || V >= Hi --> V - Lo u>= Hi - Lo
  Value *VMinusLo =
      Builder->CreateSub(V, ConstantInt::get(Ty, Lo), V->getName() + ".off");
  Constant *HiMinusLo = ConstantInt::get(Ty, Hi - Lo);
  return Builder->CreateICmp(Pred, VMinusLo, HiMinusLo);
}

/// Classify (icmp eq (A & B), C) and (icmp ne (A & B), C) as matching patterns
/// that can be simplified.
/// One of A and B is considered the mask. The other is the value. This is
/// described as the "AMask" or "BMask" part of the enum. If the enum contains
/// only "Mask", then both A and B can be considered masks. If A is the mask,
/// then it was proven that (A & C) == C. This is trivial if C == A or C == 0.
/// If both A and C are constants, this proof is also easy.
/// For the following explanations, we assume that A is the mask.
///
/// "AllOnes" declares that the comparison is true only if (A & B) == A or all
/// bits of A are set in B.
///   Example: (icmp eq (A & 3), 3) -> AMask_AllOnes
///
/// "AllZeros" declares that the comparison is true only if (A & B) == 0 or all
/// bits of A are cleared in B.
///   Example: (icmp eq (A & 3), 0) -> Mask_AllZeroes
///
/// "Mixed" declares that (A & B) == C and C might or might not contain any
/// number of one bits and zero bits.
///   Example: (icmp eq (A & 3), 1) -> AMask_Mixed
///
/// "Not" means that in above descriptions "==" should be replaced by "!=".
///   Example: (icmp ne (A & 3), 3) -> AMask_NotAllOnes
///
/// If the mask A contains a single bit, then the following is equivalent:
///    (icmp eq (A & B), A) equals (icmp ne (A & B), 0)
///    (icmp ne (A & B), A) equals (icmp eq (A & B), 0)
enum MaskedICmpType {
  AMask_AllOnes           =     1,
  AMask_NotAllOnes        =     2,
  BMask_AllOnes           =     4,
  BMask_NotAllOnes        =     8,
  Mask_AllZeros           =    16,
  Mask_NotAllZeros        =    32,
  AMask_Mixed             =    64,
  AMask_NotMixed          =   128,
  BMask_Mixed             =   256,
  BMask_NotMixed          =   512
};

/// Return the set of patterns (from MaskedICmpType) that (icmp SCC (A & B), C)
/// satisfies.
static unsigned getMaskedICmpType(Value *A, Value *B, Value *C,
                                  ICmpInst::Predicate Pred) {
  ConstantInt *ACst = dyn_cast<ConstantInt>(A);
  ConstantInt *BCst = dyn_cast<ConstantInt>(B);
  ConstantInt *CCst = dyn_cast<ConstantInt>(C);
  bool IsEq = (Pred == ICmpInst::ICMP_EQ);
  bool IsAPow2 = (ACst && !ACst->isZero() && ACst->getValue().isPowerOf2());
  bool IsBPow2 = (BCst && !BCst->isZero() && BCst->getValue().isPowerOf2());
  unsigned MaskVal = 0;
  if (CCst && CCst->isZero()) {
    // if C is zero, then both A and B qualify as mask
    MaskVal |= (IsEq ? (Mask_AllZeros | AMask_Mixed | BMask_Mixed)
                     : (Mask_NotAllZeros | AMask_NotMixed | BMask_NotMixed));
    if (IsAPow2)
      MaskVal |= (IsEq ? (AMask_NotAllOnes | AMask_NotMixed)
                       : (AMask_AllOnes | AMask_Mixed));
    if (IsBPow2)
      MaskVal |= (IsEq ? (BMask_NotAllOnes | BMask_NotMixed)
                       : (BMask_AllOnes | BMask_Mixed));
    return MaskVal;
  }

  if (A == C) {
    MaskVal |= (IsEq ? (AMask_AllOnes | AMask_Mixed)
                     : (AMask_NotAllOnes | AMask_NotMixed));
    if (IsAPow2)
      MaskVal |= (IsEq ? (Mask_NotAllZeros | AMask_NotMixed)
                       : (Mask_AllZeros | AMask_Mixed));
  } else if (ACst && CCst && ConstantExpr::getAnd(ACst, CCst) == CCst) {
    MaskVal |= (IsEq ? AMask_Mixed : AMask_NotMixed);
  }

  if (B == C) {
    MaskVal |= (IsEq ? (BMask_AllOnes | BMask_Mixed)
                     : (BMask_NotAllOnes | BMask_NotMixed));
    if (IsBPow2)
      MaskVal |= (IsEq ? (Mask_NotAllZeros | BMask_NotMixed)
                       : (Mask_AllZeros | BMask_Mixed));
  } else if (BCst && CCst && ConstantExpr::getAnd(BCst, CCst) == CCst) {
    MaskVal |= (IsEq ? BMask_Mixed : BMask_NotMixed);
  }

  return MaskVal;
}

/// Convert an analysis of a masked ICmp into its equivalent if all boolean
/// operations had the opposite sense. Since each "NotXXX" flag (recording !=)
/// is adjacent to the corresponding normal flag (recording ==), this just
/// involves swapping those bits over.
static unsigned conjugateICmpMask(unsigned Mask) {
  unsigned NewMask;
  NewMask = (Mask & (AMask_AllOnes | BMask_AllOnes | Mask_AllZeros |
                     AMask_Mixed | BMask_Mixed))
            << 1;

  NewMask |= (Mask & (AMask_NotAllOnes | BMask_NotAllOnes | Mask_NotAllZeros |
                      AMask_NotMixed | BMask_NotMixed))
             >> 1;

  return NewMask;
}

/// Handle (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E).
/// Return the set of pattern classes (from MaskedICmpType) that both LHS and
/// RHS satisfy.
static unsigned getMaskedTypeForICmpPair(Value *&A, Value *&B, Value *&C,
                                         Value *&D, Value *&E, ICmpInst *LHS,
                                         ICmpInst *RHS,
                                         ICmpInst::Predicate &PredL,
                                         ICmpInst::Predicate &PredR) {
  if (LHS->getOperand(0)->getType() != RHS->getOperand(0)->getType())
    return 0;
  // vectors are not (yet?) supported
  if (LHS->getOperand(0)->getType()->isVectorTy())
    return 0;

  // Here comes the tricky part:
  // LHS might be of the form L11 & L12 == X, X == L21 & L22,
  // and L11 & L12 == L21 & L22. The same goes for RHS.
  // Now we must find those components L** and R**, that are equal, so
  // that we can extract the parameters A, B, C, D, and E for the canonical
  // above.
  Value *L1 = LHS->getOperand(0);
  Value *L2 = LHS->getOperand(1);
  Value *L11, *L12, *L21, *L22;
  // Check whether the icmp can be decomposed into a bit test.
  if (decomposeBitTestICmp(LHS, PredL, L11, L12, L2)) {
    L21 = L22 = L1 = nullptr;
  } else {
    // Look for ANDs in the LHS icmp.
    if (!L1->getType()->isIntegerTy()) {
      // You can icmp pointers, for example. They really aren't masks.
      L11 = L12 = nullptr;
    } else if (!match(L1, m_And(m_Value(L11), m_Value(L12)))) {
      // Any icmp can be viewed as being trivially masked; if it allows us to
      // remove one, it's worth it.
      L11 = L1;
      L12 = Constant::getAllOnesValue(L1->getType());
    }

    if (!L2->getType()->isIntegerTy()) {
      // You can icmp pointers, for example. They really aren't masks.
      L21 = L22 = nullptr;
    } else if (!match(L2, m_And(m_Value(L21), m_Value(L22)))) {
      L21 = L2;
      L22 = Constant::getAllOnesValue(L2->getType());
    }
  }

  // Bail if LHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(PredL))
    return 0;

  Value *R1 = RHS->getOperand(0);
  Value *R2 = RHS->getOperand(1);
  Value *R11, *R12;
  bool Ok = false;
  if (decomposeBitTestICmp(RHS, PredR, R11, R12, R2)) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11;
      D = R12;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12;
      D = R11;
    } else {
      return 0;
    }
    E = R2;
    R1 = nullptr;
    Ok = true;
  } else if (R1->getType()->isIntegerTy()) {
    if (!match(R1, m_And(m_Value(R11), m_Value(R12)))) {
      // As before, model no mask as a trivial mask if it'll let us do an
      // optimization.
      R11 = R1;
      R12 = Constant::getAllOnesValue(R1->getType());
    }

    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11;
      D = R12;
      E = R2;
      Ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12;
      D = R11;
      E = R2;
      Ok = true;
    }
  }

  // Bail if RHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(PredR))
    return 0;

  // Look for ANDs on the right side of the RHS icmp.
  if (!Ok && R2->getType()->isIntegerTy()) {
    if (!match(R2, m_And(m_Value(R11), m_Value(R12)))) {
      R11 = R2;
      R12 = Constant::getAllOnesValue(R2->getType());
    }

    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11;
      D = R12;
      E = R1;
      Ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12;
      D = R11;
      E = R1;
      Ok = true;
    } else {
      return 0;
    }
  }
  if (!Ok)
    return 0;

  if (L11 == A) {
    B = L12;
    C = L2;
  } else if (L12 == A) {
    B = L11;
    C = L2;
  } else if (L21 == A) {
    B = L22;
    C = L1;
  } else if (L22 == A) {
    B = L21;
    C = L1;
  }

  unsigned LeftType = getMaskedICmpType(A, B, C, PredL);
  unsigned RightType = getMaskedICmpType(A, D, E, PredR);
  return LeftType & RightType;
}

/// Try to fold (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// into a single (icmp(A & X) ==/!= Y).
static Value *foldLogOpOfMaskedICmps(ICmpInst *LHS, ICmpInst *RHS, bool IsAnd,
                                     llvm::InstCombiner::BuilderTy *Builder) {
  Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr, *E = nullptr;
  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();
  unsigned Mask =
      getMaskedTypeForICmpPair(A, B, C, D, E, LHS, RHS, PredL, PredR);
  if (Mask == 0)
    return nullptr;

  assert(ICmpInst::isEquality(PredL) && ICmpInst::isEquality(PredR) &&
         "Expected equality predicates for masked type of icmps.");

  // In full generality:
  //     (icmp (A & B) Op C) | (icmp (A & D) Op E)
  // ==  ![ (icmp (A & B) !Op C) & (icmp (A & D) !Op E) ]
  //
  // If the latter can be converted into (icmp (A & X) Op Y) then the former is
  // equivalent to (icmp (A & X) !Op Y).
  //
  // Therefore, we can pretend for the rest of this function that we're dealing
  // with the conjunction, provided we flip the sense of any comparisons (both
  // input and output).

  // In most cases we're going to produce an EQ for the "&&" case.
  ICmpInst::Predicate NewCC = IsAnd ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE;
  if (!IsAnd) {
    // Convert the masking analysis into its equivalent with negated
    // comparisons.
    Mask = conjugateICmpMask(Mask);
  }

  if (Mask & Mask_AllZeros) {
    // (icmp eq (A & B), 0) & (icmp eq (A & D), 0)
    // -> (icmp eq (A & (B|D)), 0)
    Value *NewOr = Builder->CreateOr(B, D);
    Value *NewAnd = Builder->CreateAnd(A, NewOr);
    // We can't use C as zero because we might actually handle
    //   (icmp ne (A & B), B) & (icmp ne (A & D), D)
    // with B and D, having a single bit set.
    Value *Zero = Constant::getNullValue(A->getType());
    return Builder->CreateICmp(NewCC, NewAnd, Zero);
  }
  if (Mask & BMask_AllOnes) {
    // (icmp eq (A & B), B) & (icmp eq (A & D), D)
    // -> (icmp eq (A & (B|D)), (B|D))
    Value *NewOr = Builder->CreateOr(B, D);
    Value *NewAnd = Builder->CreateAnd(A, NewOr);
    return Builder->CreateICmp(NewCC, NewAnd, NewOr);
  }
  if (Mask & AMask_AllOnes) {
    // (icmp eq (A & B), A) & (icmp eq (A & D), A)
    // -> (icmp eq (A & (B&D)), A)
    Value *NewAnd1 = Builder->CreateAnd(B, D);
    Value *NewAnd2 = Builder->CreateAnd(A, NewAnd1);
    return Builder->CreateICmp(NewCC, NewAnd2, A);
  }

  // Remaining cases assume at least that B and D are constant, and depend on
  // their actual values. This isn't strictly necessary, just a "handle the
  // easy cases for now" decision.
  ConstantInt *BCst = dyn_cast<ConstantInt>(B);
  if (!BCst)
    return nullptr;
  ConstantInt *DCst = dyn_cast<ConstantInt>(D);
  if (!DCst)
    return nullptr;

  if (Mask & (Mask_NotAllZeros | BMask_NotAllOnes)) {
    // (icmp ne (A & B), 0) & (icmp ne (A & D), 0) and
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), 0) or (icmp ne (A & D), 0)
    // Only valid if one of the masks is a superset of the other (check "B&D" is
    // the same as either B or D).
    APInt NewMask = BCst->getValue() & DCst->getValue();

    if (NewMask == BCst->getValue())
      return LHS;
    else if (NewMask == DCst->getValue())
      return RHS;
  }

  if (Mask & AMask_NotAllOnes) {
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), A) or (icmp ne (A & D), A)
    // Only valid if one of the masks is a superset of the other (check "B|D" is
    // the same as either B or D).
    APInt NewMask = BCst->getValue() | DCst->getValue();

    if (NewMask == BCst->getValue())
      return LHS;
    else if (NewMask == DCst->getValue())
      return RHS;
  }

  if (Mask & BMask_Mixed) {
    // (icmp eq (A & B), C) & (icmp eq (A & D), E)
    // We already know that B & C == C && D & E == E.
    // If we can prove that (B & D) & (C ^ E) == 0, that is, the bits of
    // C and E, which are shared by both the mask B and the mask D, don't
    // contradict, then we can transform to
    // -> (icmp eq (A & (B|D)), (C|E))
    // Currently, we only handle the case of B, C, D, and E being constant.
    // We can't simply use C and E because we might actually handle
    //   (icmp ne (A & B), B) & (icmp eq (A & D), D)
    // with B and D, having a single bit set.
    ConstantInt *CCst = dyn_cast<ConstantInt>(C);
    if (!CCst)
      return nullptr;
    ConstantInt *ECst = dyn_cast<ConstantInt>(E);
    if (!ECst)
      return nullptr;
    if (PredL != NewCC)
      CCst = cast<ConstantInt>(ConstantExpr::getXor(BCst, CCst));
    if (PredR != NewCC)
      ECst = cast<ConstantInt>(ConstantExpr::getXor(DCst, ECst));

    // If there is a conflict, we should actually return a false for the
    // whole construct.
    if (((BCst->getValue() & DCst->getValue()) &
         (CCst->getValue() ^ ECst->getValue())) != 0)
      return ConstantInt::get(LHS->getType(), !IsAnd);

    Value *NewOr1 = Builder->CreateOr(B, D);
    Value *NewOr2 = ConstantExpr::getOr(CCst, ECst);
    Value *NewAnd = Builder->CreateAnd(A, NewOr1);
    return Builder->CreateICmp(NewCC, NewAnd, NewOr2);
  }

  return nullptr;
}

/// Try to fold a signed range checked with lower bound 0 to an unsigned icmp.
/// Example: (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
/// If \p Inverted is true then the check is for the inverted range, e.g.
/// (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
Value *InstCombiner::simplifyRangeCheck(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                        bool Inverted) {
  // Check the lower range comparison, e.g. x >= 0
  // InstCombine already ensured that if there is a constant it's on the RHS.
  ConstantInt *RangeStart = dyn_cast<ConstantInt>(Cmp0->getOperand(1));
  if (!RangeStart)
    return nullptr;

  ICmpInst::Predicate Pred0 = (Inverted ? Cmp0->getInversePredicate() :
                               Cmp0->getPredicate());

  // Accept x > -1 or x >= 0 (after potentially inverting the predicate).
  if (!((Pred0 == ICmpInst::ICMP_SGT && RangeStart->isMinusOne()) ||
        (Pred0 == ICmpInst::ICMP_SGE && RangeStart->isZero())))
    return nullptr;

  ICmpInst::Predicate Pred1 = (Inverted ? Cmp1->getInversePredicate() :
                               Cmp1->getPredicate());

  Value *Input = Cmp0->getOperand(0);
  Value *RangeEnd;
  if (Cmp1->getOperand(0) == Input) {
    // For the upper range compare we have: icmp x, n
    RangeEnd = Cmp1->getOperand(1);
  } else if (Cmp1->getOperand(1) == Input) {
    // For the upper range compare we have: icmp n, x
    RangeEnd = Cmp1->getOperand(0);
    Pred1 = ICmpInst::getSwappedPredicate(Pred1);
  } else {
    return nullptr;
  }

  // Check the upper range comparison, e.g. x < n
  ICmpInst::Predicate NewPred;
  switch (Pred1) {
    case ICmpInst::ICMP_SLT: NewPred = ICmpInst::ICMP_ULT; break;
    case ICmpInst::ICMP_SLE: NewPred = ICmpInst::ICMP_ULE; break;
    default: return nullptr;
  }

  // This simplification is only valid if the upper range is not negative.
  bool IsNegative, IsNotNegative;
  ComputeSignBit(RangeEnd, IsNotNegative, IsNegative, /*Depth=*/0, Cmp1);
  if (!IsNotNegative)
    return nullptr;

  if (Inverted)
    NewPred = ICmpInst::getInversePredicate(NewPred);

  return Builder->CreateICmp(NewPred, Input, RangeEnd);
}

/// Fold (icmp)&(icmp) if possible.
Value *InstCombiner::FoldAndOfICmps(ICmpInst *LHS, ICmpInst *RHS) {
  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();

  // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(PredL, PredR)) {
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
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, true, Builder))
    return V;

  // E.g. (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
  if (Value *V = simplifyRangeCheck(LHS, RHS, /*Inverted=*/false))
    return V;

  // E.g. (icmp slt x, n) & (icmp sge x, 0) --> icmp ult x, n
  if (Value *V = simplifyRangeCheck(RHS, LHS, /*Inverted=*/false))
    return V;

  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSC = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (!LHSC || !RHSC)
    return nullptr;

  if (LHSC == RHSC && PredL == PredR) {
    // (icmp ult A, C) & (icmp ult B, C) --> (icmp ult (A|B), C)
    // where C is a power of 2 or
    // (icmp eq A, 0) & (icmp eq B, 0) --> (icmp eq (A|B), 0)
    if ((PredL == ICmpInst::ICMP_ULT && LHSC->getValue().isPowerOf2()) ||
        (PredL == ICmpInst::ICMP_EQ && LHSC->isZero())) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(PredL, NewOr, LHSC);
    }
  }

  // (trunc x) == C1 & (and x, CA) == C2 -> (and x, CA|CMAX) == C1|C2
  // where CMAX is the all ones value for the truncated type,
  // iff the lower bits of C2 and CA are zero.
  if (PredL == ICmpInst::ICMP_EQ && PredL == PredR && LHS->hasOneUse() &&
      RHS->hasOneUse()) {
    Value *V;
    ConstantInt *AndC, *SmallC = nullptr, *BigC = nullptr;

    // (trunc x) == C1 & (and x, CA) == C2
    // (and x, CA) == C2 & (trunc x) == C1
    if (match(Val2, m_Trunc(m_Value(V))) &&
        match(Val, m_And(m_Specific(V), m_ConstantInt(AndC)))) {
      SmallC = RHSC;
      BigC = LHSC;
    } else if (match(Val, m_Trunc(m_Value(V))) &&
               match(Val2, m_And(m_Specific(V), m_ConstantInt(AndC)))) {
      SmallC = LHSC;
      BigC = RHSC;
    }

    if (SmallC && BigC) {
      unsigned BigBitSize = BigC->getType()->getBitWidth();
      unsigned SmallBitSize = SmallC->getType()->getBitWidth();

      // Check that the low bits are zero.
      APInt Low = APInt::getLowBitsSet(BigBitSize, SmallBitSize);
      if ((Low & AndC->getValue()) == 0 && (Low & BigC->getValue()) == 0) {
        Value *NewAnd = Builder->CreateAnd(V, Low | AndC->getValue());
        APInt N = SmallC->getValue().zext(BigBitSize) | BigC->getValue();
        Value *NewVal = ConstantInt::get(AndC->getType()->getContext(), N);
        return Builder->CreateICmp(PredL, NewAnd, NewVal);
      }
    }
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) & (icmp2 A, C2) --> something simpler.
  if (Val != Val2)
    return nullptr;

  // ICMP_[US][GL]E X, C is folded to ICMP_[US][GL]T elsewhere.
  if (PredL == ICmpInst::ICMP_UGE || PredL == ICmpInst::ICMP_ULE ||
      PredR == ICmpInst::ICMP_UGE || PredR == ICmpInst::ICMP_ULE ||
      PredL == ICmpInst::ICMP_SGE || PredL == ICmpInst::ICMP_SLE ||
      PredR == ICmpInst::ICMP_SGE || PredR == ICmpInst::ICMP_SLE)
    return nullptr;

  // We can't fold (ugt x, C) & (sgt x, C2).
  if (!PredicatesFoldable(PredL, PredR))
    return nullptr;

  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(PredL) ||
      (ICmpInst::isEquality(PredL) && CmpInst::isSigned(PredR)))
    ShouldSwap = LHSC->getValue().sgt(RHSC->getValue());
  else
    ShouldSwap = LHSC->getValue().ugt(RHSC->getValue());

  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSC, RHSC);
    std::swap(PredL, PredR);
  }

  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and and'ing the result
  // together.  Because of the above check, we know that we only have
  // icmp eq, icmp ne, icmp [su]lt, and icmp [SU]gt here. We also know
  // (from the icmp folding check above), that the two constants
  // are not equal and that the larger constant is on the RHS
  assert(LHSC != RHSC && "Compares not folded above?");

  switch (PredL) {
  default:
    llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_NE:  // (X == 13 & X != 15) -> X == 13
    case ICmpInst::ICMP_ULT: // (X == 13 & X <  15) -> X == 13
    case ICmpInst::ICMP_SLT: // (X == 13 & X <  15) -> X == 13
      return LHS;
    }
  case ICmpInst::ICMP_NE:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_ULT:
      if (LHSC == SubOne(RHSC)) // (X != 13 & X u< 14) -> X < 13
        return Builder->CreateICmpULT(Val, LHSC);
      if (LHSC->isNullValue()) // (X !=  0 & X u< 14) -> X-1 u< 13
        return insertRangeTest(Val, LHSC->getValue() + 1, RHSC->getValue(),
                               false, true);
      break; // (X != 13 & X u< 15) -> no change
    case ICmpInst::ICMP_SLT:
      if (LHSC == SubOne(RHSC)) // (X != 13 & X s< 14) -> X < 13
        return Builder->CreateICmpSLT(Val, LHSC);
      break;                 // (X != 13 & X s< 15) -> no change
    case ICmpInst::ICMP_EQ:  // (X != 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT: // (X != 13 & X u> 15) -> X u> 15
    case ICmpInst::ICMP_SGT: // (X != 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_NE:
      // Special case to get the ordering right when the values wrap around
      // zero.
      if (LHSC->getValue() == 0 && RHSC->getValue().isAllOnesValue())
        std::swap(LHSC, RHSC);
      if (LHSC == SubOne(RHSC)) { // (X != 13 & X != 14) -> X-13 >u 1
        Constant *AddC = ConstantExpr::getNeg(LHSC);
        Value *Add = Builder->CreateAdd(Val, AddC, Val->getName() + ".off");
        return Builder->CreateICmpUGT(Add, ConstantInt::get(Add->getType(), 1),
                                      Val->getName() + ".cmp");
      }
      break; // (X != 13 & X != 15) -> no change
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X u< 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT: // (X u< 13 & X u> 15) -> false
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    case ICmpInst::ICMP_NE:  // (X u< 13 & X != 15) -> X u< 13
    case ICmpInst::ICMP_ULT: // (X u< 13 & X u< 15) -> X u< 13
      return LHS;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_NE:  // (X s< 13 & X != 15) -> X < 13
    case ICmpInst::ICMP_SLT: // (X s< 13 & X s< 15) -> X < 13
      return LHS;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X u> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT: // (X u> 13 & X u> 15) -> X u> 15
      return RHS;
    case ICmpInst::ICMP_NE:
      if (RHSC == AddOne(LHSC)) // (X u> 13 & X != 14) -> X u> 14
        return Builder->CreateICmp(PredL, Val, RHSC);
      break;                 // (X u> 13 & X != 15) -> no change
    case ICmpInst::ICMP_ULT: // (X u> 13 & X u< 15) -> (X-14) <u 1
      return insertRangeTest(Val, LHSC->getValue() + 1, RHSC->getValue(), false,
                             true);
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X s> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_SGT: // (X s> 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_NE:
      if (RHSC == AddOne(LHSC)) // (X s> 13 & X != 14) -> X s> 14
        return Builder->CreateICmp(PredL, Val, RHSC);
      break;                 // (X s> 13 & X != 15) -> no change
    case ICmpInst::ICMP_SLT: // (X s> 13 & X s< 15) -> (X-14) s< 1
      return insertRangeTest(Val, LHSC->getValue() + 1, RHSC->getValue(), true,
                             true);
    }
    break;
  }

  return nullptr;
}

/// Optimize (fcmp)&(fcmp).  NOTE: Unlike the rest of instcombine, this returns
/// a Value which should already be inserted into the function.
Value *InstCombiner::FoldAndOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();

  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }

  // Simplify (fcmp cc0 x, y) & (fcmp cc1 x, y).
  // Suppose the relation between x and y is R, where R is one of
  // U(1000), L(0100), G(0010) or E(0001), and CC0 and CC1 are the bitmasks for
  // testing the desired relations.
  //
  // Since (R & CC0) and (R & CC1) are either R or 0, we actually have this:
  //    bool(R & CC0) && bool(R & CC1)
  //  = bool((R & CC0) & (R & CC1))
  //  = bool(R & (CC0 & CC1)) <= by re-association, commutation, and idempotency
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS)
    return getFCmpValue(getFCmpCode(Op0CC) & getFCmpCode(Op1CC), Op0LHS, Op0RHS,
                        Builder);

  if (LHS->getPredicate() == FCmpInst::FCMP_ORD &&
      RHS->getPredicate() == FCmpInst::FCMP_ORD) {
    if (LHS->getOperand(0)->getType() != RHS->getOperand(0)->getType())
      return nullptr;

    // (fcmp ord x, c) & (fcmp ord y, c)  -> (fcmp ord x, y)
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // false.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return Builder->getFalse();
        return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
      }

    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp ord x,x" is "fcmp ord x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
    return nullptr;
  }

  return nullptr;
}

/// Match De Morgan's Laws:
/// (~A & ~B) == (~(A | B))
/// (~A | ~B) == (~(A & B))
static Instruction *matchDeMorgansLaws(BinaryOperator &I,
                                       InstCombiner::BuilderTy *Builder) {
  auto Opcode = I.getOpcode();
  assert((Opcode == Instruction::And || Opcode == Instruction::Or) &&
         "Trying to match De Morgan's Laws with something other than and/or");
  // Flip the logic operation.
  if (Opcode == Instruction::And)
    Opcode = Instruction::Or;
  else
    Opcode = Instruction::And;

  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);
  // TODO: Use pattern matchers instead of dyn_cast.
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *LogicOp = Builder->CreateBinOp(Opcode, Op0NotVal, Op1NotVal,
                                              I.getName() + ".demorgan");
        return BinaryOperator::CreateNot(LogicOp);
      }

  return nullptr;
}

bool InstCombiner::shouldOptimizeCast(CastInst *CI) {
  Value *CastSrc = CI->getOperand(0);

  // Noop casts and casts of constants should be eliminated trivially.
  if (CI->getSrcTy() == CI->getDestTy() || isa<Constant>(CastSrc))
    return false;

  // If this cast is paired with another cast that can be eliminated, we prefer
  // to have it eliminated.
  if (const auto *PrecedingCI = dyn_cast<CastInst>(CastSrc))
    if (isEliminableCastPair(PrecedingCI, CI))
      return false;

  // If this is a vector sext from a compare, then we don't want to break the
  // idiom where each element of the extended vector is either zero or all ones.
  if (CI->getOpcode() == Instruction::SExt &&
      isa<CmpInst>(CastSrc) && CI->getDestTy()->isVectorTy())
    return false;

  return true;
}

/// Fold {and,or,xor} (cast X), C.
static Instruction *foldLogicCastConstant(BinaryOperator &Logic, CastInst *Cast,
                                          InstCombiner::BuilderTy *Builder) {
  Constant *C;
  if (!match(Logic.getOperand(1), m_Constant(C)))
    return nullptr;

  auto LogicOpc = Logic.getOpcode();
  Type *DestTy = Logic.getType();
  Type *SrcTy = Cast->getSrcTy();

  // If the first operand is bitcast, move the logic operation ahead of the
  // bitcast (do the logic operation in the original type). This can eliminate
  // bitcasts and allow combines that would otherwise be impeded by the bitcast.
  Value *X;
  if (match(Cast, m_BitCast(m_Value(X)))) {
    Value *NewConstant = ConstantExpr::getBitCast(C, SrcTy);
    Value *NewOp = Builder->CreateBinOp(LogicOpc, X, NewConstant);
    return CastInst::CreateBitOrPointerCast(NewOp, DestTy);
  }

  // Similarly, move the logic operation ahead of a zext if the constant is
  // unchanged in the smaller source type. Performing the logic in a smaller
  // type may provide more information to later folds, and the smaller logic
  // instruction may be cheaper (particularly in the case of vectors).
  if (match(Cast, m_OneUse(m_ZExt(m_Value(X))))) {
    Constant *TruncC = ConstantExpr::getTrunc(C, SrcTy);
    Constant *ZextTruncC = ConstantExpr::getZExt(TruncC, DestTy);
    if (ZextTruncC == C) {
      // LogicOpc (zext X), C --> zext (LogicOpc X, C)
      Value *NewOp = Builder->CreateBinOp(LogicOpc, X, TruncC);
      return new ZExtInst(NewOp, DestTy);
    }
  }

  return nullptr;
}

/// Fold {and,or,xor} (cast X), Y.
Instruction *InstCombiner::foldCastedBitwiseLogic(BinaryOperator &I) {
  auto LogicOpc = I.getOpcode();
  assert(I.isBitwiseLogicOp() && "Unexpected opcode for bitwise logic folding");

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  CastInst *Cast0 = dyn_cast<CastInst>(Op0);
  if (!Cast0)
    return nullptr;

  // This must be a cast from an integer or integer vector source type to allow
  // transformation of the logic operation to the source type.
  Type *DestTy = I.getType();
  Type *SrcTy = Cast0->getSrcTy();
  if (!SrcTy->isIntOrIntVectorTy())
    return nullptr;

  if (Instruction *Ret = foldLogicCastConstant(I, Cast0, Builder))
    return Ret;

  CastInst *Cast1 = dyn_cast<CastInst>(Op1);
  if (!Cast1)
    return nullptr;

  // Both operands of the logic operation are casts. The casts must be of the
  // same type for reduction.
  auto CastOpcode = Cast0->getOpcode();
  if (CastOpcode != Cast1->getOpcode() || SrcTy != Cast1->getSrcTy())
    return nullptr;

  Value *Cast0Src = Cast0->getOperand(0);
  Value *Cast1Src = Cast1->getOperand(0);

  // fold logic(cast(A), cast(B)) -> cast(logic(A, B))
  if (shouldOptimizeCast(Cast0) && shouldOptimizeCast(Cast1)) {
    Value *NewOp = Builder->CreateBinOp(LogicOpc, Cast0Src, Cast1Src,
                                        I.getName());
    return CastInst::Create(CastOpcode, NewOp, DestTy);
  }

  // For now, only 'and'/'or' have optimizations after this.
  if (LogicOpc == Instruction::Xor)
    return nullptr;

  // If this is logic(cast(icmp), cast(icmp)), try to fold this even if the
  // cast is otherwise not optimizable.  This happens for vector sexts.
  ICmpInst *ICmp0 = dyn_cast<ICmpInst>(Cast0Src);
  ICmpInst *ICmp1 = dyn_cast<ICmpInst>(Cast1Src);
  if (ICmp0 && ICmp1) {
    Value *Res = LogicOpc == Instruction::And ? FoldAndOfICmps(ICmp0, ICmp1)
                                              : FoldOrOfICmps(ICmp0, ICmp1, &I);
    if (Res)
      return CastInst::Create(CastOpcode, Res, DestTy);
    return nullptr;
  }

  // If this is logic(cast(fcmp), cast(fcmp)), try to fold this even if the
  // cast is otherwise not optimizable.  This happens for vector sexts.
  FCmpInst *FCmp0 = dyn_cast<FCmpInst>(Cast0Src);
  FCmpInst *FCmp1 = dyn_cast<FCmpInst>(Cast1Src);
  if (FCmp0 && FCmp1) {
    Value *Res = LogicOpc == Instruction::And ? FoldAndOfFCmps(FCmp0, FCmp1)
                                              : FoldOrOfFCmps(FCmp0, FCmp1);
    if (Res)
      return CastInst::Create(CastOpcode, Res, DestTy);
    return nullptr;
  }

  return nullptr;
}

static Instruction *foldBoolSextMaskToSelect(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Canonicalize SExt or Not to the LHS
  if (match(Op1, m_SExt(m_Value())) || match(Op1, m_Not(m_Value()))) {
    std::swap(Op0, Op1);
  }

  // Fold (and (sext bool to A), B) --> (select bool, B, 0)
  Value *X = nullptr;
  if (match(Op0, m_SExt(m_Value(X))) &&
      X->getType()->getScalarType()->isIntegerTy(1)) {
    Value *Zero = Constant::getNullValue(Op1->getType());
    return SelectInst::Create(X, Op1, Zero);
  }

  // Fold (and ~(sext bool to A), B) --> (select bool, 0, B)
  if (match(Op0, m_Not(m_SExt(m_Value(X)))) &&
      X->getType()->getScalarType()->isIntegerTy(1)) {
    Value *Zero = Constant::getNullValue(Op0->getType());
    return SelectInst::Create(X, Zero, Op1);
  }

  return nullptr;
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyAndInst(Op0, Op1, DL, &TLI, &DT, &AC))
    return replaceInstUsesWith(I, V);

  // (A|B)&(A|C) -> A|(B&C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return replaceInstUsesWith(I, V);

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
        if (MaskedValueIsZero(Op0LHS, NotAndRHS, 0, &I)) {
          // Not masking anything out for the LHS, move to RHS.
          Value *NewRHS = Builder->CreateAnd(Op0RHS, AndRHS,
                                             Op0RHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), Op0LHS, NewRHS);
        }
        if (!isa<Constant>(Op0RHS) &&
            MaskedValueIsZero(Op0RHS, NotAndRHS, 0, &I)) {
          // Not masking anything out for the RHS, move to LHS.
          Value *NewLHS = Builder->CreateAnd(Op0LHS, AndRHS,
                                             Op0LHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), NewLHS, Op0RHS);
        }

        break;
      }
      case Instruction::Sub:
        // -x & 1 -> x & 1
        if (AndRHSMask == 1 && match(Op0LHS, m_Zero()))
          return BinaryOperator::CreateAnd(Op0RHS, AndRHS);

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

      // ((C1 OP zext(X)) & C2) -> zext((C1-X) & C2) if C2 fits in the bitwidth
      // of X and OP behaves well when given trunc(C1) and X.
      switch (Op0I->getOpcode()) {
      default:
        break;
      case Instruction::Xor:
      case Instruction::Or:
      case Instruction::Mul:
      case Instruction::Add:
      case Instruction::Sub:
        Value *X;
        ConstantInt *C1;
        if (match(Op0I, m_BinOp(m_ZExt(m_Value(X)), m_ConstantInt(C1))) ||
            match(Op0I, m_BinOp(m_ConstantInt(C1), m_ZExt(m_Value(X))))) {
          if (AndRHSMask.isIntN(X->getType()->getScalarSizeInBits())) {
            auto *TruncC1 = ConstantExpr::getTrunc(C1, X->getType());
            Value *BinOp;
            if (isa<ZExtInst>(Op0LHS))
              BinOp = Builder->CreateBinOp(Op0I->getOpcode(), X, TruncC1);
            else
              BinOp = Builder->CreateBinOp(Op0I->getOpcode(), TruncC1, X);
            auto *TruncC2 = ConstantExpr::getTrunc(AndRHS, X->getType());
            auto *And = Builder->CreateAnd(BinOp, TruncC2);
            return new ZExtInst(And, I.getType());
          }
        }
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    }

    // If this is an integer truncation, and if the source is an 'and' with
    // immediate, transform it.  This frequently occurs for bitfield accesses.
    {
      Value *X = nullptr; ConstantInt *YC = nullptr;
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
  }

  if (isa<Constant>(Op1))
    if (Instruction *FoldedLogic = foldOpWithConstantIntoOperand(I))
      return FoldedLogic;

  if (Instruction *DeMorgan = matchDeMorgansLaws(I, Builder))
    return DeMorgan;

  {
    Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
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
      if (match(Op0, m_OneUse(m_Xor(m_Value(A), m_Value(B))))) {
        if (A == Op1 || B == Op1 ) {
          tmpOp1 = Op0;
          tmpOp0 = Op1;
          // Simplify below
        }
      }

      if (match(tmpOp1, m_OneUse(m_Xor(m_Value(A), m_Value(B))))) {
        if (B == tmpOp0) {
          std::swap(A, B);
        }
        // Notice that the pattern (A&(~B)) is actually (A&(-1^B)), so if
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

    // (A ^ B) & ((B ^ C) ^ A) -> (A ^ B) & ~C
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
      if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
        if (Op1->hasOneUse() || cast<BinaryOperator>(Op1)->hasOneUse())
          return BinaryOperator::CreateAnd(Op0, Builder->CreateNot(C));

    // ((A ^ C) ^ B) & (B ^ A) -> (B ^ A) & ~C
    if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
      if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
        if (Op0->hasOneUse() || cast<BinaryOperator>(Op0)->hasOneUse())
          return BinaryOperator::CreateAnd(Op1, Builder->CreateNot(C));

    // (A | B) & ((~A) ^ B) -> (A & B)
    if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1, m_Xor(m_Not(m_Specific(A)), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);

    // ((~A) ^ B) & (A | B) -> (A & B)
    // ((~A) ^ B) & (B | A) -> (A & B)
    if (match(Op0, m_Xor(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1, m_c_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);
  }

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = FoldAndOfICmps(LHS, RHS))
        return replaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'and' instructions might have to be created.
    Value *X, *Y;
    if (LHS && match(Op1, m_OneUse(m_And(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldAndOfICmps(LHS, Cmp))
          return replaceInstUsesWith(I, Builder->CreateAnd(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldAndOfICmps(LHS, Cmp))
          return replaceInstUsesWith(I, Builder->CreateAnd(Res, X));
    }
    if (RHS && match(Op0, m_OneUse(m_And(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldAndOfICmps(Cmp, RHS))
          return replaceInstUsesWith(I, Builder->CreateAnd(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldAndOfICmps(Cmp, RHS))
          return replaceInstUsesWith(I, Builder->CreateAnd(Res, X));
    }
  }

  // If and'ing two fcmp, try combine them into one.
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldAndOfFCmps(LHS, RHS))
        return replaceInstUsesWith(I, Res);

  if (Instruction *CastedAnd = foldCastedBitwiseLogic(I))
    return CastedAnd;

  if (Instruction *Select = foldBoolSextMaskToSelect(I))
    return Select;

  return Changed ? &I : nullptr;
}

/// Given an OR instruction, check to see if this is a bswap idiom. If so,
/// insert the new intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Look through zero extends.
  if (Instruction *Ext = dyn_cast<ZExtInst>(Op0))
    Op0 = Ext->getOperand(0);

  if (Instruction *Ext = dyn_cast<ZExtInst>(Op1))
    Op1 = Ext->getOperand(0);

  // (A | B) | C  and  A | (B | C)                  -> bswap if possible.
  bool OrOfOrs = match(Op0, m_Or(m_Value(), m_Value())) ||
                 match(Op1, m_Or(m_Value(), m_Value()));

  // (A >> B) | (C << D)  and  (A << B) | (B >> C)  -> bswap if possible.
  bool OrOfShifts = match(Op0, m_LogicalShift(m_Value(), m_Value())) &&
                    match(Op1, m_LogicalShift(m_Value(), m_Value()));

  // (A & B) | (C & D)                              -> bswap if possible.
  bool OrOfAnds = match(Op0, m_And(m_Value(), m_Value())) &&
                  match(Op1, m_And(m_Value(), m_Value()));

  if (!OrOfOrs && !OrOfShifts && !OrOfAnds)
    return nullptr;

  SmallVector<Instruction*, 4> Insts;
  if (!recognizeBSwapOrBitReverseIdiom(&I, true, false, Insts))
    return nullptr;
  Instruction *LastInst = Insts.pop_back_val();
  LastInst->removeFromParent();

  for (auto *Inst : Insts)
    Worklist.Add(Inst);
  return LastInst;
}

/// If all elements of two constant vectors are 0/-1 and inverses, return true.
static bool areInverseVectorBitmasks(Constant *C1, Constant *C2) {
  unsigned NumElts = C1->getType()->getVectorNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *EltC1 = C1->getAggregateElement(i);
    Constant *EltC2 = C2->getAggregateElement(i);
    if (!EltC1 || !EltC2)
      return false;

    // One element must be all ones, and the other must be all zeros.
    // FIXME: Allow undef elements.
    if (!((match(EltC1, m_Zero()) && match(EltC2, m_AllOnes())) ||
          (match(EltC2, m_Zero()) && match(EltC1, m_AllOnes()))))
      return false;
  }
  return true;
}

/// We have an expression of the form (A & C) | (B & D). If A is a scalar or
/// vector composed of all-zeros or all-ones values and is the bitwise 'not' of
/// B, it can be used as the condition operand of a select instruction.
static Value *getSelectCondition(Value *A, Value *B,
                                 InstCombiner::BuilderTy &Builder) {
  // If these are scalars or vectors of i1, A can be used directly.
  Type *Ty = A->getType();
  if (match(A, m_Not(m_Specific(B))) && Ty->getScalarType()->isIntegerTy(1))
    return A;

  // If A and B are sign-extended, look through the sexts to find the booleans.
  Value *Cond;
  if (match(A, m_SExt(m_Value(Cond))) &&
      Cond->getType()->getScalarType()->isIntegerTy(1) &&
      match(B, m_CombineOr(m_Not(m_SExt(m_Specific(Cond))),
                           m_SExt(m_Not(m_Specific(Cond))))))
    return Cond;

  // All scalar (and most vector) possibilities should be handled now.
  // Try more matches that only apply to non-splat constant vectors.
  if (!Ty->isVectorTy())
    return nullptr;

  // If both operands are constants, see if the constants are inverse bitmasks.
  Constant *AC, *BC;
  if (match(A, m_Constant(AC)) && match(B, m_Constant(BC)) &&
      areInverseVectorBitmasks(AC, BC))
    return ConstantExpr::getTrunc(AC, CmpInst::makeCmpResultType(Ty));

  // If both operands are xor'd with constants using the same sexted boolean
  // operand, see if the constants are inverse bitmasks.
  if (match(A, (m_Xor(m_SExt(m_Value(Cond)), m_Constant(AC)))) &&
      match(B, (m_Xor(m_SExt(m_Specific(Cond)), m_Constant(BC)))) &&
      Cond->getType()->getScalarType()->isIntegerTy(1) &&
      areInverseVectorBitmasks(AC, BC)) {
    AC = ConstantExpr::getTrunc(AC, CmpInst::makeCmpResultType(Ty));
    return Builder.CreateXor(Cond, AC);
  }
  return nullptr;
}

/// We have an expression of the form (A & C) | (B & D). Try to simplify this
/// to "A' ? C : D", where A' is a boolean or vector of booleans.
static Value *matchSelectFromAndOr(Value *A, Value *C, Value *B, Value *D,
                                   InstCombiner::BuilderTy &Builder) {
  // The potential condition of the select may be bitcasted. In that case, look
  // through its bitcast and the corresponding bitcast of the 'not' condition.
  Type *OrigType = A->getType();
  Value *SrcA, *SrcB;
  if (match(A, m_OneUse(m_BitCast(m_Value(SrcA)))) &&
      match(B, m_OneUse(m_BitCast(m_Value(SrcB))))) {
    A = SrcA;
    B = SrcB;
  }

  if (Value *Cond = getSelectCondition(A, B, Builder)) {
    // ((bc Cond) & C) | ((bc ~Cond) & D) --> bc (select Cond, (bc C), (bc D))
    // The bitcasts will either all exist or all not exist. The builder will
    // not create unnecessary casts if the types already match.
    Value *BitcastC = Builder.CreateBitCast(C, A->getType());
    Value *BitcastD = Builder.CreateBitCast(D, A->getType());
    Value *Select = Builder.CreateSelect(Cond, BitcastC, BitcastD);
    return Builder.CreateBitCast(Select, OrigType);
  }

  return nullptr;
}

/// Fold (icmp)|(icmp) if possible.
Value *InstCombiner::FoldOrOfICmps(ICmpInst *LHS, ICmpInst *RHS,
                                   Instruction *CxtI) {
  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();

  // Fold (iszero(A & K1) | iszero(A & K2)) ->  (A & (K1 | K2)) != (K1 | K2)
  // if K1 and K2 are a one-bit mask.
  ConstantInt *LHSC = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS->getOperand(1));

  if (LHS->getPredicate() == ICmpInst::ICMP_EQ && LHSC && LHSC->isZero() &&
      RHS->getPredicate() == ICmpInst::ICMP_EQ && RHSC && RHSC->isZero()) {

    BinaryOperator *LAnd = dyn_cast<BinaryOperator>(LHS->getOperand(0));
    BinaryOperator *RAnd = dyn_cast<BinaryOperator>(RHS->getOperand(0));
    if (LAnd && RAnd && LAnd->hasOneUse() && RHS->hasOneUse() &&
        LAnd->getOpcode() == Instruction::And &&
        RAnd->getOpcode() == Instruction::And) {

      Value *Mask = nullptr;
      Value *Masked = nullptr;
      if (LAnd->getOperand(0) == RAnd->getOperand(0) &&
          isKnownToBeAPowerOfTwo(LAnd->getOperand(1), DL, false, 0, &AC, CxtI,
                                 &DT) &&
          isKnownToBeAPowerOfTwo(RAnd->getOperand(1), DL, false, 0, &AC, CxtI,
                                 &DT)) {
        Mask = Builder->CreateOr(LAnd->getOperand(1), RAnd->getOperand(1));
        Masked = Builder->CreateAnd(LAnd->getOperand(0), Mask);
      } else if (LAnd->getOperand(1) == RAnd->getOperand(1) &&
                 isKnownToBeAPowerOfTwo(LAnd->getOperand(0), DL, false, 0, &AC,
                                        CxtI, &DT) &&
                 isKnownToBeAPowerOfTwo(RAnd->getOperand(0), DL, false, 0, &AC,
                                        CxtI, &DT)) {
        Mask = Builder->CreateOr(LAnd->getOperand(0), RAnd->getOperand(0));
        Masked = Builder->CreateAnd(LAnd->getOperand(1), Mask);
      }

      if (Masked)
        return Builder->CreateICmp(ICmpInst::ICMP_NE, Masked, Mask);
    }
  }

  // Fold (icmp ult/ule (A + C1), C3) | (icmp ult/ule (A + C2), C3)
  //                   -->  (icmp ult/ule ((A & ~(C1 ^ C2)) + max(C1, C2)), C3)
  // The original condition actually refers to the following two ranges:
  // [MAX_UINT-C1+1, MAX_UINT-C1+1+C3] and [MAX_UINT-C2+1, MAX_UINT-C2+1+C3]
  // We can fold these two ranges if:
  // 1) C1 and C2 is unsigned greater than C3.
  // 2) The two ranges are separated.
  // 3) C1 ^ C2 is one-bit mask.
  // 4) LowRange1 ^ LowRange2 and HighRange1 ^ HighRange2 are one-bit mask.
  // This implies all values in the two ranges differ by exactly one bit.

  if ((PredL == ICmpInst::ICMP_ULT || PredL == ICmpInst::ICMP_ULE) &&
      PredL == PredR && LHSC && RHSC && LHS->hasOneUse() && RHS->hasOneUse() &&
      LHSC->getType() == RHSC->getType() &&
      LHSC->getValue() == (RHSC->getValue())) {

    Value *LAdd = LHS->getOperand(0);
    Value *RAdd = RHS->getOperand(0);

    Value *LAddOpnd, *RAddOpnd;
    ConstantInt *LAddC, *RAddC;
    if (match(LAdd, m_Add(m_Value(LAddOpnd), m_ConstantInt(LAddC))) &&
        match(RAdd, m_Add(m_Value(RAddOpnd), m_ConstantInt(RAddC))) &&
        LAddC->getValue().ugt(LHSC->getValue()) &&
        RAddC->getValue().ugt(LHSC->getValue())) {

      APInt DiffC = LAddC->getValue() ^ RAddC->getValue();
      if (LAddOpnd == RAddOpnd && DiffC.isPowerOf2()) {
        ConstantInt *MaxAddC = nullptr;
        if (LAddC->getValue().ult(RAddC->getValue()))
          MaxAddC = RAddC;
        else
          MaxAddC = LAddC;

        APInt RRangeLow = -RAddC->getValue();
        APInt RRangeHigh = RRangeLow + LHSC->getValue();
        APInt LRangeLow = -LAddC->getValue();
        APInt LRangeHigh = LRangeLow + LHSC->getValue();
        APInt LowRangeDiff = RRangeLow ^ LRangeLow;
        APInt HighRangeDiff = RRangeHigh ^ LRangeHigh;
        APInt RangeDiff = LRangeLow.sgt(RRangeLow) ? LRangeLow - RRangeLow
                                                   : RRangeLow - LRangeLow;

        if (LowRangeDiff.isPowerOf2() && LowRangeDiff == HighRangeDiff &&
            RangeDiff.ugt(LHSC->getValue())) {
          Value *MaskC = ConstantInt::get(LAddC->getType(), ~DiffC);

          Value *NewAnd = Builder->CreateAnd(LAddOpnd, MaskC);
          Value *NewAdd = Builder->CreateAdd(NewAnd, MaxAddC);
          return (Builder->CreateICmp(LHS->getPredicate(), NewAdd, LHSC));
        }
      }
    }
  }

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(PredL, PredR)) {
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
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, false, Builder))
    return V;

  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  if (LHS->hasOneUse() || RHS->hasOneUse()) {
    // (icmp eq B, 0) | (icmp ult A, B) -> (icmp ule A, B-1)
    // (icmp eq B, 0) | (icmp ugt B, A) -> (icmp ule A, B-1)
    Value *A = nullptr, *B = nullptr;
    if (PredL == ICmpInst::ICMP_EQ && LHSC && LHSC->isZero()) {
      B = Val;
      if (PredR == ICmpInst::ICMP_ULT && Val == RHS->getOperand(1))
        A = Val2;
      else if (PredR == ICmpInst::ICMP_UGT && Val == Val2)
        A = RHS->getOperand(1);
    }
    // (icmp ult A, B) | (icmp eq B, 0) -> (icmp ule A, B-1)
    // (icmp ugt B, A) | (icmp eq B, 0) -> (icmp ule A, B-1)
    else if (PredR == ICmpInst::ICMP_EQ && RHSC && RHSC->isZero()) {
      B = Val2;
      if (PredL == ICmpInst::ICMP_ULT && Val2 == LHS->getOperand(1))
        A = Val;
      else if (PredL == ICmpInst::ICMP_UGT && Val2 == Val)
        A = LHS->getOperand(1);
    }
    if (A && B)
      return Builder->CreateICmp(
          ICmpInst::ICMP_UGE,
          Builder->CreateAdd(B, ConstantInt::getSigned(B->getType(), -1)), A);
  }

  // E.g. (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
  if (Value *V = simplifyRangeCheck(LHS, RHS, /*Inverted=*/true))
    return V;

  // E.g. (icmp sgt x, n) | (icmp slt x, 0) --> icmp ugt x, n
  if (Value *V = simplifyRangeCheck(RHS, LHS, /*Inverted=*/true))
    return V;

  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  if (!LHSC || !RHSC)
    return nullptr;

  if (LHSC == RHSC && PredL == PredR) {
    // (icmp ne A, 0) | (icmp ne B, 0) --> (icmp ne (A|B), 0)
    if (PredL == ICmpInst::ICMP_NE && LHSC->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(PredL, NewOr, LHSC);
    }
  }

  // (icmp ult (X + CA), C1) | (icmp eq X, C2) -> (icmp ule (X + CA), C1)
  //   iff C2 + CA == C1.
  if (PredL == ICmpInst::ICMP_ULT && PredR == ICmpInst::ICMP_EQ) {
    ConstantInt *AddC;
    if (match(Val, m_Add(m_Specific(Val2), m_ConstantInt(AddC))))
      if (RHSC->getValue() + AddC->getValue() == LHSC->getValue())
        return Builder->CreateICmpULE(Val, LHSC);
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) | (icmp2 A, C2) --> something simpler.
  if (Val != Val2)
    return nullptr;

  // ICMP_[US][GL]E X, C is folded to ICMP_[US][GL]T elsewhere.
  if (PredL == ICmpInst::ICMP_UGE || PredL == ICmpInst::ICMP_ULE ||
      PredR == ICmpInst::ICMP_UGE || PredR == ICmpInst::ICMP_ULE ||
      PredL == ICmpInst::ICMP_SGE || PredL == ICmpInst::ICMP_SLE ||
      PredR == ICmpInst::ICMP_SGE || PredR == ICmpInst::ICMP_SLE)
    return nullptr;

  // We can't fold (ugt x, C) | (sgt x, C2).
  if (!PredicatesFoldable(PredL, PredR))
    return nullptr;

  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(PredL) ||
      (ICmpInst::isEquality(PredL) && CmpInst::isSigned(PredR)))
    ShouldSwap = LHSC->getValue().sgt(RHSC->getValue());
  else
    ShouldSwap = LHSC->getValue().ugt(RHSC->getValue());

  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSC, RHSC);
    std::swap(PredL, PredR);
  }

  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and or'ing the result
  // together.  Because of the above check, we know that we only have
  // ICMP_EQ, ICMP_NE, ICMP_LT, and ICMP_GT here. We also know (from the
  // icmp folding check above), that the two constants are not
  // equal.
  assert(LHSC != RHSC && "Compares not folded above?");

  switch (PredL) {
  default:
    llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:
      if (LHS->getOperand(0) == RHS->getOperand(0)) {
        // if LHSC and RHSC differ only by one bit:
        // (A == C1 || A == C2) -> (A | (C1 ^ C2)) == C2
        assert(LHSC->getValue().ule(LHSC->getValue()));

        APInt Xor = LHSC->getValue() ^ RHSC->getValue();
        if (Xor.isPowerOf2()) {
          Value *C = Builder->getInt(Xor);
          Value *Or = Builder->CreateOr(LHS->getOperand(0), C);
          return Builder->CreateICmp(ICmpInst::ICMP_EQ, Or, RHSC);
        }
      }

      if (LHSC == SubOne(RHSC)) {
        // (X == 13 | X == 14) -> X-13 <u 2
        Constant *AddC = ConstantExpr::getNeg(LHSC);
        Value *Add = Builder->CreateAdd(Val, AddC, Val->getName() + ".off");
        AddC = ConstantExpr::getSub(AddOne(RHSC), LHSC);
        return Builder->CreateICmpULT(Add, AddC);
      }

      break;                 // (X == 13 | X == 15) -> no change
    case ICmpInst::ICMP_UGT: // (X == 13 | X u> 14) -> no change
    case ICmpInst::ICMP_SGT: // (X == 13 | X s> 14) -> no change
      break;
    case ICmpInst::ICMP_NE:  // (X == 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT: // (X == 13 | X u< 15) -> X u< 15
    case ICmpInst::ICMP_SLT: // (X == 13 | X s< 15) -> X s< 15
      return RHS;
    }
    break;
  case ICmpInst::ICMP_NE:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X != 13 | X == 15) -> X != 13
    case ICmpInst::ICMP_UGT: // (X != 13 | X u> 15) -> X != 13
    case ICmpInst::ICMP_SGT: // (X != 13 | X s> 15) -> X != 13
      return LHS;
    case ICmpInst::ICMP_NE:  // (X != 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT: // (X != 13 | X u< 15) -> true
    case ICmpInst::ICMP_SLT: // (X != 13 | X s< 15) -> true
      return Builder->getTrue();
    }
  case ICmpInst::ICMP_ULT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ: // (X u< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_UGT: // (X u< 13 | X u> 15) -> (X-13) u> 2
      // If RHSC is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSC->isMaxValue(false))
        return LHS;
      return insertRangeTest(Val, LHSC->getValue(), RHSC->getValue() + 1, false,
                             false);
    case ICmpInst::ICMP_NE:  // (X u< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT: // (X u< 13 | X u< 15) -> X u< 15
      return RHS;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ: // (X s< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_SGT: // (X s< 13 | X s> 15) -> (X-13) s> 2
      // If RHSC is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSC->isMaxValue(true))
        return LHS;
      return insertRangeTest(Val, LHSC->getValue(), RHSC->getValue() + 1, true,
                             false);
    case ICmpInst::ICMP_NE:  // (X s< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_SLT: // (X s< 13 | X s< 15) -> X s< 15
      return RHS;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X u> 13 | X == 15) -> X u> 13
    case ICmpInst::ICMP_UGT: // (X u> 13 | X u> 15) -> X u> 13
      return LHS;
    case ICmpInst::ICMP_NE:  // (X u> 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT: // (X u> 13 | X u< 15) -> true
      return Builder->getTrue();
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (PredR) {
    default:
      llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:  // (X s> 13 | X == 15) -> X > 13
    case ICmpInst::ICMP_SGT: // (X s> 13 | X s> 15) -> X > 13
      return LHS;
    case ICmpInst::ICMP_NE:  // (X s> 13 | X != 15) -> true
    case ICmpInst::ICMP_SLT: // (X s> 13 | X s< 15) -> true
      return Builder->getTrue();
    }
    break;
  }
  return nullptr;
}

/// Optimize (fcmp)|(fcmp).  NOTE: Unlike the rest of instcombine, this returns
/// a Value which should already be inserted into the function.
Value *InstCombiner::FoldOrOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();

  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }

  // Simplify (fcmp cc0 x, y) | (fcmp cc1 x, y).
  // This is a similar transformation to the one in FoldAndOfFCmps.
  //
  // Since (R & CC0) and (R & CC1) are either R or 0, we actually have this:
  //    bool(R & CC0) || bool(R & CC1)
  //  = bool((R & CC0) | (R & CC1))
  //  = bool(R & (CC0 | CC1)) <= by reversed distribution (contribution? ;)
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS)
    return getFCmpValue(getFCmpCode(Op0CC) | getFCmpCode(Op1CC), Op0LHS, Op0RHS,
                        Builder);

  if (LHS->getPredicate() == FCmpInst::FCMP_UNO &&
      RHS->getPredicate() == FCmpInst::FCMP_UNO &&
      LHS->getOperand(0)->getType() == RHS->getOperand(0)->getType()) {
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // true.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return Builder->getTrue();

        // Otherwise, no need to compare the two constants, compare the
        // rest.
        return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));
      }

    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp uno x,x" is "fcmp uno x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));

    return nullptr;
  }

  return nullptr;
}

/// This helper function folds:
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
  if (!CI1) return nullptr;

  Value *V1 = nullptr;
  ConstantInt *CI2 = nullptr;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2)))) return nullptr;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue()) return nullptr;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd((V1 == A) ? B : A, CI1);
    return BinaryOperator::CreateOr(NewOp, V1);
  }

  return nullptr;
}

/// \brief This helper function folds:
///
///     ((A | B) & C1) ^ (B & C2)
///
/// into:
///
///     (A & C1) ^ B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldXorWithConstants(BinaryOperator &I, Value *Op,
                                                Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1)
    return nullptr;

  Value *V1 = nullptr;
  ConstantInt *CI2 = nullptr;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2))))
    return nullptr;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue())
    return nullptr;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd(V1 == A ? B : A, CI1);
    return BinaryOperator::CreateXor(NewOp, V1);
  }

  return nullptr;
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyOrInst(Op0, Op1, DL, &TLI, &DT, &AC))
    return replaceInstUsesWith(I, V);

  // (A&B)|(A&C) -> A&(B|C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return replaceInstUsesWith(I, V);

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    ConstantInt *C1 = nullptr; Value *X = nullptr;
    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) &&
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateXor(Or,
                            Builder->getInt(C1->getValue() & ~RHS->getValue()));
    }
  }

  if (isa<Constant>(Op1))
    if (Instruction *FoldedLogic = foldOpWithConstantIntoOperand(I))
      return FoldedLogic;

  // Given an OR instruction, check to see if this is a bswap.
  if (Instruction *BSwap = MatchBSwap(I))
    return BSwap;

  {
    Value *A;
    const APInt *C;
    // (X^C)|Y -> (X|Y)^C iff Y&C == 0
    if (match(Op0, m_OneUse(m_Xor(m_Value(A), m_APInt(C)))) &&
        MaskedValueIsZero(Op1, *C, 0, &I)) {
      Value *NOr = Builder->CreateOr(A, Op1);
      NOr->takeName(Op0);
      return BinaryOperator::CreateXor(NOr,
                                       cast<Instruction>(Op0)->getOperand(1));
    }

    // Y|(X^C) -> (X|Y)^C iff Y&C == 0
    if (match(Op1, m_OneUse(m_Xor(m_Value(A), m_APInt(C)))) &&
        MaskedValueIsZero(Op0, *C, 0, &I)) {
      Value *NOr = Builder->CreateOr(A, Op0);
      NOr->takeName(Op0);
      return BinaryOperator::CreateXor(NOr,
                                       cast<Instruction>(Op1)->getOperand(1));
    }
  }

  Value *A, *B;

  // ((~A & B) | A) -> (A | B)
  if (match(Op0, m_c_And(m_Not(m_Specific(Op1)), m_Value(A))))
    return BinaryOperator::CreateOr(A, Op1);
  if (match(Op1, m_c_And(m_Not(m_Specific(Op0)), m_Value(A))))
    return BinaryOperator::CreateOr(Op0, A);

  // ((A & B) | ~A) -> (~A | B)
  // The NOT is guaranteed to be in the RHS by complexity ordering.
  if (match(Op1, m_Not(m_Value(A))) &&
      match(Op0, m_c_And(m_Specific(A), m_Value(B))))
    return BinaryOperator::CreateOr(Op1, B);

  // (A & ~B) | (A ^ B) -> (A ^ B)
  // (~B & A) | (A ^ B) -> (A ^ B)
  if (match(Op0, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_Xor(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateXor(A, B);

  // Commute the 'or' operands.
  // (A ^ B) | (A & ~B) -> (A ^ B)
  // (A ^ B) | (~B & A) -> (A ^ B)
  if (match(Op1, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op0, m_Xor(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateXor(A, B);

  // (A & C)|(B & D)
  Value *C = nullptr, *D = nullptr;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    Value *V1 = nullptr, *V2 = nullptr;
    ConstantInt *C1 = dyn_cast<ConstantInt>(C);
    ConstantInt *C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2) {  // (A & C1)|(B & C2)
      if ((C1->getValue() & C2->getValue()) == 0) {
        // ((V | N) & C1) | (V & C2) --> (V|N) & (C1|C2)
        // iff (C1&C2) == 0 and (N&~C1) == 0
        if (match(A, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == B &&
              MaskedValueIsZero(V2, ~C1->getValue(), 0, &I)) || // (V|N)
             (V2 == B &&
              MaskedValueIsZero(V1, ~C1->getValue(), 0, &I))))  // (N|V)
          return BinaryOperator::CreateAnd(A,
                                Builder->getInt(C1->getValue()|C2->getValue()));
        // Or commutes, try both ways.
        if (match(B, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == A &&
              MaskedValueIsZero(V2, ~C2->getValue(), 0, &I)) || // (V|N)
             (V2 == A &&
              MaskedValueIsZero(V1, ~C2->getValue(), 0, &I))))  // (N|V)
          return BinaryOperator::CreateAnd(B,
                                Builder->getInt(C1->getValue()|C2->getValue()));

        // ((V|C3)&C1) | ((V|C4)&C2) --> (V|C3|C4)&(C1|C2)
        // iff (C1&C2) == 0 and (C3&~C1) == 0 and (C4&~C2) == 0.
        ConstantInt *C3 = nullptr, *C4 = nullptr;
        if (match(A, m_Or(m_Value(V1), m_ConstantInt(C3))) &&
            (C3->getValue() & ~C1->getValue()) == 0 &&
            match(B, m_Or(m_Specific(V1), m_ConstantInt(C4))) &&
            (C4->getValue() & ~C2->getValue()) == 0) {
          V2 = Builder->CreateOr(V1, ConstantExpr::getOr(C3, C4), "bitfield");
          return BinaryOperator::CreateAnd(V2,
                                Builder->getInt(C1->getValue()|C2->getValue()));
        }
      }
    }

    // Don't try to form a select if it's unlikely that we'll get rid of at
    // least one of the operands. A select is generally more expensive than the
    // 'or' that it is replacing.
    if (Op0->hasOneUse() || Op1->hasOneUse()) {
      // (Cond & C) | (~Cond & D) -> Cond ? C : D, and commuted variants.
      if (Value *V = matchSelectFromAndOr(A, C, B, D, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(A, C, D, B, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(C, A, B, D, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(C, A, D, B, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(B, D, A, C, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(B, D, C, A, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(D, B, A, C, *Builder))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(D, B, C, A, *Builder))
        return replaceInstUsesWith(I, V);
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
    // ((A^B)&1)|(B&-2) -> (A&1) ^ B
    if (match(A, m_Xor(m_Value(V1), m_Specific(B))) ||
        match(A, m_Xor(m_Specific(B), m_Value(V1)))) {
      Instruction *Ret = FoldXorWithConstants(I, Op1, V1, B, C);
      if (Ret) return Ret;
    }
    // (B&-2)|((A^B)&1) -> (A&1) ^ B
    if (match(B, m_Xor(m_Specific(A), m_Value(V1))) ||
        match(B, m_Xor(m_Value(V1), m_Specific(A)))) {
      Instruction *Ret = FoldXorWithConstants(I, Op0, A, V1, D);
      if (Ret) return Ret;
    }
  }

  // (A ^ B) | ((B ^ C) ^ A) -> (A ^ B) | C
  if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
    if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
      if (Op1->hasOneUse() || cast<BinaryOperator>(Op1)->hasOneUse())
        return BinaryOperator::CreateOr(Op0, C);

  // ((A ^ C) ^ B) | (B ^ A) -> (B ^ A) | C
  if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
    if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
      if (Op0->hasOneUse() || cast<BinaryOperator>(Op0)->hasOneUse())
        return BinaryOperator::CreateOr(Op1, C);

  // ((B | C) & A) | B -> B | (A & C)
  if (match(Op0, m_And(m_Or(m_Specific(Op1), m_Value(C)), m_Value(A))))
    return BinaryOperator::CreateOr(Op1, Builder->CreateAnd(A, C));

  if (Instruction *DeMorgan = matchDeMorgansLaws(I, Builder))
    return DeMorgan;

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

  // (A & B) | (~A ^ B) -> (~A ^ B)
  // (A & B) | (B ^ ~A) -> (~A ^ B)
  // (B & A) | (~A ^ B) -> (~A ^ B)
  // (B & A) | (B ^ ~A) -> (~A ^ B)
  // The match order is important: match the xor first because the 'not'
  // operation defines 'A'. We do not need to match the xor as Op0 because the
  // xor was canonicalized to Op1 above.
  if (match(Op1, m_c_Xor(m_Not(m_Value(A)), m_Value(B))) &&
      match(Op0, m_c_And(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateXor(Builder->CreateNot(A), B);

  if (SwappedForXor)
    std::swap(Op0, Op1);

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = FoldOrOfICmps(LHS, RHS, &I))
        return replaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'or' instructions might have to be created.
    Value *X, *Y;
    if (LHS && match(Op1, m_OneUse(m_Or(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldOrOfICmps(LHS, Cmp, &I))
          return replaceInstUsesWith(I, Builder->CreateOr(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldOrOfICmps(LHS, Cmp, &I))
          return replaceInstUsesWith(I, Builder->CreateOr(Res, X));
    }
    if (RHS && match(Op0, m_OneUse(m_Or(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldOrOfICmps(Cmp, RHS, &I))
          return replaceInstUsesWith(I, Builder->CreateOr(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldOrOfICmps(Cmp, RHS, &I))
          return replaceInstUsesWith(I, Builder->CreateOr(Res, X));
    }
  }

  // (fcmp uno x, c) | (fcmp uno y, c)  -> (fcmp uno x, y)
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldOrOfFCmps(LHS, RHS))
        return replaceInstUsesWith(I, Res);

  if (Instruction *CastedOr = foldCastedBitwiseLogic(I))
    return CastedOr;

  // or(sext(A), B) / or(B, sext(A)) --> A ? -1 : B, where A is i1 or <N x i1>.
  if (match(Op0, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->getScalarType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op1);
  if (match(Op1, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->getScalarType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op0);

  // Note: If we've gotten to the point of visiting the outer OR, then the
  // inner one couldn't be simplified.  If it was a constant, then it won't
  // be simplified by a later pass either, so we try swapping the inner/outer
  // ORs in the hopes that we'll be able to simplify it this way.
  // (X|C) | V --> (X|V) | C
  ConstantInt *C1;
  if (Op0->hasOneUse() && !isa<ConstantInt>(Op1) &&
      match(Op0, m_Or(m_Value(A), m_ConstantInt(C1)))) {
    Value *Inner = Builder->CreateOr(A, Op1);
    Inner->takeName(Op0);
    return BinaryOperator::CreateOr(Inner, C1);
  }

  // Change (or (bool?A:B),(bool?C:D)) --> (bool?(or A,C):(or B,D))
  // Since this OR statement hasn't been optimized further yet, we hope
  // that this transformation will allow the new ORs to be optimized.
  {
    Value *X = nullptr, *Y = nullptr;
    if (Op0->hasOneUse() && Op1->hasOneUse() &&
        match(Op0, m_Select(m_Value(X), m_Value(A), m_Value(B))) &&
        match(Op1, m_Select(m_Value(Y), m_Value(C), m_Value(D))) && X == Y) {
      Value *orTrue = Builder->CreateOr(A, C);
      Value *orFalse = Builder->CreateOr(B, D);
      return SelectInst::Create(X, orTrue, orFalse);
    }
  }

  return Changed ? &I : nullptr;
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyXorInst(Op0, Op1, DL, &TLI, &DT, &AC))
    return replaceInstUsesWith(I, V);

  // (A&B)^(A&C) -> A&(B^C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return replaceInstUsesWith(I, V);

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
        if (IsFreeToInvert(Op0I->getOperand(0),
                           Op0I->getOperand(0)->hasOneUse()) &&
            IsFreeToInvert(Op0I->getOperand(1),
                           Op0I->getOperand(1)->hasOneUse())) {
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

  if (Constant *RHS = dyn_cast<Constant>(Op1)) {
    if (RHS->isAllOnesValue() && Op0->hasOneUse())
      // xor (cmp A, B), true = not (cmp A, B) = !cmp A, B
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0))
        return CmpInst::Create(CI->getOpcode(),
                               CI->getInversePredicate(),
                               CI->getOperand(0), CI->getOperand(1));
  }

  if (ConstantInt *RHSC = dyn_cast<ConstantInt>(Op1)) {
    // fold (xor(zext(cmp)), 1) and (xor(sext(cmp)), -1) to ext(!cmp).
    if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0C->getOperand(0))) {
        if (CI->hasOneUse() && Op0C->hasOneUse()) {
          Instruction::CastOps Opcode = Op0C->getOpcode();
          if ((Opcode == Instruction::ZExt || Opcode == Instruction::SExt) &&
              (RHSC == ConstantExpr::getCast(Opcode, Builder->getTrue(),
                                            Op0C->getDestTy()))) {
            CI->setPredicate(CI->getInversePredicate());
            return CastInst::Create(Opcode, CI, Op0C->getType());
          }
        }
      }
    }

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHSC->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          return BinaryOperator::CreateAdd(Op0I->getOperand(1),
                                           SubOne(NegOp0I0C));
        }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHSC->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::CreateSub(SubOne(NegOp0CI),
                                             Op0I->getOperand(0));
          } else if (RHSC->getValue().isSignBit()) {
            // (X + C) ^ signbit -> (X + C + signbit)
            Constant *C = Builder->getInt(RHSC->getValue() + Op0CI->getValue());
            return BinaryOperator::CreateAdd(Op0I->getOperand(0), C);

          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getValue(),
                                0, &I)) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHSC);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = ConstantExpr::getAnd(Op0CI, RHSC);
            NewRHS = ConstantExpr::getAnd(NewRHS,
                                       ConstantExpr::getNot(CommonBits));
            Worklist.Add(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
        } else if (Op0I->getOpcode() == Instruction::LShr) {
          // ((X^C1) >> C2) ^ C3 -> (X>>C2) ^ ((C1>>C2)^C3)
          // E1 = "X ^ C1"
          BinaryOperator *E1;
          ConstantInt *C1;
          if (Op0I->hasOneUse() &&
              (E1 = dyn_cast<BinaryOperator>(Op0I->getOperand(0))) &&
              E1->getOpcode() == Instruction::Xor &&
              (C1 = dyn_cast<ConstantInt>(E1->getOperand(1)))) {
            // fold (C1 >> C2) ^ C3
            ConstantInt *C2 = Op0CI, *C3 = RHSC;
            APInt FoldConst = C1->getValue().lshr(C2->getValue());
            FoldConst ^= C3->getValue();
            // Prepare the two operands.
            Value *Opnd0 = Builder->CreateLShr(E1->getOperand(0), C2);
            Opnd0->takeName(Op0I);
            cast<Instruction>(Opnd0)->setDebugLoc(I.getDebugLoc());
            Value *FoldVal = ConstantInt::get(Opnd0->getType(), FoldConst);

            return BinaryOperator::CreateXor(Opnd0, FoldVal);
          }
        }
      }
    }
  }

  if (isa<Constant>(Op1))
    if (Instruction *FoldedLogic = foldOpWithConstantIntoOperand(I))
      return FoldedLogic;

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
      const APInt *C;
      if (B == Op1 &&                                      // (B&A)^A == ~B & A
          !match(Op1, m_APInt(C))) {  // Canonical form is (B&C)^C
        return BinaryOperator::CreateAnd(Builder->CreateNot(A), Op1);
      }
    }
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
    // (A | ~B) ^ (~A | B) -> A ^ B
    // (~B | A) ^ (~A | B) -> A ^ B
    if (match(Op0I, m_c_Or(m_Value(A), m_Not(m_Value(B)))) &&
        match(Op1I, m_Or(m_Not(m_Specific(A)), m_Specific(B))))
      return BinaryOperator::CreateXor(A, B);

    // (~A | B) ^ (A | ~B) -> A ^ B
    if (match(Op0I, m_Or(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1I, m_Or(m_Specific(A), m_Not(m_Specific(B))))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (A & ~B) ^ (~A & B) -> A ^ B
    // (~B & A) ^ (~A & B) -> A ^ B
    if (match(Op0I, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
        match(Op1I, m_And(m_Not(m_Specific(A)), m_Specific(B))))
      return BinaryOperator::CreateXor(A, B);

    // (~A & B) ^ (A & ~B) -> A ^ B
    if (match(Op0I, m_And(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1I, m_And(m_Specific(A), m_Not(m_Specific(B))))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (A ^ C)^(A | B) -> ((~A) & B) ^ C
    if (match(Op0I, m_Xor(m_Value(D), m_Value(C))) &&
        match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (D == A)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(A), B), C);
      if (D == B)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(B), A), C);
    }
    // (A | B)^(A ^ C) -> ((~A) & B) ^ C
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Xor(m_Value(D), m_Value(C)))) {
      if (D == A)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(A), B), C);
      if (D == B)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(B), A), C);
    }
    // (A & B) ^ (A ^ B) -> (A | B)
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Xor(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateOr(A, B);
    // (A ^ B) ^ (A & B) -> (A | B)
    if (match(Op0I, m_Xor(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateOr(A, B);
  }

  // (A & ~B) ^ ~A -> ~(A & B)
  // (~B & A) ^ ~A -> ~(A & B)
  Value *A, *B;
  if (match(Op0, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_Not(m_Specific(A))))
    return BinaryOperator::CreateNot(Builder->CreateAnd(A, B));

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
          return replaceInstUsesWith(I,
                               getNewICmpValue(isSigned, Code, Op0, Op1,
                                               Builder));
        }
      }

  if (Instruction *CastedXor = foldCastedBitwiseLogic(I))
    return CastedXor;

  return Changed ? &I : nullptr;
}
