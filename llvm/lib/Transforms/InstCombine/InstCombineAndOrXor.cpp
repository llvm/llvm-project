//===- InstCombineAndOrXor.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitAnd, visitOr, and visitXor functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/CmpInstAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

/// This is the complement of getICmpCode, which turns an opcode and two
/// operands into either a constant true or false, or a brand new ICmp
/// instruction. The sign is passed in to determine which kind of predicate to
/// use in the new icmp instruction.
static Value *getNewICmpValue(unsigned Code, bool Sign, Value *LHS, Value *RHS,
                              InstCombiner::BuilderTy &Builder) {
  ICmpInst::Predicate NewPred;
  if (Constant *TorF = getPredForICmpCode(Code, Sign, LHS->getType(), NewPred))
    return TorF;
  return Builder.CreateICmp(NewPred, LHS, RHS);
}

/// This is the complement of getFCmpCode, which turns an opcode and two
/// operands into either a FCmp instruction, or a true/false constant.
static Value *getFCmpValue(unsigned Code, Value *LHS, Value *RHS,
                           InstCombiner::BuilderTy &Builder) {
  FCmpInst::Predicate NewPred;
  if (Constant *TorF = getPredForFCmpCode(Code, LHS->getType(), NewPred))
    return TorF;
  return Builder.CreateFCmp(NewPred, LHS, RHS);
}

/// Transform BITWISE_OP(BSWAP(A),BSWAP(B)) or
/// BITWISE_OP(BSWAP(A), Constant) to BSWAP(BITWISE_OP(A, B))
/// \param I Binary operator to transform.
/// \return Pointer to node that must replace the original binary operator, or
///         null pointer if no transformation was made.
static Value *SimplifyBSwap(BinaryOperator &I,
                            InstCombiner::BuilderTy &Builder) {
  assert(I.isBitwiseLogicOp() && "Unexpected opcode for bswap simplifying");

  Value *OldLHS = I.getOperand(0);
  Value *OldRHS = I.getOperand(1);

  Value *NewLHS;
  if (!match(OldLHS, m_BSwap(m_Value(NewLHS))))
    return nullptr;

  Value *NewRHS;
  const APInt *C;

  if (match(OldRHS, m_BSwap(m_Value(NewRHS)))) {
    // OP( BSWAP(x), BSWAP(y) ) -> BSWAP( OP(x, y) )
    if (!OldLHS->hasOneUse() && !OldRHS->hasOneUse())
      return nullptr;
    // NewRHS initialized by the matcher.
  } else if (match(OldRHS, m_APInt(C))) {
    // OP( BSWAP(x), CONSTANT ) -> BSWAP( OP(x, BSWAP(CONSTANT) ) )
    if (!OldLHS->hasOneUse())
      return nullptr;
    NewRHS = ConstantInt::get(I.getType(), C->byteSwap());
  } else
    return nullptr;

  Value *BinOp = Builder.CreateBinOp(I.getOpcode(), NewLHS, NewRHS);
  Function *F = Intrinsic::getDeclaration(I.getModule(), Intrinsic::bswap,
                                          I.getType());
  return Builder.CreateCall(F, BinOp);
}

/// Emit a computation of: (V >= Lo && V < Hi) if Inside is true, otherwise
/// (V < Lo || V >= Hi). This method expects that Lo < Hi. IsSigned indicates
/// whether to treat V, Lo, and Hi as signed or not.
Value *InstCombinerImpl::insertRangeTest(Value *V, const APInt &Lo,
                                         const APInt &Hi, bool isSigned,
                                         bool Inside) {
  assert((isSigned ? Lo.slt(Hi) : Lo.ult(Hi)) &&
         "Lo is not < Hi in range emission code!");

  Type *Ty = V->getType();

  // V >= Min && V <  Hi --> V <  Hi
  // V <  Min || V >= Hi --> V >= Hi
  ICmpInst::Predicate Pred = Inside ? ICmpInst::ICMP_ULT : ICmpInst::ICMP_UGE;
  if (isSigned ? Lo.isMinSignedValue() : Lo.isMinValue()) {
    Pred = isSigned ? ICmpInst::getSignedPredicate(Pred) : Pred;
    return Builder.CreateICmp(Pred, V, ConstantInt::get(Ty, Hi));
  }

  // V >= Lo && V <  Hi --> V - Lo u<  Hi - Lo
  // V <  Lo || V >= Hi --> V - Lo u>= Hi - Lo
  Value *VMinusLo =
      Builder.CreateSub(V, ConstantInt::get(Ty, Lo), V->getName() + ".off");
  Constant *HiMinusLo = ConstantInt::get(Ty, Hi - Lo);
  return Builder.CreateICmp(Pred, VMinusLo, HiMinusLo);
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
  const APInt *ConstA = nullptr, *ConstB = nullptr, *ConstC = nullptr;
  match(A, m_APInt(ConstA));
  match(B, m_APInt(ConstB));
  match(C, m_APInt(ConstC));
  bool IsEq = (Pred == ICmpInst::ICMP_EQ);
  bool IsAPow2 = ConstA && ConstA->isPowerOf2();
  bool IsBPow2 = ConstB && ConstB->isPowerOf2();
  unsigned MaskVal = 0;
  if (ConstC && ConstC->isZero()) {
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
  } else if (ConstA && ConstC && ConstC->isSubsetOf(*ConstA)) {
    MaskVal |= (IsEq ? AMask_Mixed : AMask_NotMixed);
  }

  if (B == C) {
    MaskVal |= (IsEq ? (BMask_AllOnes | BMask_Mixed)
                     : (BMask_NotAllOnes | BMask_NotMixed));
    if (IsBPow2)
      MaskVal |= (IsEq ? (Mask_NotAllZeros | BMask_NotMixed)
                       : (Mask_AllZeros | BMask_Mixed));
  } else if (ConstB && ConstC && ConstC->isSubsetOf(*ConstB)) {
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

// Adapts the external decomposeBitTestICmp for local use.
static bool decomposeBitTestICmp(Value *LHS, Value *RHS, CmpInst::Predicate &Pred,
                                 Value *&X, Value *&Y, Value *&Z) {
  APInt Mask;
  if (!llvm::decomposeBitTestICmp(LHS, RHS, Pred, X, Mask))
    return false;

  Y = ConstantInt::get(X->getType(), Mask);
  Z = ConstantInt::get(X->getType(), 0);
  return true;
}

/// Handle (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E).
/// Return the pattern classes (from MaskedICmpType) for the left hand side and
/// the right hand side as a pair.
/// LHS and RHS are the left hand side and the right hand side ICmps and PredL
/// and PredR are their predicates, respectively.
static
Optional<std::pair<unsigned, unsigned>>
getMaskedTypeForICmpPair(Value *&A, Value *&B, Value *&C,
                         Value *&D, Value *&E, ICmpInst *LHS,
                         ICmpInst *RHS,
                         ICmpInst::Predicate &PredL,
                         ICmpInst::Predicate &PredR) {
  // Don't allow pointers. Splat vectors are fine.
  if (!LHS->getOperand(0)->getType()->isIntOrIntVectorTy() ||
      !RHS->getOperand(0)->getType()->isIntOrIntVectorTy())
    return None;

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
  if (decomposeBitTestICmp(L1, L2, PredL, L11, L12, L2)) {
    L21 = L22 = L1 = nullptr;
  } else {
    // Look for ANDs in the LHS icmp.
    if (!match(L1, m_And(m_Value(L11), m_Value(L12)))) {
      // Any icmp can be viewed as being trivially masked; if it allows us to
      // remove one, it's worth it.
      L11 = L1;
      L12 = Constant::getAllOnesValue(L1->getType());
    }

    if (!match(L2, m_And(m_Value(L21), m_Value(L22)))) {
      L21 = L2;
      L22 = Constant::getAllOnesValue(L2->getType());
    }
  }

  // Bail if LHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(PredL))
    return None;

  Value *R1 = RHS->getOperand(0);
  Value *R2 = RHS->getOperand(1);
  Value *R11, *R12;
  bool Ok = false;
  if (decomposeBitTestICmp(R1, R2, PredR, R11, R12, R2)) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11;
      D = R12;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12;
      D = R11;
    } else {
      return None;
    }
    E = R2;
    R1 = nullptr;
    Ok = true;
  } else {
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
    return None;

  // Look for ANDs on the right side of the RHS icmp.
  if (!Ok) {
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
      return None;
    }

    assert(Ok && "Failed to find AND on the right side of the RHS icmp.");
  }

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
  return Optional<std::pair<unsigned, unsigned>>(std::make_pair(LeftType, RightType));
}

/// Try to fold (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E) into a single
/// (icmp(A & X) ==/!= Y), where the left-hand side is of type Mask_NotAllZeros
/// and the right hand side is of type BMask_Mixed. For example,
/// (icmp (A & 12) != 0) & (icmp (A & 15) == 8) -> (icmp (A & 15) == 8).
/// Also used for logical and/or, must be poison safe.
static Value *foldLogOpOfMaskedICmps_NotAllZeros_BMask_Mixed(
    ICmpInst *LHS, ICmpInst *RHS, bool IsAnd, Value *A, Value *B, Value *C,
    Value *D, Value *E, ICmpInst::Predicate PredL, ICmpInst::Predicate PredR,
    InstCombiner::BuilderTy &Builder) {
  // We are given the canonical form:
  //   (icmp ne (A & B), 0) & (icmp eq (A & D), E).
  // where D & E == E.
  //
  // If IsAnd is false, we get it in negated form:
  //   (icmp eq (A & B), 0) | (icmp ne (A & D), E) ->
  //      !((icmp ne (A & B), 0) & (icmp eq (A & D), E)).
  //
  // We currently handle the case of B, C, D, E are constant.
  //
  const APInt *BCst, *CCst, *DCst, *OrigECst;
  if (!match(B, m_APInt(BCst)) || !match(C, m_APInt(CCst)) ||
      !match(D, m_APInt(DCst)) || !match(E, m_APInt(OrigECst)))
    return nullptr;

  ICmpInst::Predicate NewCC = IsAnd ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE;

  // Update E to the canonical form when D is a power of two and RHS is
  // canonicalized as,
  // (icmp ne (A & D), 0) -> (icmp eq (A & D), D) or
  // (icmp ne (A & D), D) -> (icmp eq (A & D), 0).
  APInt ECst = *OrigECst;
  if (PredR != NewCC)
    ECst ^= *DCst;

  // If B or D is zero, skip because if LHS or RHS can be trivially folded by
  // other folding rules and this pattern won't apply any more.
  if (*BCst == 0 || *DCst == 0)
    return nullptr;

  // If B and D don't intersect, ie. (B & D) == 0, no folding because we can't
  // deduce anything from it.
  // For example,
  // (icmp ne (A & 12), 0) & (icmp eq (A & 3), 1) -> no folding.
  if ((*BCst & *DCst) == 0)
    return nullptr;

  // If the following two conditions are met:
  //
  // 1. mask B covers only a single bit that's not covered by mask D, that is,
  // (B & (B ^ D)) is a power of 2 (in other words, B minus the intersection of
  // B and D has only one bit set) and,
  //
  // 2. RHS (and E) indicates that the rest of B's bits are zero (in other
  // words, the intersection of B and D is zero), that is, ((B & D) & E) == 0
  //
  // then that single bit in B must be one and thus the whole expression can be
  // folded to
  //   (A & (B | D)) == (B & (B ^ D)) | E.
  //
  // For example,
  // (icmp ne (A & 12), 0) & (icmp eq (A & 7), 1) -> (icmp eq (A & 15), 9)
  // (icmp ne (A & 15), 0) & (icmp eq (A & 7), 0) -> (icmp eq (A & 15), 8)
  if ((((*BCst & *DCst) & ECst) == 0) &&
      (*BCst & (*BCst ^ *DCst)).isPowerOf2()) {
    APInt BorD = *BCst | *DCst;
    APInt BandBxorDorE = (*BCst & (*BCst ^ *DCst)) | ECst;
    Value *NewMask = ConstantInt::get(A->getType(), BorD);
    Value *NewMaskedValue = ConstantInt::get(A->getType(), BandBxorDorE);
    Value *NewAnd = Builder.CreateAnd(A, NewMask);
    return Builder.CreateICmp(NewCC, NewAnd, NewMaskedValue);
  }

  auto IsSubSetOrEqual = [](const APInt *C1, const APInt *C2) {
    return (*C1 & *C2) == *C1;
  };
  auto IsSuperSetOrEqual = [](const APInt *C1, const APInt *C2) {
    return (*C1 & *C2) == *C2;
  };

  // In the following, we consider only the cases where B is a superset of D, B
  // is a subset of D, or B == D because otherwise there's at least one bit
  // covered by B but not D, in which case we can't deduce much from it, so
  // no folding (aside from the single must-be-one bit case right above.)
  // For example,
  // (icmp ne (A & 14), 0) & (icmp eq (A & 3), 1) -> no folding.
  if (!IsSubSetOrEqual(BCst, DCst) && !IsSuperSetOrEqual(BCst, DCst))
    return nullptr;

  // At this point, either B is a superset of D, B is a subset of D or B == D.

  // If E is zero, if B is a subset of (or equal to) D, LHS and RHS contradict
  // and the whole expression becomes false (or true if negated), otherwise, no
  // folding.
  // For example,
  // (icmp ne (A & 3), 0) & (icmp eq (A & 7), 0) -> false.
  // (icmp ne (A & 15), 0) & (icmp eq (A & 3), 0) -> no folding.
  if (ECst.isZero()) {
    if (IsSubSetOrEqual(BCst, DCst))
      return ConstantInt::get(LHS->getType(), !IsAnd);
    return nullptr;
  }

  // At this point, B, D, E aren't zero and (B & D) == B, (B & D) == D or B ==
  // D. If B is a superset of (or equal to) D, since E is not zero, LHS is
  // subsumed by RHS (RHS implies LHS.) So the whole expression becomes
  // RHS. For example,
  // (icmp ne (A & 255), 0) & (icmp eq (A & 15), 8) -> (icmp eq (A & 15), 8).
  // (icmp ne (A & 15), 0) & (icmp eq (A & 15), 8) -> (icmp eq (A & 15), 8).
  if (IsSuperSetOrEqual(BCst, DCst))
    return RHS;
  // Otherwise, B is a subset of D. If B and E have a common bit set,
  // ie. (B & E) != 0, then LHS is subsumed by RHS. For example.
  // (icmp ne (A & 12), 0) & (icmp eq (A & 15), 8) -> (icmp eq (A & 15), 8).
  assert(IsSubSetOrEqual(BCst, DCst) && "Precondition due to above code");
  if ((*BCst & ECst) != 0)
    return RHS;
  // Otherwise, LHS and RHS contradict and the whole expression becomes false
  // (or true if negated.) For example,
  // (icmp ne (A & 7), 0) & (icmp eq (A & 15), 8) -> false.
  // (icmp ne (A & 6), 0) & (icmp eq (A & 15), 8) -> false.
  return ConstantInt::get(LHS->getType(), !IsAnd);
}

/// Try to fold (icmp(A & B) ==/!= 0) &/| (icmp(A & D) ==/!= E) into a single
/// (icmp(A & X) ==/!= Y), where the left-hand side and the right hand side
/// aren't of the common mask pattern type.
/// Also used for logical and/or, must be poison safe.
static Value *foldLogOpOfMaskedICmpsAsymmetric(
    ICmpInst *LHS, ICmpInst *RHS, bool IsAnd, Value *A, Value *B, Value *C,
    Value *D, Value *E, ICmpInst::Predicate PredL, ICmpInst::Predicate PredR,
    unsigned LHSMask, unsigned RHSMask, InstCombiner::BuilderTy &Builder) {
  assert(ICmpInst::isEquality(PredL) && ICmpInst::isEquality(PredR) &&
         "Expected equality predicates for masked type of icmps.");
  // Handle Mask_NotAllZeros-BMask_Mixed cases.
  // (icmp ne/eq (A & B), C) &/| (icmp eq/ne (A & D), E), or
  // (icmp eq/ne (A & B), C) &/| (icmp ne/eq (A & D), E)
  //    which gets swapped to
  //    (icmp ne/eq (A & D), E) &/| (icmp eq/ne (A & B), C).
  if (!IsAnd) {
    LHSMask = conjugateICmpMask(LHSMask);
    RHSMask = conjugateICmpMask(RHSMask);
  }
  if ((LHSMask & Mask_NotAllZeros) && (RHSMask & BMask_Mixed)) {
    if (Value *V = foldLogOpOfMaskedICmps_NotAllZeros_BMask_Mixed(
            LHS, RHS, IsAnd, A, B, C, D, E,
            PredL, PredR, Builder)) {
      return V;
    }
  } else if ((LHSMask & BMask_Mixed) && (RHSMask & Mask_NotAllZeros)) {
    if (Value *V = foldLogOpOfMaskedICmps_NotAllZeros_BMask_Mixed(
            RHS, LHS, IsAnd, A, D, E, B, C,
            PredR, PredL, Builder)) {
      return V;
    }
  }
  return nullptr;
}

/// Try to fold (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// into a single (icmp(A & X) ==/!= Y).
static Value *foldLogOpOfMaskedICmps(ICmpInst *LHS, ICmpInst *RHS, bool IsAnd,
                                     bool IsLogical,
                                     InstCombiner::BuilderTy &Builder) {
  Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr, *E = nullptr;
  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();
  Optional<std::pair<unsigned, unsigned>> MaskPair =
      getMaskedTypeForICmpPair(A, B, C, D, E, LHS, RHS, PredL, PredR);
  if (!MaskPair)
    return nullptr;
  assert(ICmpInst::isEquality(PredL) && ICmpInst::isEquality(PredR) &&
         "Expected equality predicates for masked type of icmps.");
  unsigned LHSMask = MaskPair->first;
  unsigned RHSMask = MaskPair->second;
  unsigned Mask = LHSMask & RHSMask;
  if (Mask == 0) {
    // Even if the two sides don't share a common pattern, check if folding can
    // still happen.
    if (Value *V = foldLogOpOfMaskedICmpsAsymmetric(
            LHS, RHS, IsAnd, A, B, C, D, E, PredL, PredR, LHSMask, RHSMask,
            Builder))
      return V;
    return nullptr;
  }

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
    if (IsLogical && !isGuaranteedNotToBeUndefOrPoison(D))
      return nullptr; // TODO: Use freeze?
    Value *NewOr = Builder.CreateOr(B, D);
    Value *NewAnd = Builder.CreateAnd(A, NewOr);
    // We can't use C as zero because we might actually handle
    //   (icmp ne (A & B), B) & (icmp ne (A & D), D)
    // with B and D, having a single bit set.
    Value *Zero = Constant::getNullValue(A->getType());
    return Builder.CreateICmp(NewCC, NewAnd, Zero);
  }
  if (Mask & BMask_AllOnes) {
    // (icmp eq (A & B), B) & (icmp eq (A & D), D)
    // -> (icmp eq (A & (B|D)), (B|D))
    if (IsLogical && !isGuaranteedNotToBeUndefOrPoison(D))
      return nullptr; // TODO: Use freeze?
    Value *NewOr = Builder.CreateOr(B, D);
    Value *NewAnd = Builder.CreateAnd(A, NewOr);
    return Builder.CreateICmp(NewCC, NewAnd, NewOr);
  }
  if (Mask & AMask_AllOnes) {
    // (icmp eq (A & B), A) & (icmp eq (A & D), A)
    // -> (icmp eq (A & (B&D)), A)
    if (IsLogical && !isGuaranteedNotToBeUndefOrPoison(D))
      return nullptr; // TODO: Use freeze?
    Value *NewAnd1 = Builder.CreateAnd(B, D);
    Value *NewAnd2 = Builder.CreateAnd(A, NewAnd1);
    return Builder.CreateICmp(NewCC, NewAnd2, A);
  }

  // Remaining cases assume at least that B and D are constant, and depend on
  // their actual values. This isn't strictly necessary, just a "handle the
  // easy cases for now" decision.
  const APInt *ConstB, *ConstD;
  if (!match(B, m_APInt(ConstB)) || !match(D, m_APInt(ConstD)))
    return nullptr;

  if (Mask & (Mask_NotAllZeros | BMask_NotAllOnes)) {
    // (icmp ne (A & B), 0) & (icmp ne (A & D), 0) and
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), 0) or (icmp ne (A & D), 0)
    // Only valid if one of the masks is a superset of the other (check "B&D" is
    // the same as either B or D).
    APInt NewMask = *ConstB & *ConstD;
    if (NewMask == *ConstB)
      return LHS;
    else if (NewMask == *ConstD)
      return RHS;
  }

  if (Mask & AMask_NotAllOnes) {
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), A) or (icmp ne (A & D), A)
    // Only valid if one of the masks is a superset of the other (check "B|D" is
    // the same as either B or D).
    APInt NewMask = *ConstB | *ConstD;
    if (NewMask == *ConstB)
      return LHS;
    else if (NewMask == *ConstD)
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
    const APInt *OldConstC, *OldConstE;
    if (!match(C, m_APInt(OldConstC)) || !match(E, m_APInt(OldConstE)))
      return nullptr;

    const APInt ConstC = PredL != NewCC ? *ConstB ^ *OldConstC : *OldConstC;
    const APInt ConstE = PredR != NewCC ? *ConstD ^ *OldConstE : *OldConstE;

    // If there is a conflict, we should actually return a false for the
    // whole construct.
    if (((*ConstB & *ConstD) & (ConstC ^ ConstE)).getBoolValue())
      return ConstantInt::get(LHS->getType(), !IsAnd);

    Value *NewOr1 = Builder.CreateOr(B, D);
    Value *NewAnd = Builder.CreateAnd(A, NewOr1);
    Constant *NewOr2 = ConstantInt::get(A->getType(), ConstC | ConstE);
    return Builder.CreateICmp(NewCC, NewAnd, NewOr2);
  }

  return nullptr;
}

/// Try to fold a signed range checked with lower bound 0 to an unsigned icmp.
/// Example: (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
/// If \p Inverted is true then the check is for the inverted range, e.g.
/// (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
Value *InstCombinerImpl::simplifyRangeCheck(ICmpInst *Cmp0, ICmpInst *Cmp1,
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
  KnownBits Known = computeKnownBits(RangeEnd, /*Depth=*/0, Cmp1);
  if (!Known.isNonNegative())
    return nullptr;

  if (Inverted)
    NewPred = ICmpInst::getInversePredicate(NewPred);

  return Builder.CreateICmp(NewPred, Input, RangeEnd);
}

// Fold (iszero(A & K1) | iszero(A & K2)) -> (A & (K1 | K2)) != (K1 | K2)
// Fold (!iszero(A & K1) & !iszero(A & K2)) -> (A & (K1 | K2)) == (K1 | K2)
Value *InstCombinerImpl::foldAndOrOfICmpsOfAndWithPow2(ICmpInst *LHS,
                                                       ICmpInst *RHS,
                                                       Instruction *CxtI,
                                                       bool IsAnd,
                                                       bool IsLogical) {
  CmpInst::Predicate Pred = IsAnd ? CmpInst::ICMP_NE : CmpInst::ICMP_EQ;
  if (LHS->getPredicate() != Pred || RHS->getPredicate() != Pred)
    return nullptr;

  if (!match(LHS->getOperand(1), m_Zero()) ||
      !match(RHS->getOperand(1), m_Zero()))
    return nullptr;

  Value *L1, *L2, *R1, *R2;
  if (match(LHS->getOperand(0), m_And(m_Value(L1), m_Value(L2))) &&
      match(RHS->getOperand(0), m_And(m_Value(R1), m_Value(R2)))) {
    if (L1 == R2 || L2 == R2)
      std::swap(R1, R2);
    if (L2 == R1)
      std::swap(L1, L2);

    if (L1 == R1 &&
        isKnownToBeAPowerOfTwo(L2, false, 0, CxtI) &&
        isKnownToBeAPowerOfTwo(R2, false, 0, CxtI)) {
      // If this is a logical and/or, then we must prevent propagation of a
      // poison value from the RHS by inserting freeze.
      if (IsLogical)
        R2 = Builder.CreateFreeze(R2);
      Value *Mask = Builder.CreateOr(L2, R2);
      Value *Masked = Builder.CreateAnd(L1, Mask);
      auto NewPred = IsAnd ? CmpInst::ICMP_EQ : CmpInst::ICMP_NE;
      return Builder.CreateICmp(NewPred, Masked, Mask);
    }
  }

  return nullptr;
}

/// General pattern:
///   X & Y
///
/// Where Y is checking that all the high bits (covered by a mask 4294967168)
/// are uniform, i.e.  %arg & 4294967168  can be either  4294967168  or  0
/// Pattern can be one of:
///   %t = add        i32 %arg,    128
///   %r = icmp   ult i32 %t,      256
/// Or
///   %t0 = shl       i32 %arg,    24
///   %t1 = ashr      i32 %t0,     24
///   %r  = icmp  eq  i32 %t1,     %arg
/// Or
///   %t0 = trunc     i32 %arg  to i8
///   %t1 = sext      i8  %t0   to i32
///   %r  = icmp  eq  i32 %t1,     %arg
/// This pattern is a signed truncation check.
///
/// And X is checking that some bit in that same mask is zero.
/// I.e. can be one of:
///   %r = icmp sgt i32   %arg,    -1
/// Or
///   %t = and      i32   %arg,    2147483648
///   %r = icmp eq  i32   %t,      0
///
/// Since we are checking that all the bits in that mask are the same,
/// and a particular bit is zero, what we are really checking is that all the
/// masked bits are zero.
/// So this should be transformed to:
///   %r = icmp ult i32 %arg, 128
static Value *foldSignedTruncationCheck(ICmpInst *ICmp0, ICmpInst *ICmp1,
                                        Instruction &CxtI,
                                        InstCombiner::BuilderTy &Builder) {
  assert(CxtI.getOpcode() == Instruction::And);

  // Match  icmp ult (add %arg, C01), C1   (C1 == C01 << 1; powers of two)
  auto tryToMatchSignedTruncationCheck = [](ICmpInst *ICmp, Value *&X,
                                            APInt &SignBitMask) -> bool {
    CmpInst::Predicate Pred;
    const APInt *I01, *I1; // powers of two; I1 == I01 << 1
    if (!(match(ICmp,
                m_ICmp(Pred, m_Add(m_Value(X), m_Power2(I01)), m_Power2(I1))) &&
          Pred == ICmpInst::ICMP_ULT && I1->ugt(*I01) && I01->shl(1) == *I1))
      return false;
    // Which bit is the new sign bit as per the 'signed truncation' pattern?
    SignBitMask = *I01;
    return true;
  };

  // One icmp needs to be 'signed truncation check'.
  // We need to match this first, else we will mismatch commutative cases.
  Value *X1;
  APInt HighestBit;
  ICmpInst *OtherICmp;
  if (tryToMatchSignedTruncationCheck(ICmp1, X1, HighestBit))
    OtherICmp = ICmp0;
  else if (tryToMatchSignedTruncationCheck(ICmp0, X1, HighestBit))
    OtherICmp = ICmp1;
  else
    return nullptr;

  assert(HighestBit.isPowerOf2() && "expected to be power of two (non-zero)");

  // Try to match/decompose into:  icmp eq (X & Mask), 0
  auto tryToDecompose = [](ICmpInst *ICmp, Value *&X,
                           APInt &UnsetBitsMask) -> bool {
    CmpInst::Predicate Pred = ICmp->getPredicate();
    // Can it be decomposed into  icmp eq (X & Mask), 0  ?
    if (llvm::decomposeBitTestICmp(ICmp->getOperand(0), ICmp->getOperand(1),
                                   Pred, X, UnsetBitsMask,
                                   /*LookThroughTrunc=*/false) &&
        Pred == ICmpInst::ICMP_EQ)
      return true;
    // Is it  icmp eq (X & Mask), 0  already?
    const APInt *Mask;
    if (match(ICmp, m_ICmp(Pred, m_And(m_Value(X), m_APInt(Mask)), m_Zero())) &&
        Pred == ICmpInst::ICMP_EQ) {
      UnsetBitsMask = *Mask;
      return true;
    }
    return false;
  };

  // And the other icmp needs to be decomposable into a bit test.
  Value *X0;
  APInt UnsetBitsMask;
  if (!tryToDecompose(OtherICmp, X0, UnsetBitsMask))
    return nullptr;

  assert(!UnsetBitsMask.isZero() && "empty mask makes no sense.");

  // Are they working on the same value?
  Value *X;
  if (X1 == X0) {
    // Ok as is.
    X = X1;
  } else if (match(X0, m_Trunc(m_Specific(X1)))) {
    UnsetBitsMask = UnsetBitsMask.zext(X1->getType()->getScalarSizeInBits());
    X = X1;
  } else
    return nullptr;

  // So which bits should be uniform as per the 'signed truncation check'?
  // (all the bits starting with (i.e. including) HighestBit)
  APInt SignBitsMask = ~(HighestBit - 1U);

  // UnsetBitsMask must have some common bits with SignBitsMask,
  if (!UnsetBitsMask.intersects(SignBitsMask))
    return nullptr;

  // Does UnsetBitsMask contain any bits outside of SignBitsMask?
  if (!UnsetBitsMask.isSubsetOf(SignBitsMask)) {
    APInt OtherHighestBit = (~UnsetBitsMask) + 1U;
    if (!OtherHighestBit.isPowerOf2())
      return nullptr;
    HighestBit = APIntOps::umin(HighestBit, OtherHighestBit);
  }
  // Else, if it does not, then all is ok as-is.

  // %r = icmp ult %X, SignBit
  return Builder.CreateICmpULT(X, ConstantInt::get(X->getType(), HighestBit),
                               CxtI.getName() + ".simplified");
}

/// Fold (icmp eq ctpop(X) 1) | (icmp eq X 0) into (icmp ult ctpop(X) 2) and
/// fold (icmp ne ctpop(X) 1) & (icmp ne X 0) into (icmp ugt ctpop(X) 1).
/// Also used for logical and/or, must be poison safe.
static Value *foldIsPowerOf2OrZero(ICmpInst *Cmp0, ICmpInst *Cmp1, bool IsAnd,
                                   InstCombiner::BuilderTy &Builder) {
  CmpInst::Predicate Pred0, Pred1;
  Value *X;
  if (!match(Cmp0, m_ICmp(Pred0, m_Intrinsic<Intrinsic::ctpop>(m_Value(X)),
                          m_SpecificInt(1))) ||
      !match(Cmp1, m_ICmp(Pred1, m_Specific(X), m_ZeroInt())))
    return nullptr;

  Value *CtPop = Cmp0->getOperand(0);
  if (IsAnd && Pred0 == ICmpInst::ICMP_NE && Pred1 == ICmpInst::ICMP_NE)
    return Builder.CreateICmpUGT(CtPop, ConstantInt::get(CtPop->getType(), 1));
  if (!IsAnd && Pred0 == ICmpInst::ICMP_EQ && Pred1 == ICmpInst::ICMP_EQ)
    return Builder.CreateICmpULT(CtPop, ConstantInt::get(CtPop->getType(), 2));

  return nullptr;
}

/// Reduce a pair of compares that check if a value has exactly 1 bit set.
/// Also used for logical and/or, must be poison safe.
static Value *foldIsPowerOf2(ICmpInst *Cmp0, ICmpInst *Cmp1, bool JoinedByAnd,
                             InstCombiner::BuilderTy &Builder) {
  // Handle 'and' / 'or' commutation: make the equality check the first operand.
  if (JoinedByAnd && Cmp1->getPredicate() == ICmpInst::ICMP_NE)
    std::swap(Cmp0, Cmp1);
  else if (!JoinedByAnd && Cmp1->getPredicate() == ICmpInst::ICMP_EQ)
    std::swap(Cmp0, Cmp1);

  // (X != 0) && (ctpop(X) u< 2) --> ctpop(X) == 1
  CmpInst::Predicate Pred0, Pred1;
  Value *X;
  if (JoinedByAnd && match(Cmp0, m_ICmp(Pred0, m_Value(X), m_ZeroInt())) &&
      match(Cmp1, m_ICmp(Pred1, m_Intrinsic<Intrinsic::ctpop>(m_Specific(X)),
                         m_SpecificInt(2))) &&
      Pred0 == ICmpInst::ICMP_NE && Pred1 == ICmpInst::ICMP_ULT) {
    Value *CtPop = Cmp1->getOperand(0);
    return Builder.CreateICmpEQ(CtPop, ConstantInt::get(CtPop->getType(), 1));
  }
  // (X == 0) || (ctpop(X) u> 1) --> ctpop(X) != 1
  if (!JoinedByAnd && match(Cmp0, m_ICmp(Pred0, m_Value(X), m_ZeroInt())) &&
      match(Cmp1, m_ICmp(Pred1, m_Intrinsic<Intrinsic::ctpop>(m_Specific(X)),
                         m_SpecificInt(1))) &&
      Pred0 == ICmpInst::ICMP_EQ && Pred1 == ICmpInst::ICMP_UGT) {
    Value *CtPop = Cmp1->getOperand(0);
    return Builder.CreateICmpNE(CtPop, ConstantInt::get(CtPop->getType(), 1));
  }
  return nullptr;
}

/// Commuted variants are assumed to be handled by calling this function again
/// with the parameters swapped.
static Value *foldUnsignedUnderflowCheck(ICmpInst *ZeroICmp,
                                         ICmpInst *UnsignedICmp, bool IsAnd,
                                         const SimplifyQuery &Q,
                                         InstCombiner::BuilderTy &Builder) {
  Value *ZeroCmpOp;
  ICmpInst::Predicate EqPred;
  if (!match(ZeroICmp, m_ICmp(EqPred, m_Value(ZeroCmpOp), m_Zero())) ||
      !ICmpInst::isEquality(EqPred))
    return nullptr;

  auto IsKnownNonZero = [&](Value *V) {
    return isKnownNonZero(V, Q.DL, /*Depth=*/0, Q.AC, Q.CxtI, Q.DT);
  };

  ICmpInst::Predicate UnsignedPred;

  Value *A, *B;
  if (match(UnsignedICmp,
            m_c_ICmp(UnsignedPred, m_Specific(ZeroCmpOp), m_Value(A))) &&
      match(ZeroCmpOp, m_c_Add(m_Specific(A), m_Value(B))) &&
      (ZeroICmp->hasOneUse() || UnsignedICmp->hasOneUse())) {
    auto GetKnownNonZeroAndOther = [&](Value *&NonZero, Value *&Other) {
      if (!IsKnownNonZero(NonZero))
        std::swap(NonZero, Other);
      return IsKnownNonZero(NonZero);
    };

    // Given  ZeroCmpOp = (A + B)
    //   ZeroCmpOp <  A && ZeroCmpOp != 0  -->  (0-X) <  Y  iff
    //   ZeroCmpOp >= A || ZeroCmpOp == 0  -->  (0-X) >= Y  iff
    //     with X being the value (A/B) that is known to be non-zero,
    //     and Y being remaining value.
    if (UnsignedPred == ICmpInst::ICMP_ULT && EqPred == ICmpInst::ICMP_NE &&
        IsAnd && GetKnownNonZeroAndOther(B, A))
      return Builder.CreateICmpULT(Builder.CreateNeg(B), A);
    if (UnsignedPred == ICmpInst::ICMP_UGE && EqPred == ICmpInst::ICMP_EQ &&
        !IsAnd && GetKnownNonZeroAndOther(B, A))
      return Builder.CreateICmpUGE(Builder.CreateNeg(B), A);
  }

  Value *Base, *Offset;
  if (!match(ZeroCmpOp, m_Sub(m_Value(Base), m_Value(Offset))))
    return nullptr;

  if (!match(UnsignedICmp,
             m_c_ICmp(UnsignedPred, m_Specific(Base), m_Specific(Offset))) ||
      !ICmpInst::isUnsigned(UnsignedPred))
    return nullptr;

  // Base >=/> Offset && (Base - Offset) != 0  <-->  Base > Offset
  // (no overflow and not null)
  if ((UnsignedPred == ICmpInst::ICMP_UGE ||
       UnsignedPred == ICmpInst::ICMP_UGT) &&
      EqPred == ICmpInst::ICMP_NE && IsAnd)
    return Builder.CreateICmpUGT(Base, Offset);

  // Base <=/< Offset || (Base - Offset) == 0  <-->  Base <= Offset
  // (overflow or null)
  if ((UnsignedPred == ICmpInst::ICMP_ULE ||
       UnsignedPred == ICmpInst::ICMP_ULT) &&
      EqPred == ICmpInst::ICMP_EQ && !IsAnd)
    return Builder.CreateICmpULE(Base, Offset);

  // Base <= Offset && (Base - Offset) != 0  -->  Base < Offset
  if (UnsignedPred == ICmpInst::ICMP_ULE && EqPred == ICmpInst::ICMP_NE &&
      IsAnd)
    return Builder.CreateICmpULT(Base, Offset);

  // Base > Offset || (Base - Offset) == 0  -->  Base >= Offset
  if (UnsignedPred == ICmpInst::ICMP_UGT && EqPred == ICmpInst::ICMP_EQ &&
      !IsAnd)
    return Builder.CreateICmpUGE(Base, Offset);

  return nullptr;
}

struct IntPart {
  Value *From;
  unsigned StartBit;
  unsigned NumBits;
};

/// Match an extraction of bits from an integer.
static Optional<IntPart> matchIntPart(Value *V) {
  Value *X;
  if (!match(V, m_OneUse(m_Trunc(m_Value(X)))))
    return None;

  unsigned NumOriginalBits = X->getType()->getScalarSizeInBits();
  unsigned NumExtractedBits = V->getType()->getScalarSizeInBits();
  Value *Y;
  const APInt *Shift;
  // For a trunc(lshr Y, Shift) pattern, make sure we're only extracting bits
  // from Y, not any shifted-in zeroes.
  if (match(X, m_OneUse(m_LShr(m_Value(Y), m_APInt(Shift)))) &&
      Shift->ule(NumOriginalBits - NumExtractedBits))
    return {{Y, (unsigned)Shift->getZExtValue(), NumExtractedBits}};
  return {{X, 0, NumExtractedBits}};
}

/// Materialize an extraction of bits from an integer in IR.
static Value *extractIntPart(const IntPart &P, IRBuilderBase &Builder) {
  Value *V = P.From;
  if (P.StartBit)
    V = Builder.CreateLShr(V, P.StartBit);
  Type *TruncTy = V->getType()->getWithNewBitWidth(P.NumBits);
  if (TruncTy != V->getType())
    V = Builder.CreateTrunc(V, TruncTy);
  return V;
}

/// (icmp eq X0, Y0) & (icmp eq X1, Y1) -> icmp eq X01, Y01
/// (icmp ne X0, Y0) | (icmp ne X1, Y1) -> icmp ne X01, Y01
/// where X0, X1 and Y0, Y1 are adjacent parts extracted from an integer.
Value *InstCombinerImpl::foldEqOfParts(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                       bool IsAnd) {
  if (!Cmp0->hasOneUse() || !Cmp1->hasOneUse())
    return nullptr;

  CmpInst::Predicate Pred = IsAnd ? CmpInst::ICMP_EQ : CmpInst::ICMP_NE;
  if (Cmp0->getPredicate() != Pred || Cmp1->getPredicate() != Pred)
    return nullptr;

  Optional<IntPart> L0 = matchIntPart(Cmp0->getOperand(0));
  Optional<IntPart> R0 = matchIntPart(Cmp0->getOperand(1));
  Optional<IntPart> L1 = matchIntPart(Cmp1->getOperand(0));
  Optional<IntPart> R1 = matchIntPart(Cmp1->getOperand(1));
  if (!L0 || !R0 || !L1 || !R1)
    return nullptr;

  // Make sure the LHS/RHS compare a part of the same value, possibly after
  // an operand swap.
  if (L0->From != L1->From || R0->From != R1->From) {
    if (L0->From != R1->From || R0->From != L1->From)
      return nullptr;
    std::swap(L1, R1);
  }

  // Make sure the extracted parts are adjacent, canonicalizing to L0/R0 being
  // the low part and L1/R1 being the high part.
  if (L0->StartBit + L0->NumBits != L1->StartBit ||
      R0->StartBit + R0->NumBits != R1->StartBit) {
    if (L1->StartBit + L1->NumBits != L0->StartBit ||
        R1->StartBit + R1->NumBits != R0->StartBit)
      return nullptr;
    std::swap(L0, L1);
    std::swap(R0, R1);
  }

  // We can simplify to a comparison of these larger parts of the integers.
  IntPart L = {L0->From, L0->StartBit, L0->NumBits + L1->NumBits};
  IntPart R = {R0->From, R0->StartBit, R0->NumBits + R1->NumBits};
  Value *LValue = extractIntPart(L, Builder);
  Value *RValue = extractIntPart(R, Builder);
  return Builder.CreateICmp(Pred, LValue, RValue);
}

/// Reduce logic-of-compares with equality to a constant by substituting a
/// common operand with the constant. Callers are expected to call this with
/// Cmp0/Cmp1 switched to handle logic op commutativity.
static Value *foldAndOrOfICmpsWithConstEq(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                          bool IsAnd,
                                          InstCombiner::BuilderTy &Builder,
                                          const SimplifyQuery &Q) {
  // Match an equality compare with a non-poison constant as Cmp0.
  // Also, give up if the compare can be constant-folded to avoid looping.
  ICmpInst::Predicate Pred0;
  Value *X;
  Constant *C;
  if (!match(Cmp0, m_ICmp(Pred0, m_Value(X), m_Constant(C))) ||
      !isGuaranteedNotToBeUndefOrPoison(C) || isa<Constant>(X))
    return nullptr;
  if ((IsAnd && Pred0 != ICmpInst::ICMP_EQ) ||
      (!IsAnd && Pred0 != ICmpInst::ICMP_NE))
    return nullptr;

  // The other compare must include a common operand (X). Canonicalize the
  // common operand as operand 1 (Pred1 is swapped if the common operand was
  // operand 0).
  Value *Y;
  ICmpInst::Predicate Pred1;
  if (!match(Cmp1, m_c_ICmp(Pred1, m_Value(Y), m_Deferred(X))))
    return nullptr;

  // Replace variable with constant value equivalence to remove a variable use:
  // (X == C) && (Y Pred1 X) --> (X == C) && (Y Pred1 C)
  // (X != C) || (Y Pred1 X) --> (X != C) || (Y Pred1 C)
  // Can think of the 'or' substitution with the 'and' bool equivalent:
  // A || B --> A || (!A && B)
  Value *SubstituteCmp = SimplifyICmpInst(Pred1, Y, C, Q);
  if (!SubstituteCmp) {
    // If we need to create a new instruction, require that the old compare can
    // be removed.
    if (!Cmp1->hasOneUse())
      return nullptr;
    SubstituteCmp = Builder.CreateICmp(Pred1, Y, C);
  }
  return Builder.CreateBinOp(IsAnd ? Instruction::And : Instruction::Or, Cmp0,
                             SubstituteCmp);
}

/// Fold (icmp Pred1 V1, C1) & (icmp Pred2 V2, C2)
/// or   (icmp Pred1 V1, C1) | (icmp Pred2 V2, C2)
/// into a single comparison using range-based reasoning.
/// NOTE: This is also used for logical and/or, must be poison-safe!
Value *InstCombinerImpl::foldAndOrOfICmpsUsingRanges(ICmpInst *ICmp1,
                                                     ICmpInst *ICmp2,
                                                     bool IsAnd) {
  ICmpInst::Predicate Pred1, Pred2;
  Value *V1, *V2;
  const APInt *C1, *C2;
  if (!match(ICmp1, m_ICmp(Pred1, m_Value(V1), m_APInt(C1))) ||
      !match(ICmp2, m_ICmp(Pred2, m_Value(V2), m_APInt(C2))))
    return nullptr;

  // Look through add of a constant offset on V1, V2, or both operands. This
  // allows us to interpret the V + C' < C'' range idiom into a proper range.
  const APInt *Offset1 = nullptr, *Offset2 = nullptr;
  if (V1 != V2) {
    Value *X;
    if (match(V1, m_Add(m_Value(X), m_APInt(Offset1))))
      V1 = X;
    if (match(V2, m_Add(m_Value(X), m_APInt(Offset2))))
      V2 = X;
  }

  if (V1 != V2)
    return nullptr;

  ConstantRange CR1 = ConstantRange::makeExactICmpRegion(
      IsAnd ? ICmpInst::getInversePredicate(Pred1) : Pred1, *C1);
  if (Offset1)
    CR1 = CR1.subtract(*Offset1);

  ConstantRange CR2 = ConstantRange::makeExactICmpRegion(
      IsAnd ? ICmpInst::getInversePredicate(Pred2) : Pred2, *C2);
  if (Offset2)
    CR2 = CR2.subtract(*Offset2);

  Type *Ty = V1->getType();
  Value *NewV = V1;
  Optional<ConstantRange> CR = CR1.exactUnionWith(CR2);
  if (!CR) {
    if (!(ICmp1->hasOneUse() && ICmp2->hasOneUse()) || CR1.isWrappedSet() ||
        CR2.isWrappedSet())
      return nullptr;

    // Check whether we have equal-size ranges that only differ by one bit.
    // In that case we can apply a mask to map one range onto the other.
    APInt LowerDiff = CR1.getLower() ^ CR2.getLower();
    APInt UpperDiff = (CR1.getUpper() - 1) ^ (CR2.getUpper() - 1);
    APInt CR1Size = CR1.getUpper() - CR1.getLower();
    if (!LowerDiff.isPowerOf2() || LowerDiff != UpperDiff ||
        CR1Size != CR2.getUpper() - CR2.getLower())
      return nullptr;

    CR = CR1.getLower().ult(CR2.getLower()) ? CR1 : CR2;
    NewV = Builder.CreateAnd(NewV, ConstantInt::get(Ty, ~LowerDiff));
  }

  if (IsAnd)
    CR = CR->inverse();

  CmpInst::Predicate NewPred;
  APInt NewC, Offset;
  CR->getEquivalentICmp(NewPred, NewC, Offset);

  if (Offset != 0)
    NewV = Builder.CreateAdd(NewV, ConstantInt::get(Ty, Offset));
  return Builder.CreateICmp(NewPred, NewV, ConstantInt::get(Ty, NewC));
}

Value *InstCombinerImpl::foldLogicOfFCmps(FCmpInst *LHS, FCmpInst *RHS,
                                          bool IsAnd, bool IsLogicalSelect) {
  Value *LHS0 = LHS->getOperand(0), *LHS1 = LHS->getOperand(1);
  Value *RHS0 = RHS->getOperand(0), *RHS1 = RHS->getOperand(1);
  FCmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();

  if (LHS0 == RHS1 && RHS0 == LHS1) {
    // Swap RHS operands to match LHS.
    PredR = FCmpInst::getSwappedPredicate(PredR);
    std::swap(RHS0, RHS1);
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
  //
  // Since (R & CC0) and (R & CC1) are either R or 0, we actually have this:
  //    bool(R & CC0) || bool(R & CC1)
  //  = bool((R & CC0) | (R & CC1))
  //  = bool(R & (CC0 | CC1)) <= by reversed distribution (contribution? ;)
  if (LHS0 == RHS0 && LHS1 == RHS1) {
    unsigned FCmpCodeL = getFCmpCode(PredL);
    unsigned FCmpCodeR = getFCmpCode(PredR);
    unsigned NewPred = IsAnd ? FCmpCodeL & FCmpCodeR : FCmpCodeL | FCmpCodeR;

    // Intersect the fast math flags.
    // TODO: We can union the fast math flags unless this is a logical select.
    IRBuilder<>::FastMathFlagGuard FMFG(Builder);
    FastMathFlags FMF = LHS->getFastMathFlags();
    FMF &= RHS->getFastMathFlags();
    Builder.setFastMathFlags(FMF);

    return getFCmpValue(NewPred, LHS0, LHS1, Builder);
  }

  // This transform is not valid for a logical select.
  if (!IsLogicalSelect &&
      ((PredL == FCmpInst::FCMP_ORD && PredR == FCmpInst::FCMP_ORD && IsAnd) ||
       (PredL == FCmpInst::FCMP_UNO && PredR == FCmpInst::FCMP_UNO &&
        !IsAnd))) {
    if (LHS0->getType() != RHS0->getType())
      return nullptr;

    // FCmp canonicalization ensures that (fcmp ord/uno X, X) and
    // (fcmp ord/uno X, C) will be transformed to (fcmp X, +0.0).
    if (match(LHS1, m_PosZeroFP()) && match(RHS1, m_PosZeroFP()))
      // Ignore the constants because they are obviously not NANs:
      // (fcmp ord x, 0.0) & (fcmp ord y, 0.0)  -> (fcmp ord x, y)
      // (fcmp uno x, 0.0) | (fcmp uno y, 0.0)  -> (fcmp uno x, y)
      return Builder.CreateFCmp(PredL, LHS0, RHS0);
  }

  return nullptr;
}

/// This a limited reassociation for a special case (see above) where we are
/// checking if two values are either both NAN (unordered) or not-NAN (ordered).
/// This could be handled more generally in '-reassociation', but it seems like
/// an unlikely pattern for a large number of logic ops and fcmps.
static Instruction *reassociateFCmps(BinaryOperator &BO,
                                     InstCombiner::BuilderTy &Builder) {
  Instruction::BinaryOps Opcode = BO.getOpcode();
  assert((Opcode == Instruction::And || Opcode == Instruction::Or) &&
         "Expecting and/or op for fcmp transform");

  // There are 4 commuted variants of the pattern. Canonicalize operands of this
  // logic op so an fcmp is operand 0 and a matching logic op is operand 1.
  Value *Op0 = BO.getOperand(0), *Op1 = BO.getOperand(1), *X;
  FCmpInst::Predicate Pred;
  if (match(Op1, m_FCmp(Pred, m_Value(), m_AnyZeroFP())))
    std::swap(Op0, Op1);

  // Match inner binop and the predicate for combining 2 NAN checks into 1.
  Value *BO10, *BO11;
  FCmpInst::Predicate NanPred = Opcode == Instruction::And ? FCmpInst::FCMP_ORD
                                                           : FCmpInst::FCMP_UNO;
  if (!match(Op0, m_FCmp(Pred, m_Value(X), m_AnyZeroFP())) || Pred != NanPred ||
      !match(Op1, m_BinOp(Opcode, m_Value(BO10), m_Value(BO11))))
    return nullptr;

  // The inner logic op must have a matching fcmp operand.
  Value *Y;
  if (!match(BO10, m_FCmp(Pred, m_Value(Y), m_AnyZeroFP())) ||
      Pred != NanPred || X->getType() != Y->getType())
    std::swap(BO10, BO11);

  if (!match(BO10, m_FCmp(Pred, m_Value(Y), m_AnyZeroFP())) ||
      Pred != NanPred || X->getType() != Y->getType())
    return nullptr;

  // and (fcmp ord X, 0), (and (fcmp ord Y, 0), Z) --> and (fcmp ord X, Y), Z
  // or  (fcmp uno X, 0), (or  (fcmp uno Y, 0), Z) --> or  (fcmp uno X, Y), Z
  Value *NewFCmp = Builder.CreateFCmp(Pred, X, Y);
  if (auto *NewFCmpInst = dyn_cast<FCmpInst>(NewFCmp)) {
    // Intersect FMF from the 2 source fcmps.
    NewFCmpInst->copyIRFlags(Op0);
    NewFCmpInst->andIRFlags(BO10);
  }
  return BinaryOperator::Create(Opcode, NewFCmp, BO11);
}

/// Match variations of De Morgan's Laws:
/// (~A & ~B) == (~(A | B))
/// (~A | ~B) == (~(A & B))
static Instruction *matchDeMorgansLaws(BinaryOperator &I,
                                       InstCombiner::BuilderTy &Builder) {
  const Instruction::BinaryOps Opcode = I.getOpcode();
  assert((Opcode == Instruction::And || Opcode == Instruction::Or) &&
         "Trying to match De Morgan's Laws with something other than and/or");

  // Flip the logic operation.
  const Instruction::BinaryOps FlippedOpcode =
      (Opcode == Instruction::And) ? Instruction::Or : Instruction::And;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *A, *B;
  if (match(Op0, m_OneUse(m_Not(m_Value(A)))) &&
      match(Op1, m_OneUse(m_Not(m_Value(B)))) &&
      !InstCombiner::isFreeToInvert(A, A->hasOneUse()) &&
      !InstCombiner::isFreeToInvert(B, B->hasOneUse())) {
    Value *AndOr =
        Builder.CreateBinOp(FlippedOpcode, A, B, I.getName() + ".demorgan");
    return BinaryOperator::CreateNot(AndOr);
  }

  // The 'not' ops may require reassociation.
  // (A & ~B) & ~C --> A & ~(B | C)
  // (~B & A) & ~C --> A & ~(B | C)
  // (A | ~B) | ~C --> A | ~(B & C)
  // (~B | A) | ~C --> A | ~(B & C)
  Value *C;
  if (match(Op0, m_OneUse(m_c_BinOp(Opcode, m_Value(A), m_Not(m_Value(B))))) &&
      match(Op1, m_Not(m_Value(C)))) {
    Value *FlippedBO = Builder.CreateBinOp(FlippedOpcode, B, C);
    return BinaryOperator::Create(Opcode, A, Builder.CreateNot(FlippedBO));
  }

  return nullptr;
}

bool InstCombinerImpl::shouldOptimizeCast(CastInst *CI) {
  Value *CastSrc = CI->getOperand(0);

  // Noop casts and casts of constants should be eliminated trivially.
  if (CI->getSrcTy() == CI->getDestTy() || isa<Constant>(CastSrc))
    return false;

  // If this cast is paired with another cast that can be eliminated, we prefer
  // to have it eliminated.
  if (const auto *PrecedingCI = dyn_cast<CastInst>(CastSrc))
    if (isEliminableCastPair(PrecedingCI, CI))
      return false;

  return true;
}

/// Fold {and,or,xor} (cast X), C.
static Instruction *foldLogicCastConstant(BinaryOperator &Logic, CastInst *Cast,
                                          InstCombiner::BuilderTy &Builder) {
  Constant *C = dyn_cast<Constant>(Logic.getOperand(1));
  if (!C)
    return nullptr;

  auto LogicOpc = Logic.getOpcode();
  Type *DestTy = Logic.getType();
  Type *SrcTy = Cast->getSrcTy();

  // Move the logic operation ahead of a zext or sext if the constant is
  // unchanged in the smaller source type. Performing the logic in a smaller
  // type may provide more information to later folds, and the smaller logic
  // instruction may be cheaper (particularly in the case of vectors).
  Value *X;
  if (match(Cast, m_OneUse(m_ZExt(m_Value(X))))) {
    Constant *TruncC = ConstantExpr::getTrunc(C, SrcTy);
    Constant *ZextTruncC = ConstantExpr::getZExt(TruncC, DestTy);
    if (ZextTruncC == C) {
      // LogicOpc (zext X), C --> zext (LogicOpc X, C)
      Value *NewOp = Builder.CreateBinOp(LogicOpc, X, TruncC);
      return new ZExtInst(NewOp, DestTy);
    }
  }

  if (match(Cast, m_OneUse(m_SExt(m_Value(X))))) {
    Constant *TruncC = ConstantExpr::getTrunc(C, SrcTy);
    Constant *SextTruncC = ConstantExpr::getSExt(TruncC, DestTy);
    if (SextTruncC == C) {
      // LogicOpc (sext X), C --> sext (LogicOpc X, C)
      Value *NewOp = Builder.CreateBinOp(LogicOpc, X, TruncC);
      return new SExtInst(NewOp, DestTy);
    }
  }

  return nullptr;
}

/// Fold {and,or,xor} (cast X), Y.
Instruction *InstCombinerImpl::foldCastedBitwiseLogic(BinaryOperator &I) {
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
  if ((Cast0->hasOneUse() || Cast1->hasOneUse()) &&
      shouldOptimizeCast(Cast0) && shouldOptimizeCast(Cast1)) {
    Value *NewOp = Builder.CreateBinOp(LogicOpc, Cast0Src, Cast1Src,
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
    if (Value *Res =
            foldAndOrOfICmps(ICmp0, ICmp1, I, LogicOpc == Instruction::And))
      return CastInst::Create(CastOpcode, Res, DestTy);
    return nullptr;
  }

  // If this is logic(cast(fcmp), cast(fcmp)), try to fold this even if the
  // cast is otherwise not optimizable.  This happens for vector sexts.
  FCmpInst *FCmp0 = dyn_cast<FCmpInst>(Cast0Src);
  FCmpInst *FCmp1 = dyn_cast<FCmpInst>(Cast1Src);
  if (FCmp0 && FCmp1)
    if (Value *R = foldLogicOfFCmps(FCmp0, FCmp1, LogicOpc == Instruction::And))
      return CastInst::Create(CastOpcode, R, DestTy);

  return nullptr;
}

static Instruction *foldAndToXor(BinaryOperator &I,
                                 InstCombiner::BuilderTy &Builder) {
  assert(I.getOpcode() == Instruction::And);
  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);
  Value *A, *B;

  // Operand complexity canonicalization guarantees that the 'or' is Op0.
  // (A | B) & ~(A & B) --> A ^ B
  // (A | B) & ~(B & A) --> A ^ B
  if (match(&I, m_BinOp(m_Or(m_Value(A), m_Value(B)),
                        m_Not(m_c_And(m_Deferred(A), m_Deferred(B))))))
    return BinaryOperator::CreateXor(A, B);

  // (A | ~B) & (~A | B) --> ~(A ^ B)
  // (A | ~B) & (B | ~A) --> ~(A ^ B)
  // (~B | A) & (~A | B) --> ~(A ^ B)
  // (~B | A) & (B | ~A) --> ~(A ^ B)
  if (Op0->hasOneUse() || Op1->hasOneUse())
    if (match(&I, m_BinOp(m_c_Or(m_Value(A), m_Not(m_Value(B))),
                          m_c_Or(m_Not(m_Deferred(A)), m_Deferred(B)))))
      return BinaryOperator::CreateNot(Builder.CreateXor(A, B));

  return nullptr;
}

static Instruction *foldOrToXor(BinaryOperator &I,
                                InstCombiner::BuilderTy &Builder) {
  assert(I.getOpcode() == Instruction::Or);
  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);
  Value *A, *B;

  // Operand complexity canonicalization guarantees that the 'and' is Op0.
  // (A & B) | ~(A | B) --> ~(A ^ B)
  // (A & B) | ~(B | A) --> ~(A ^ B)
  if (Op0->hasOneUse() || Op1->hasOneUse())
    if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
        match(Op1, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))))
      return BinaryOperator::CreateNot(Builder.CreateXor(A, B));

  // Operand complexity canonicalization guarantees that the 'xor' is Op0.
  // (A ^ B) | ~(A | B) --> ~(A & B)
  // (A ^ B) | ~(B | A) --> ~(A & B)
  if (Op0->hasOneUse() || Op1->hasOneUse())
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))) &&
        match(Op1, m_Not(m_c_Or(m_Specific(A), m_Specific(B)))))
      return BinaryOperator::CreateNot(Builder.CreateAnd(A, B));

  // (A & ~B) | (~A & B) --> A ^ B
  // (A & ~B) | (B & ~A) --> A ^ B
  // (~B & A) | (~A & B) --> A ^ B
  // (~B & A) | (B & ~A) --> A ^ B
  if (match(Op0, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_c_And(m_Not(m_Specific(A)), m_Specific(B))))
    return BinaryOperator::CreateXor(A, B);

  return nullptr;
}

/// Return true if a constant shift amount is always less than the specified
/// bit-width. If not, the shift could create poison in the narrower type.
static bool canNarrowShiftAmt(Constant *C, unsigned BitWidth) {
  APInt Threshold(C->getType()->getScalarSizeInBits(), BitWidth);
  return match(C, m_SpecificInt_ICMP(ICmpInst::ICMP_ULT, Threshold));
}

/// Try to use narrower ops (sink zext ops) for an 'and' with binop operand and
/// a common zext operand: and (binop (zext X), C), (zext X).
Instruction *InstCombinerImpl::narrowMaskedBinOp(BinaryOperator &And) {
  // This transform could also apply to {or, and, xor}, but there are better
  // folds for those cases, so we don't expect those patterns here. AShr is not
  // handled because it should always be transformed to LShr in this sequence.
  // The subtract transform is different because it has a constant on the left.
  // Add/mul commute the constant to RHS; sub with constant RHS becomes add.
  Value *Op0 = And.getOperand(0), *Op1 = And.getOperand(1);
  Constant *C;
  if (!match(Op0, m_OneUse(m_Add(m_Specific(Op1), m_Constant(C)))) &&
      !match(Op0, m_OneUse(m_Mul(m_Specific(Op1), m_Constant(C)))) &&
      !match(Op0, m_OneUse(m_LShr(m_Specific(Op1), m_Constant(C)))) &&
      !match(Op0, m_OneUse(m_Shl(m_Specific(Op1), m_Constant(C)))) &&
      !match(Op0, m_OneUse(m_Sub(m_Constant(C), m_Specific(Op1)))))
    return nullptr;

  Value *X;
  if (!match(Op1, m_ZExt(m_Value(X))) || Op1->hasNUsesOrMore(3))
    return nullptr;

  Type *Ty = And.getType();
  if (!isa<VectorType>(Ty) && !shouldChangeType(Ty, X->getType()))
    return nullptr;

  // If we're narrowing a shift, the shift amount must be safe (less than the
  // width) in the narrower type. If the shift amount is greater, instsimplify
  // usually handles that case, but we can't guarantee/assert it.
  Instruction::BinaryOps Opc = cast<BinaryOperator>(Op0)->getOpcode();
  if (Opc == Instruction::LShr || Opc == Instruction::Shl)
    if (!canNarrowShiftAmt(C, X->getType()->getScalarSizeInBits()))
      return nullptr;

  // and (sub C, (zext X)), (zext X) --> zext (and (sub C', X), X)
  // and (binop (zext X), C), (zext X) --> zext (and (binop X, C'), X)
  Value *NewC = ConstantExpr::getTrunc(C, X->getType());
  Value *NewBO = Opc == Instruction::Sub ? Builder.CreateBinOp(Opc, NewC, X)
                                         : Builder.CreateBinOp(Opc, X, NewC);
  return new ZExtInst(Builder.CreateAnd(NewBO, X), Ty);
}

/// Try folding relatively complex patterns for both And and Or operations
/// with all And and Or swapped.
static Instruction *foldComplexAndOrPatterns(BinaryOperator &I,
                                             InstCombiner::BuilderTy &Builder) {
  const Instruction::BinaryOps Opcode = I.getOpcode();
  assert(Opcode == Instruction::And || Opcode == Instruction::Or);

  // Flip the logic operation.
  const Instruction::BinaryOps FlippedOpcode =
      (Opcode == Instruction::And) ? Instruction::Or : Instruction::And;

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *A, *B, *C, *X, *Y, *Dummy;

  // Match following expressions:
  // (~(A | B) & C)
  // (~(A & B) | C)
  // Captures X = ~(A | B) or ~(A & B)
  const auto matchNotOrAnd =
      [Opcode, FlippedOpcode](Value *Op, auto m_A, auto m_B, auto m_C,
                              Value *&X, bool CountUses = false) -> bool {
    if (CountUses && !Op->hasOneUse())
      return false;

    if (match(Op, m_c_BinOp(FlippedOpcode,
                            m_CombineAnd(m_Value(X),
                                         m_Not(m_c_BinOp(Opcode, m_A, m_B))),
                            m_C)))
      return !CountUses || X->hasOneUse();

    return false;
  };

  // (~(A | B) & C) | ... --> ...
  // (~(A & B) | C) & ... --> ...
  // TODO: One use checks are conservative. We just need to check that a total
  //       number of multiple used values does not exceed reduction
  //       in operations.
  if (matchNotOrAnd(Op0, m_Value(A), m_Value(B), m_Value(C), X)) {
    // (~(A | B) & C) | (~(A | C) & B) --> (B ^ C) & ~A
    // (~(A & B) | C) & (~(A & C) | B) --> ~((B ^ C) & A)
    if (matchNotOrAnd(Op1, m_Specific(A), m_Specific(C), m_Specific(B), Dummy,
                      true)) {
      Value *Xor = Builder.CreateXor(B, C);
      return (Opcode == Instruction::Or)
                 ? BinaryOperator::CreateAnd(Xor, Builder.CreateNot(A))
                 : BinaryOperator::CreateNot(Builder.CreateAnd(Xor, A));
    }

    // (~(A | B) & C) | (~(B | C) & A) --> (A ^ C) & ~B
    // (~(A & B) | C) & (~(B & C) | A) --> ~((A ^ C) & B)
    if (matchNotOrAnd(Op1, m_Specific(B), m_Specific(C), m_Specific(A), Dummy,
                      true)) {
      Value *Xor = Builder.CreateXor(A, C);
      return (Opcode == Instruction::Or)
                 ? BinaryOperator::CreateAnd(Xor, Builder.CreateNot(B))
                 : BinaryOperator::CreateNot(Builder.CreateAnd(Xor, B));
    }

    // (~(A | B) & C) | ~(A | C) --> ~((B & C) | A)
    // (~(A & B) | C) & ~(A & C) --> ~((B | C) & A)
    if (match(Op1, m_OneUse(m_Not(m_OneUse(
                       m_c_BinOp(Opcode, m_Specific(A), m_Specific(C)))))))
      return BinaryOperator::CreateNot(Builder.CreateBinOp(
          Opcode, Builder.CreateBinOp(FlippedOpcode, B, C), A));

    // (~(A | B) & C) | ~(B | C) --> ~((A & C) | B)
    // (~(A & B) | C) & ~(B & C) --> ~((A | C) & B)
    if (match(Op1, m_OneUse(m_Not(m_OneUse(
                       m_c_BinOp(Opcode, m_Specific(B), m_Specific(C)))))))
      return BinaryOperator::CreateNot(Builder.CreateBinOp(
          Opcode, Builder.CreateBinOp(FlippedOpcode, A, C), B));

    // (~(A | B) & C) | ~(C | (A ^ B)) --> ~((A | B) & (C | (A ^ B)))
    // Note, the pattern with swapped and/or is not handled because the
    // result is more undefined than a source:
    // (~(A & B) | C) & ~(C & (A ^ B)) --> (A ^ B ^ C) | ~(A | C) is invalid.
    if (Opcode == Instruction::Or && Op0->hasOneUse() &&
        match(Op1, m_OneUse(m_Not(m_CombineAnd(
                       m_Value(Y),
                       m_c_BinOp(Opcode, m_Specific(C),
                                 m_c_Xor(m_Specific(A), m_Specific(B)))))))) {
      // X = ~(A | B)
      // Y = (C | (A ^ B)
      Value *Or = cast<BinaryOperator>(X)->getOperand(0);
      return BinaryOperator::CreateNot(Builder.CreateAnd(Or, Y));
    }
  }

  // (~A & B & C) | ... --> ...
  // (~A | B | C) | ... --> ...
  // TODO: One use checks are conservative. We just need to check that a total
  //       number of multiple used values does not exceed reduction
  //       in operations.
  if (match(Op0,
            m_OneUse(m_c_BinOp(FlippedOpcode,
                               m_BinOp(FlippedOpcode, m_Value(B), m_Value(C)),
                               m_CombineAnd(m_Value(X), m_Not(m_Value(A)))))) ||
      match(Op0, m_OneUse(m_c_BinOp(
                     FlippedOpcode,
                     m_c_BinOp(FlippedOpcode, m_Value(C),
                               m_CombineAnd(m_Value(X), m_Not(m_Value(A)))),
                     m_Value(B))))) {
    // X = ~A
    // (~A & B & C) | ~(A | B | C) --> ~(A | (B ^ C))
    // (~A | B | C) & ~(A & B & C) --> (~A | (B ^ C))
    if (match(Op1, m_OneUse(m_Not(m_c_BinOp(
                       Opcode, m_c_BinOp(Opcode, m_Specific(A), m_Specific(B)),
                       m_Specific(C))))) ||
        match(Op1, m_OneUse(m_Not(m_c_BinOp(
                       Opcode, m_c_BinOp(Opcode, m_Specific(B), m_Specific(C)),
                       m_Specific(A))))) ||
        match(Op1, m_OneUse(m_Not(m_c_BinOp(
                       Opcode, m_c_BinOp(Opcode, m_Specific(A), m_Specific(C)),
                       m_Specific(B)))))) {
      Value *Xor = Builder.CreateXor(B, C);
      return (Opcode == Instruction::Or)
                 ? BinaryOperator::CreateNot(Builder.CreateOr(Xor, A))
                 : BinaryOperator::CreateOr(Xor, X);
    }

    // (~A & B & C) | ~(A | B) --> (C | ~B) & ~A
    // (~A | B | C) & ~(A & B) --> (C & ~B) | ~A
    if (match(Op1, m_OneUse(m_Not(m_OneUse(
                       m_c_BinOp(Opcode, m_Specific(A), m_Specific(B)))))))
      return BinaryOperator::Create(
          FlippedOpcode, Builder.CreateBinOp(Opcode, C, Builder.CreateNot(B)),
          X);

    // (~A & B & C) | ~(A | C) --> (B | ~C) & ~A
    // (~A | B | C) & ~(A & C) --> (B & ~C) | ~A
    if (match(Op1, m_OneUse(m_Not(m_OneUse(
                       m_c_BinOp(Opcode, m_Specific(A), m_Specific(C)))))))
      return BinaryOperator::Create(
          FlippedOpcode, Builder.CreateBinOp(Opcode, B, Builder.CreateNot(C)),
          X);
  }

  return nullptr;
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombinerImpl::visitAnd(BinaryOperator &I) {
  Type *Ty = I.getType();

  if (Value *V = SimplifyAndInst(I.getOperand(0), I.getOperand(1),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *Phi = foldBinopWithPhiOperands(I))
    return Phi;

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Do this before using distributive laws to catch simple and/or/not patterns.
  if (Instruction *Xor = foldAndToXor(I, Builder))
    return Xor;

  if (Instruction *X = foldComplexAndOrPatterns(I, Builder))
    return X;

  // (A|B)&(A|C) -> A|(B&C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyBSwap(I, Builder))
    return replaceInstUsesWith(I, V);

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  Value *X, *Y;
  if (match(Op0, m_OneUse(m_LogicalShift(m_One(), m_Value(X)))) &&
      match(Op1, m_One())) {
    // (1 << X) & 1 --> zext(X == 0)
    // (1 >> X) & 1 --> zext(X == 0)
    Value *IsZero = Builder.CreateICmpEQ(X, ConstantInt::get(Ty, 0));
    return new ZExtInst(IsZero, Ty);
  }

  const APInt *C;
  if (match(Op1, m_APInt(C))) {
    const APInt *XorC;
    if (match(Op0, m_OneUse(m_Xor(m_Value(X), m_APInt(XorC))))) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      Constant *NewC = ConstantInt::get(Ty, *C & *XorC);
      Value *And = Builder.CreateAnd(X, Op1);
      And->takeName(Op0);
      return BinaryOperator::CreateXor(And, NewC);
    }

    const APInt *OrC;
    if (match(Op0, m_OneUse(m_Or(m_Value(X), m_APInt(OrC))))) {
      // (X | C1) & C2 --> (X & C2^(C1&C2)) | (C1&C2)
      // NOTE: This reduces the number of bits set in the & mask, which
      // can expose opportunities for store narrowing for scalars.
      // NOTE: SimplifyDemandedBits should have already removed bits from C1
      // that aren't set in C2. Meaning we can replace (C1&C2) with C1 in
      // above, but this feels safer.
      APInt Together = *C & *OrC;
      Value *And = Builder.CreateAnd(X, ConstantInt::get(Ty, Together ^ *C));
      And->takeName(Op0);
      return BinaryOperator::CreateOr(And, ConstantInt::get(Ty, Together));
    }

    // If the mask is only needed on one incoming arm, push the 'and' op up.
    if (match(Op0, m_OneUse(m_Xor(m_Value(X), m_Value(Y)))) ||
        match(Op0, m_OneUse(m_Or(m_Value(X), m_Value(Y))))) {
      APInt NotAndMask(~(*C));
      BinaryOperator::BinaryOps BinOp = cast<BinaryOperator>(Op0)->getOpcode();
      if (MaskedValueIsZero(X, NotAndMask, 0, &I)) {
        // Not masking anything out for the LHS, move mask to RHS.
        // and ({x}or X, Y), C --> {x}or X, (and Y, C)
        Value *NewRHS = Builder.CreateAnd(Y, Op1, Y->getName() + ".masked");
        return BinaryOperator::Create(BinOp, X, NewRHS);
      }
      if (!isa<Constant>(Y) && MaskedValueIsZero(Y, NotAndMask, 0, &I)) {
        // Not masking anything out for the RHS, move mask to LHS.
        // and ({x}or X, Y), C --> {x}or (and X, C), Y
        Value *NewLHS = Builder.CreateAnd(X, Op1, X->getName() + ".masked");
        return BinaryOperator::Create(BinOp, NewLHS, Y);
      }
    }

    unsigned Width = Ty->getScalarSizeInBits();
    const APInt *ShiftC;
    if (match(Op0, m_OneUse(m_SExt(m_AShr(m_Value(X), m_APInt(ShiftC)))))) {
      if (*C == APInt::getLowBitsSet(Width, Width - ShiftC->getZExtValue())) {
        // We are clearing high bits that were potentially set by sext+ashr:
        // and (sext (ashr X, ShiftC)), C --> lshr (sext X), ShiftC
        Value *Sext = Builder.CreateSExt(X, Ty);
        Constant *ShAmtC = ConstantInt::get(Ty, ShiftC->zext(Width));
        return BinaryOperator::CreateLShr(Sext, ShAmtC);
      }
    }

    // If this 'and' clears the sign-bits added by ashr, replace with lshr:
    // and (ashr X, ShiftC), C --> lshr X, ShiftC
    if (match(Op0, m_AShr(m_Value(X), m_APInt(ShiftC))) && ShiftC->ult(Width) &&
        C->isMask(Width - ShiftC->getZExtValue()))
      return BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, *ShiftC));

    const APInt *AddC;
    if (match(Op0, m_Add(m_Value(X), m_APInt(AddC)))) {
      // If we add zeros to every bit below a mask, the add has no effect:
      // (X + AddC) & LowMaskC --> X & LowMaskC
      unsigned Ctlz = C->countLeadingZeros();
      APInt LowMask(APInt::getLowBitsSet(Width, Width - Ctlz));
      if ((*AddC & LowMask).isZero())
        return BinaryOperator::CreateAnd(X, Op1);

      // If we are masking the result of the add down to exactly one bit and
      // the constant we are adding has no bits set below that bit, then the
      // add is flipping a single bit. Example:
      // (X + 4) & 4 --> (X & 4) ^ 4
      if (Op0->hasOneUse() && C->isPowerOf2() && (*AddC & (*C - 1)) == 0) {
        assert((*C & *AddC) != 0 && "Expected common bit");
        Value *NewAnd = Builder.CreateAnd(X, Op1);
        return BinaryOperator::CreateXor(NewAnd, Op1);
      }
    }

    // ((C1 OP zext(X)) & C2) -> zext((C1 OP X) & C2) if C2 fits in the
    // bitwidth of X and OP behaves well when given trunc(C1) and X.
    auto isSuitableBinOpcode = [](BinaryOperator *B) {
      switch (B->getOpcode()) {
      case Instruction::Xor:
      case Instruction::Or:
      case Instruction::Mul:
      case Instruction::Add:
      case Instruction::Sub:
        return true;
      default:
        return false;
      }
    };
    BinaryOperator *BO;
    if (match(Op0, m_OneUse(m_BinOp(BO))) && isSuitableBinOpcode(BO)) {
      Value *X;
      const APInt *C1;
      // TODO: The one-use restrictions could be relaxed a little if the AND
      // is going to be removed.
      if (match(BO, m_c_BinOp(m_OneUse(m_ZExt(m_Value(X))), m_APInt(C1))) &&
          C->isIntN(X->getType()->getScalarSizeInBits())) {
        unsigned XWidth = X->getType()->getScalarSizeInBits();
        Constant *TruncC1 = ConstantInt::get(X->getType(), C1->trunc(XWidth));
        Value *BinOp = isa<ZExtInst>(BO->getOperand(0))
                           ? Builder.CreateBinOp(BO->getOpcode(), X, TruncC1)
                           : Builder.CreateBinOp(BO->getOpcode(), TruncC1, X);
        Constant *TruncC = ConstantInt::get(X->getType(), C->trunc(XWidth));
        Value *And = Builder.CreateAnd(BinOp, TruncC);
        return new ZExtInst(And, Ty);
      }
    }
  }

  if (match(&I, m_And(m_OneUse(m_Shl(m_ZExt(m_Value(X)), m_Value(Y))),
                      m_SignMask())) &&
      match(Y, m_SpecificInt_ICMP(
                   ICmpInst::Predicate::ICMP_EQ,
                   APInt(Ty->getScalarSizeInBits(),
                         Ty->getScalarSizeInBits() -
                             X->getType()->getScalarSizeInBits())))) {
    auto *SExt = Builder.CreateSExt(X, Ty, X->getName() + ".signext");
    auto *SanitizedSignMask = cast<Constant>(Op1);
    // We must be careful with the undef elements of the sign bit mask, however:
    // the mask elt can be undef iff the shift amount for that lane was undef,
    // otherwise we need to sanitize undef masks to zero.
    SanitizedSignMask = Constant::replaceUndefsWith(
        SanitizedSignMask, ConstantInt::getNullValue(Ty->getScalarType()));
    SanitizedSignMask =
        Constant::mergeUndefsWith(SanitizedSignMask, cast<Constant>(Y));
    return BinaryOperator::CreateAnd(SExt, SanitizedSignMask);
  }

  if (Instruction *Z = narrowMaskedBinOp(I))
    return Z;

  if (I.getType()->isIntOrIntVectorTy(1)) {
    if (auto *SI0 = dyn_cast<SelectInst>(Op0)) {
      if (auto *I =
              foldAndOrOfSelectUsingImpliedCond(Op1, *SI0, /* IsAnd */ true))
        return I;
    }
    if (auto *SI1 = dyn_cast<SelectInst>(Op1)) {
      if (auto *I =
              foldAndOrOfSelectUsingImpliedCond(Op0, *SI1, /* IsAnd */ true))
        return I;
    }
  }

  if (Instruction *FoldedLogic = foldBinOpIntoSelectOrPhi(I))
    return FoldedLogic;

  if (Instruction *DeMorgan = matchDeMorgansLaws(I, Builder))
    return DeMorgan;

  {
    Value *A, *B, *C;
    // A & (A ^ B) --> A & ~B
    if (match(Op1, m_OneUse(m_c_Xor(m_Specific(Op0), m_Value(B)))))
      return BinaryOperator::CreateAnd(Op0, Builder.CreateNot(B));
    // (A ^ B) & A --> A & ~B
    if (match(Op0, m_OneUse(m_c_Xor(m_Specific(Op1), m_Value(B)))))
      return BinaryOperator::CreateAnd(Op1, Builder.CreateNot(B));

    // A & ~(A ^ B) --> A & B
    if (match(Op1, m_Not(m_c_Xor(m_Specific(Op0), m_Value(B)))))
      return BinaryOperator::CreateAnd(Op0, B);
    // ~(A ^ B) & A --> A & B
    if (match(Op0, m_Not(m_c_Xor(m_Specific(Op1), m_Value(B)))))
      return BinaryOperator::CreateAnd(Op1, B);

    // (A ^ B) & ((B ^ C) ^ A) -> (A ^ B) & ~C
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
      if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
        if (Op1->hasOneUse() || isFreeToInvert(C, C->hasOneUse()))
          return BinaryOperator::CreateAnd(Op0, Builder.CreateNot(C));

    // ((A ^ C) ^ B) & (B ^ A) -> (B ^ A) & ~C
    if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
      if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
        if (Op0->hasOneUse() || isFreeToInvert(C, C->hasOneUse()))
          return BinaryOperator::CreateAnd(Op1, Builder.CreateNot(C));

    // (A | B) & (~A ^ B) -> A & B
    // (A | B) & (B ^ ~A) -> A & B
    // (B | A) & (~A ^ B) -> A & B
    // (B | A) & (B ^ ~A) -> A & B
    if (match(Op1, m_c_Xor(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op0, m_c_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);

    // (~A ^ B) & (A | B) -> A & B
    // (~A ^ B) & (B | A) -> A & B
    // (B ^ ~A) & (A | B) -> A & B
    // (B ^ ~A) & (B | A) -> A & B
    if (match(Op0, m_c_Xor(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1, m_c_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);

    // (~A | B) & (A ^ B) -> ~A & B
    // (~A | B) & (B ^ A) -> ~A & B
    // (B | ~A) & (A ^ B) -> ~A & B
    // (B | ~A) & (B ^ A) -> ~A & B
    if (match(Op0, m_c_Or(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1, m_c_Xor(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(Builder.CreateNot(A), B);

    // (A ^ B) & (~A | B) -> ~A & B
    // (B ^ A) & (~A | B) -> ~A & B
    // (A ^ B) & (B | ~A) -> ~A & B
    // (B ^ A) & (B | ~A) -> ~A & B
    if (match(Op1, m_c_Or(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op0, m_c_Xor(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(Builder.CreateNot(A), B);
  }

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = foldAndOrOfICmps(LHS, RHS, I, /* IsAnd */ true))
        return replaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'and' instructions might have to be created.
    if (LHS && match(Op1, m_OneUse(m_LogicalAnd(m_Value(X), m_Value(Y))))) {
      bool IsLogical = isa<SelectInst>(Op1);
      // LHS & (X && Y) --> (LHS && X) && Y
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res =
                foldAndOrOfICmps(LHS, Cmp, I, /* IsAnd */ true, IsLogical))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalAnd(Res, Y)
                                            : Builder.CreateAnd(Res, Y));
      // LHS & (X && Y) --> X && (LHS & Y)
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = foldAndOrOfICmps(LHS, Cmp, I, /* IsAnd */ true,
                                          /* IsLogical */ false))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalAnd(X, Res)
                                            : Builder.CreateAnd(X, Res));
    }
    if (RHS && match(Op0, m_OneUse(m_LogicalAnd(m_Value(X), m_Value(Y))))) {
      bool IsLogical = isa<SelectInst>(Op0);
      // (X && Y) & RHS --> (X && RHS) && Y
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res =
                foldAndOrOfICmps(Cmp, RHS, I, /* IsAnd */ true, IsLogical))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalAnd(Res, Y)
                                            : Builder.CreateAnd(Res, Y));
      // (X && Y) & RHS --> X && (Y & RHS)
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = foldAndOrOfICmps(Cmp, RHS, I, /* IsAnd */ true,
                                          /* IsLogical */ false))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalAnd(X, Res)
                                            : Builder.CreateAnd(X, Res));
    }
  }

  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = foldLogicOfFCmps(LHS, RHS, /*IsAnd*/ true))
        return replaceInstUsesWith(I, Res);

  if (Instruction *FoldedFCmps = reassociateFCmps(I, Builder))
    return FoldedFCmps;

  if (Instruction *CastedAnd = foldCastedBitwiseLogic(I))
    return CastedAnd;

  if (Instruction *Sel = foldBinopOfSextBoolToSelect(I))
    return Sel;

  // and(sext(A), B) / and(B, sext(A)) --> A ? B : 0, where A is i1 or <N x i1>.
  // TODO: Move this into foldBinopOfSextBoolToSelect as a more generalized fold
  //       with binop identity constant. But creating a select with non-constant
  //       arm may not be reversible due to poison semantics. Is that a good
  //       canonicalization?
  Value *A;
  if (match(Op0, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(A, Op1, Constant::getNullValue(Ty));
  if (match(Op1, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(A, Op0, Constant::getNullValue(Ty));

  // (iN X s>> (N-1)) & Y --> (X s< 0) ? Y : 0
  unsigned FullShift = Ty->getScalarSizeInBits() - 1;
  if (match(&I, m_c_And(m_OneUse(m_AShr(m_Value(X), m_SpecificInt(FullShift))),
                        m_Value(Y)))) {
    Value *IsNeg = Builder.CreateIsNeg(X, "isneg");
    return SelectInst::Create(IsNeg, Y, ConstantInt::getNullValue(Ty));
  }
  // If there's a 'not' of the shifted value, swap the select operands:
  // ~(iN X s>> (N-1)) & Y --> (X s< 0) ? 0 : Y
  if (match(&I, m_c_And(m_OneUse(m_Not(
                            m_AShr(m_Value(X), m_SpecificInt(FullShift)))),
                        m_Value(Y)))) {
    Value *IsNeg = Builder.CreateIsNeg(X, "isneg");
    return SelectInst::Create(IsNeg, ConstantInt::getNullValue(Ty), Y);
  }

  // (~x) & y  -->  ~(x | (~y))  iff that gets rid of inversions
  if (sinkNotIntoOtherHandOfAndOrOr(I))
    return &I;

  // An and recurrence w/loop invariant step is equivelent to (and start, step)
  PHINode *PN = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  if (matchSimpleRecurrence(&I, PN, Start, Step) && DT.dominates(Step, PN))
    return replaceInstUsesWith(I, Builder.CreateAnd(Start, Step));

  return nullptr;
}

Instruction *InstCombinerImpl::matchBSwapOrBitReverse(Instruction &I,
                                                      bool MatchBSwaps,
                                                      bool MatchBitReversals) {
  SmallVector<Instruction *, 4> Insts;
  if (!recognizeBSwapOrBitReverseIdiom(&I, MatchBSwaps, MatchBitReversals,
                                       Insts))
    return nullptr;
  Instruction *LastInst = Insts.pop_back_val();
  LastInst->removeFromParent();

  for (auto *Inst : Insts)
    Worklist.push(Inst);
  return LastInst;
}

/// Match UB-safe variants of the funnel shift intrinsic.
static Instruction *matchFunnelShift(Instruction &Or, InstCombinerImpl &IC) {
  // TODO: Can we reduce the code duplication between this and the related
  // rotate matching code under visitSelect and visitTrunc?
  unsigned Width = Or.getType()->getScalarSizeInBits();

  // First, find an or'd pair of opposite shifts:
  // or (lshr ShVal0, ShAmt0), (shl ShVal1, ShAmt1)
  BinaryOperator *Or0, *Or1;
  if (!match(Or.getOperand(0), m_BinOp(Or0)) ||
      !match(Or.getOperand(1), m_BinOp(Or1)))
    return nullptr;

  Value *ShVal0, *ShVal1, *ShAmt0, *ShAmt1;
  if (!match(Or0, m_OneUse(m_LogicalShift(m_Value(ShVal0), m_Value(ShAmt0)))) ||
      !match(Or1, m_OneUse(m_LogicalShift(m_Value(ShVal1), m_Value(ShAmt1)))) ||
      Or0->getOpcode() == Or1->getOpcode())
    return nullptr;

  // Canonicalize to or(shl(ShVal0, ShAmt0), lshr(ShVal1, ShAmt1)).
  if (Or0->getOpcode() == BinaryOperator::LShr) {
    std::swap(Or0, Or1);
    std::swap(ShVal0, ShVal1);
    std::swap(ShAmt0, ShAmt1);
  }
  assert(Or0->getOpcode() == BinaryOperator::Shl &&
         Or1->getOpcode() == BinaryOperator::LShr &&
         "Illegal or(shift,shift) pair");

  // Match the shift amount operands for a funnel shift pattern. This always
  // matches a subtraction on the R operand.
  auto matchShiftAmount = [&](Value *L, Value *R, unsigned Width) -> Value * {
    // Check for constant shift amounts that sum to the bitwidth.
    const APInt *LI, *RI;
    if (match(L, m_APIntAllowUndef(LI)) && match(R, m_APIntAllowUndef(RI)))
      if (LI->ult(Width) && RI->ult(Width) && (*LI + *RI) == Width)
        return ConstantInt::get(L->getType(), *LI);

    Constant *LC, *RC;
    if (match(L, m_Constant(LC)) && match(R, m_Constant(RC)) &&
        match(L, m_SpecificInt_ICMP(ICmpInst::ICMP_ULT, APInt(Width, Width))) &&
        match(R, m_SpecificInt_ICMP(ICmpInst::ICMP_ULT, APInt(Width, Width))) &&
        match(ConstantExpr::getAdd(LC, RC), m_SpecificIntAllowUndef(Width)))
      return ConstantExpr::mergeUndefsWith(LC, RC);

    // (shl ShVal, X) | (lshr ShVal, (Width - x)) iff X < Width.
    // We limit this to X < Width in case the backend re-expands the intrinsic,
    // and has to reintroduce a shift modulo operation (InstCombine might remove
    // it after this fold). This still doesn't guarantee that the final codegen
    // will match this original pattern.
    if (match(R, m_OneUse(m_Sub(m_SpecificInt(Width), m_Specific(L))))) {
      KnownBits KnownL = IC.computeKnownBits(L, /*Depth*/ 0, &Or);
      return KnownL.getMaxValue().ult(Width) ? L : nullptr;
    }

    // For non-constant cases, the following patterns currently only work for
    // rotation patterns.
    // TODO: Add general funnel-shift compatible patterns.
    if (ShVal0 != ShVal1)
      return nullptr;

    // For non-constant cases we don't support non-pow2 shift masks.
    // TODO: Is it worth matching urem as well?
    if (!isPowerOf2_32(Width))
      return nullptr;

    // The shift amount may be masked with negation:
    // (shl ShVal, (X & (Width - 1))) | (lshr ShVal, ((-X) & (Width - 1)))
    Value *X;
    unsigned Mask = Width - 1;
    if (match(L, m_And(m_Value(X), m_SpecificInt(Mask))) &&
        match(R, m_And(m_Neg(m_Specific(X)), m_SpecificInt(Mask))))
      return X;

    // Similar to above, but the shift amount may be extended after masking,
    // so return the extended value as the parameter for the intrinsic.
    if (match(L, m_ZExt(m_And(m_Value(X), m_SpecificInt(Mask)))) &&
        match(R, m_And(m_Neg(m_ZExt(m_And(m_Specific(X), m_SpecificInt(Mask)))),
                       m_SpecificInt(Mask))))
      return L;

    if (match(L, m_ZExt(m_And(m_Value(X), m_SpecificInt(Mask)))) &&
        match(R, m_ZExt(m_And(m_Neg(m_Specific(X)), m_SpecificInt(Mask)))))
      return L;

    return nullptr;
  };

  Value *ShAmt = matchShiftAmount(ShAmt0, ShAmt1, Width);
  bool IsFshl = true; // Sub on LSHR.
  if (!ShAmt) {
    ShAmt = matchShiftAmount(ShAmt1, ShAmt0, Width);
    IsFshl = false; // Sub on SHL.
  }
  if (!ShAmt)
    return nullptr;

  Intrinsic::ID IID = IsFshl ? Intrinsic::fshl : Intrinsic::fshr;
  Function *F = Intrinsic::getDeclaration(Or.getModule(), IID, Or.getType());
  return CallInst::Create(F, {ShVal0, ShVal1, ShAmt});
}

/// Attempt to combine or(zext(x),shl(zext(y),bw/2) concat packing patterns.
static Instruction *matchOrConcat(Instruction &Or,
                                  InstCombiner::BuilderTy &Builder) {
  assert(Or.getOpcode() == Instruction::Or && "bswap requires an 'or'");
  Value *Op0 = Or.getOperand(0), *Op1 = Or.getOperand(1);
  Type *Ty = Or.getType();

  unsigned Width = Ty->getScalarSizeInBits();
  if ((Width & 1) != 0)
    return nullptr;
  unsigned HalfWidth = Width / 2;

  // Canonicalize zext (lower half) to LHS.
  if (!isa<ZExtInst>(Op0))
    std::swap(Op0, Op1);

  // Find lower/upper half.
  Value *LowerSrc, *ShlVal, *UpperSrc;
  const APInt *C;
  if (!match(Op0, m_OneUse(m_ZExt(m_Value(LowerSrc)))) ||
      !match(Op1, m_OneUse(m_Shl(m_Value(ShlVal), m_APInt(C)))) ||
      !match(ShlVal, m_OneUse(m_ZExt(m_Value(UpperSrc)))))
    return nullptr;
  if (*C != HalfWidth || LowerSrc->getType() != UpperSrc->getType() ||
      LowerSrc->getType()->getScalarSizeInBits() != HalfWidth)
    return nullptr;

  auto ConcatIntrinsicCalls = [&](Intrinsic::ID id, Value *Lo, Value *Hi) {
    Value *NewLower = Builder.CreateZExt(Lo, Ty);
    Value *NewUpper = Builder.CreateZExt(Hi, Ty);
    NewUpper = Builder.CreateShl(NewUpper, HalfWidth);
    Value *BinOp = Builder.CreateOr(NewLower, NewUpper);
    Function *F = Intrinsic::getDeclaration(Or.getModule(), id, Ty);
    return Builder.CreateCall(F, BinOp);
  };

  // BSWAP: Push the concat down, swapping the lower/upper sources.
  // concat(bswap(x),bswap(y)) -> bswap(concat(x,y))
  Value *LowerBSwap, *UpperBSwap;
  if (match(LowerSrc, m_BSwap(m_Value(LowerBSwap))) &&
      match(UpperSrc, m_BSwap(m_Value(UpperBSwap))))
    return ConcatIntrinsicCalls(Intrinsic::bswap, UpperBSwap, LowerBSwap);

  // BITREVERSE: Push the concat down, swapping the lower/upper sources.
  // concat(bitreverse(x),bitreverse(y)) -> bitreverse(concat(x,y))
  Value *LowerBRev, *UpperBRev;
  if (match(LowerSrc, m_BitReverse(m_Value(LowerBRev))) &&
      match(UpperSrc, m_BitReverse(m_Value(UpperBRev))))
    return ConcatIntrinsicCalls(Intrinsic::bitreverse, UpperBRev, LowerBRev);

  return nullptr;
}

/// If all elements of two constant vectors are 0/-1 and inverses, return true.
static bool areInverseVectorBitmasks(Constant *C1, Constant *C2) {
  unsigned NumElts = cast<FixedVectorType>(C1->getType())->getNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    Constant *EltC1 = C1->getAggregateElement(i);
    Constant *EltC2 = C2->getAggregateElement(i);
    if (!EltC1 || !EltC2)
      return false;

    // One element must be all ones, and the other must be all zeros.
    if (!((match(EltC1, m_Zero()) && match(EltC2, m_AllOnes())) ||
          (match(EltC2, m_Zero()) && match(EltC1, m_AllOnes()))))
      return false;
  }
  return true;
}

/// We have an expression of the form (A & C) | (B & D). If A is a scalar or
/// vector composed of all-zeros or all-ones values and is the bitwise 'not' of
/// B, it can be used as the condition operand of a select instruction.
Value *InstCombinerImpl::getSelectCondition(Value *A, Value *B) {
  // We may have peeked through bitcasts in the caller.
  // Exit immediately if we don't have (vector) integer types.
  Type *Ty = A->getType();
  if (!Ty->isIntOrIntVectorTy() || !B->getType()->isIntOrIntVectorTy())
    return nullptr;

  // If A is the 'not' operand of B and has enough signbits, we have our answer.
  if (match(B, m_Not(m_Specific(A)))) {
    // If these are scalars or vectors of i1, A can be used directly.
    if (Ty->isIntOrIntVectorTy(1))
      return A;

    // If we look through a vector bitcast, the caller will bitcast the operands
    // to match the condition's number of bits (N x i1).
    // To make this poison-safe, disallow bitcast from wide element to narrow
    // element. That could allow poison in lanes where it was not present in the
    // original code.
    A = peekThroughBitcast(A);
    if (A->getType()->isIntOrIntVectorTy()) {
      unsigned NumSignBits = ComputeNumSignBits(A);
      if (NumSignBits == A->getType()->getScalarSizeInBits() &&
          NumSignBits <= Ty->getScalarSizeInBits())
        return Builder.CreateTrunc(A, CmpInst::makeCmpResultType(A->getType()));
    }
    return nullptr;
  }

  // If both operands are constants, see if the constants are inverse bitmasks.
  Constant *AConst, *BConst;
  if (match(A, m_Constant(AConst)) && match(B, m_Constant(BConst)))
    if (AConst == ConstantExpr::getNot(BConst) &&
        ComputeNumSignBits(A) == Ty->getScalarSizeInBits())
      return Builder.CreateZExtOrTrunc(A, CmpInst::makeCmpResultType(Ty));

  // Look for more complex patterns. The 'not' op may be hidden behind various
  // casts. Look through sexts and bitcasts to find the booleans.
  Value *Cond;
  Value *NotB;
  if (match(A, m_SExt(m_Value(Cond))) &&
      Cond->getType()->isIntOrIntVectorTy(1)) {
    // A = sext i1 Cond; B = sext (not (i1 Cond))
    if (match(B, m_SExt(m_Not(m_Specific(Cond)))))
      return Cond;

    // A = sext i1 Cond; B = not ({bitcast} (sext (i1 Cond)))
    // TODO: The one-use checks are unnecessary or misplaced. If the caller
    //       checked for uses on logic ops/casts, that should be enough to
    //       make this transform worthwhile.
    if (match(B, m_OneUse(m_Not(m_Value(NotB))))) {
      NotB = peekThroughBitcast(NotB, true);
      if (match(NotB, m_SExt(m_Specific(Cond))))
        return Cond;
    }
  }

  // All scalar (and most vector) possibilities should be handled now.
  // Try more matches that only apply to non-splat constant vectors.
  if (!Ty->isVectorTy())
    return nullptr;

  // If both operands are xor'd with constants using the same sexted boolean
  // operand, see if the constants are inverse bitmasks.
  // TODO: Use ConstantExpr::getNot()?
  if (match(A, (m_Xor(m_SExt(m_Value(Cond)), m_Constant(AConst)))) &&
      match(B, (m_Xor(m_SExt(m_Specific(Cond)), m_Constant(BConst)))) &&
      Cond->getType()->isIntOrIntVectorTy(1) &&
      areInverseVectorBitmasks(AConst, BConst)) {
    AConst = ConstantExpr::getTrunc(AConst, CmpInst::makeCmpResultType(Ty));
    return Builder.CreateXor(Cond, AConst);
  }
  return nullptr;
}

/// We have an expression of the form (A & C) | (B & D). Try to simplify this
/// to "A' ? C : D", where A' is a boolean or vector of booleans.
Value *InstCombinerImpl::matchSelectFromAndOr(Value *A, Value *C, Value *B,
                                              Value *D) {
  // The potential condition of the select may be bitcasted. In that case, look
  // through its bitcast and the corresponding bitcast of the 'not' condition.
  Type *OrigType = A->getType();
  A = peekThroughBitcast(A, true);
  B = peekThroughBitcast(B, true);
  if (Value *Cond = getSelectCondition(A, B)) {
    // ((bc Cond) & C) | ((bc ~Cond) & D) --> bc (select Cond, (bc C), (bc D))
    // If this is a vector, we may need to cast to match the condition's length.
    // The bitcasts will either all exist or all not exist. The builder will
    // not create unnecessary casts if the types already match.
    Type *SelTy = A->getType();
    if (auto *VecTy = dyn_cast<VectorType>(Cond->getType())) {
      // For a fixed or scalable vector get N from <{vscale x} N x iM>
      unsigned Elts = VecTy->getElementCount().getKnownMinValue();
      // For a fixed or scalable vector, get the size in bits of N x iM; for a
      // scalar this is just M.
      unsigned SelEltSize = SelTy->getPrimitiveSizeInBits().getKnownMinSize();
      Type *EltTy = Builder.getIntNTy(SelEltSize / Elts);
      SelTy = VectorType::get(EltTy, VecTy->getElementCount());
    }
    Value *BitcastC = Builder.CreateBitCast(C, SelTy);
    Value *BitcastD = Builder.CreateBitCast(D, SelTy);
    Value *Select = Builder.CreateSelect(Cond, BitcastC, BitcastD);
    return Builder.CreateBitCast(Select, OrigType);
  }

  return nullptr;
}

// (icmp eq X, 0) | (icmp ult Other, X) -> (icmp ule Other, X-1)
// (icmp ne X, 0) & (icmp uge Other, X) -> (icmp ugt Other, X-1)
Value *foldAndOrOfICmpEqZeroAndICmp(ICmpInst *LHS, ICmpInst *RHS, bool IsAnd,
                                    IRBuilderBase &Builder) {
  ICmpInst::Predicate LPred =
      IsAnd ? LHS->getInversePredicate() : LHS->getPredicate();
  ICmpInst::Predicate RPred =
      IsAnd ? RHS->getInversePredicate() : RHS->getPredicate();
  Value *LHS0 = LHS->getOperand(0);
  if (LPred != ICmpInst::ICMP_EQ || !match(LHS->getOperand(1), m_Zero()) ||
      !LHS0->getType()->isIntOrIntVectorTy() ||
      !(LHS->hasOneUse() || RHS->hasOneUse()))
    return nullptr;

  Value *Other;
  if (RPred == ICmpInst::ICMP_ULT && RHS->getOperand(1) == LHS0)
    Other = RHS->getOperand(0);
  else if (RPred == ICmpInst::ICMP_UGT && RHS->getOperand(0) == LHS0)
    Other = RHS->getOperand(1);
  else
    return nullptr;

  return Builder.CreateICmp(
      IsAnd ? ICmpInst::ICMP_ULT : ICmpInst::ICMP_UGE,
      Builder.CreateAdd(LHS0, Constant::getAllOnesValue(LHS0->getType())),
      Other);
}

/// Fold (icmp)&(icmp) or (icmp)|(icmp) if possible.
/// If IsLogical is true, then the and/or is in select form and the transform
/// must be poison-safe.
Value *InstCombinerImpl::foldAndOrOfICmps(ICmpInst *LHS, ICmpInst *RHS,
                                          Instruction &I, bool IsAnd,
                                          bool IsLogical) {
  const SimplifyQuery Q = SQ.getWithInstruction(&I);

  // Fold (iszero(A & K1) | iszero(A & K2)) ->  (A & (K1 | K2)) != (K1 | K2)
  // Fold (!iszero(A & K1) & !iszero(A & K2)) ->  (A & (K1 | K2)) == (K1 | K2)
  // if K1 and K2 are a one-bit mask.
  if (Value *V = foldAndOrOfICmpsOfAndWithPow2(LHS, RHS, &I, IsAnd, IsLogical))
    return V;

  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();
  Value *LHS0 = LHS->getOperand(0), *RHS0 = RHS->getOperand(0);
  Value *LHS1 = LHS->getOperand(1), *RHS1 = RHS->getOperand(1);
  const APInt *LHSC = nullptr, *RHSC = nullptr;
  match(LHS1, m_APInt(LHSC));
  match(RHS1, m_APInt(RHSC));

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
  if (predicatesFoldable(PredL, PredR)) {
    if (LHS0 == RHS1 && LHS1 == RHS0) {
      PredL = ICmpInst::getSwappedPredicate(PredL);
      std::swap(LHS0, LHS1);
    }
    if (LHS0 == RHS0 && LHS1 == RHS1) {
      unsigned Code = IsAnd ? getICmpCode(PredL) & getICmpCode(PredR)
                            : getICmpCode(PredL) | getICmpCode(PredR);
      bool IsSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(Code, IsSigned, LHS0, LHS1, Builder);
    }
  }

  // handle (roughly):
  // (icmp ne (A & B), C) | (icmp ne (A & D), E)
  // (icmp eq (A & B), C) & (icmp eq (A & D), E)
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, IsAnd, IsLogical, Builder))
    return V;

  // TODO: One of these directions is fine with logical and/or, the other could
  // be supported by inserting freeze.
  if (!IsLogical) {
    if (Value *V = foldAndOrOfICmpEqZeroAndICmp(LHS, RHS, IsAnd, Builder))
      return V;
    if (Value *V = foldAndOrOfICmpEqZeroAndICmp(RHS, LHS, IsAnd, Builder))
      return V;
  }

  // TODO: Verify whether this is safe for logical and/or.
  if (!IsLogical) {
    if (Value *V = foldAndOrOfICmpsWithConstEq(LHS, RHS, IsAnd, Builder, Q))
      return V;
    if (Value *V = foldAndOrOfICmpsWithConstEq(RHS, LHS, IsAnd, Builder, Q))
      return V;
  }

  if (Value *V = foldIsPowerOf2OrZero(LHS, RHS, IsAnd, Builder))
    return V;
  if (Value *V = foldIsPowerOf2OrZero(RHS, LHS, IsAnd, Builder))
    return V;

  // TODO: One of these directions is fine with logical and/or, the other could
  // be supported by inserting freeze.
  if (!IsLogical) {
    // E.g. (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
    // E.g. (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
    if (Value *V = simplifyRangeCheck(LHS, RHS, /*Inverted=*/!IsAnd))
      return V;

    // E.g. (icmp sgt x, n) | (icmp slt x, 0) --> icmp ugt x, n
    // E.g. (icmp slt x, n) & (icmp sge x, 0) --> icmp ult x, n
    if (Value *V = simplifyRangeCheck(RHS, LHS, /*Inverted=*/!IsAnd))
      return V;
  }

  // TODO: Add conjugated or fold, check whether it is safe for logical and/or.
  if (IsAnd && !IsLogical)
    if (Value *V = foldSignedTruncationCheck(LHS, RHS, I, Builder))
      return V;

  if (Value *V = foldIsPowerOf2(LHS, RHS, IsAnd, Builder))
    return V;

  // TODO: Verify whether this is safe for logical and/or.
  if (!IsLogical) {
    if (Value *X = foldUnsignedUnderflowCheck(LHS, RHS, IsAnd, Q, Builder))
      return X;
    if (Value *X = foldUnsignedUnderflowCheck(RHS, LHS, IsAnd, Q, Builder))
      return X;
  }

  if (Value *X = foldEqOfParts(LHS, RHS, IsAnd))
    return X;

  // (icmp ne A, 0) | (icmp ne B, 0) --> (icmp ne (A|B), 0)
  // (icmp eq A, 0) & (icmp eq B, 0) --> (icmp eq (A|B), 0)
  // TODO: Remove this when foldLogOpOfMaskedICmps can handle undefs.
  if (!IsLogical && PredL == (IsAnd ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE) &&
      PredL == PredR && match(LHS1, m_ZeroInt()) && match(RHS1, m_ZeroInt()) &&
      LHS0->getType() == RHS0->getType()) {
    Value *NewOr = Builder.CreateOr(LHS0, RHS0);
    return Builder.CreateICmp(PredL, NewOr,
                              Constant::getNullValue(NewOr->getType()));
  }

  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  if (!LHSC || !RHSC)
    return nullptr;

  // (trunc x) == C1 & (and x, CA) == C2 -> (and x, CA|CMAX) == C1|C2
  // (trunc x) != C1 | (and x, CA) != C2 -> (and x, CA|CMAX) != C1|C2
  // where CMAX is the all ones value for the truncated type,
  // iff the lower bits of C2 and CA are zero.
  if (PredL == (IsAnd ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE) &&
      PredL == PredR && LHS->hasOneUse() && RHS->hasOneUse()) {
    Value *V;
    const APInt *AndC, *SmallC = nullptr, *BigC = nullptr;

    // (trunc x) == C1 & (and x, CA) == C2
    // (and x, CA) == C2 & (trunc x) == C1
    if (match(RHS0, m_Trunc(m_Value(V))) &&
        match(LHS0, m_And(m_Specific(V), m_APInt(AndC)))) {
      SmallC = RHSC;
      BigC = LHSC;
    } else if (match(LHS0, m_Trunc(m_Value(V))) &&
               match(RHS0, m_And(m_Specific(V), m_APInt(AndC)))) {
      SmallC = LHSC;
      BigC = RHSC;
    }

    if (SmallC && BigC) {
      unsigned BigBitSize = BigC->getBitWidth();
      unsigned SmallBitSize = SmallC->getBitWidth();

      // Check that the low bits are zero.
      APInt Low = APInt::getLowBitsSet(BigBitSize, SmallBitSize);
      if ((Low & *AndC).isZero() && (Low & *BigC).isZero()) {
        Value *NewAnd = Builder.CreateAnd(V, Low | *AndC);
        APInt N = SmallC->zext(BigBitSize) | *BigC;
        Value *NewVal = ConstantInt::get(NewAnd->getType(), N);
        return Builder.CreateICmp(PredL, NewAnd, NewVal);
      }
    }
  }

  return foldAndOrOfICmpsUsingRanges(LHS, RHS, IsAnd);
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombinerImpl::visitOr(BinaryOperator &I) {
  if (Value *V = SimplifyOrInst(I.getOperand(0), I.getOperand(1),
                                SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *Phi = foldBinopWithPhiOperands(I))
    return Phi;

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Do this before using distributive laws to catch simple and/or/not patterns.
  if (Instruction *Xor = foldOrToXor(I, Builder))
    return Xor;

  if (Instruction *X = foldComplexAndOrPatterns(I, Builder))
    return X;

  // (A&B)|(A&C) -> A&(B|C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  if (Value *V = SimplifyBSwap(I, Builder))
    return replaceInstUsesWith(I, V);

  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Type *Ty = I.getType();
  if (Ty->isIntOrIntVectorTy(1)) {
    if (auto *SI0 = dyn_cast<SelectInst>(Op0)) {
      if (auto *I =
              foldAndOrOfSelectUsingImpliedCond(Op1, *SI0, /* IsAnd */ false))
        return I;
    }
    if (auto *SI1 = dyn_cast<SelectInst>(Op1)) {
      if (auto *I =
              foldAndOrOfSelectUsingImpliedCond(Op0, *SI1, /* IsAnd */ false))
        return I;
    }
  }

  if (Instruction *FoldedLogic = foldBinOpIntoSelectOrPhi(I))
    return FoldedLogic;

  if (Instruction *BitOp = matchBSwapOrBitReverse(I, /*MatchBSwaps*/ true,
                                                  /*MatchBitReversals*/ true))
    return BitOp;

  if (Instruction *Funnel = matchFunnelShift(I, *this))
    return Funnel;

  if (Instruction *Concat = matchOrConcat(I, Builder))
    return replaceInstUsesWith(I, Concat);

  Value *X, *Y;
  const APInt *CV;
  if (match(&I, m_c_Or(m_OneUse(m_Xor(m_Value(X), m_APInt(CV))), m_Value(Y))) &&
      !CV->isAllOnes() && MaskedValueIsZero(Y, *CV, 0, &I)) {
    // (X ^ C) | Y -> (X | Y) ^ C iff Y & C == 0
    // The check for a 'not' op is for efficiency (if Y is known zero --> ~X).
    Value *Or = Builder.CreateOr(X, Y);
    return BinaryOperator::CreateXor(Or, ConstantInt::get(Ty, *CV));
  }

  // If the operands have no common bits set:
  // or (mul X, Y), X --> add (mul X, Y), X --> mul X, (Y + 1)
  if (match(&I,
            m_c_Or(m_OneUse(m_Mul(m_Value(X), m_Value(Y))), m_Deferred(X))) &&
      haveNoCommonBitsSet(Op0, Op1, DL)) {
    Value *IncrementY = Builder.CreateAdd(Y, ConstantInt::get(Ty, 1));
    return BinaryOperator::CreateMul(X, IncrementY);
  }

  // (A & C) | (B & D)
  Value *A, *B, *C, *D;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {

    // (A & C0) | (B & C1)
    const APInt *C0, *C1;
    if (match(C, m_APInt(C0)) && match(D, m_APInt(C1))) {
      Value *X;
      if (*C0 == ~*C1) {
        // ((X | B) & MaskC) | (B & ~MaskC) -> (X & MaskC) | B
        if (match(A, m_c_Or(m_Value(X), m_Specific(B))))
          return BinaryOperator::CreateOr(Builder.CreateAnd(X, *C0), B);
        // (A & MaskC) | ((X | A) & ~MaskC) -> (X & ~MaskC) | A
        if (match(B, m_c_Or(m_Specific(A), m_Value(X))))
          return BinaryOperator::CreateOr(Builder.CreateAnd(X, *C1), A);

        // ((X ^ B) & MaskC) | (B & ~MaskC) -> (X & MaskC) ^ B
        if (match(A, m_c_Xor(m_Value(X), m_Specific(B))))
          return BinaryOperator::CreateXor(Builder.CreateAnd(X, *C0), B);
        // (A & MaskC) | ((X ^ A) & ~MaskC) -> (X & ~MaskC) ^ A
        if (match(B, m_c_Xor(m_Specific(A), m_Value(X))))
          return BinaryOperator::CreateXor(Builder.CreateAnd(X, *C1), A);
      }

      if ((*C0 & *C1).isZero()) {
        // ((X | B) & C0) | (B & C1) --> (X | B) & (C0 | C1)
        // iff (C0 & C1) == 0 and (X & ~C0) == 0
        if (match(A, m_c_Or(m_Value(X), m_Specific(B))) &&
            MaskedValueIsZero(X, ~*C0, 0, &I)) {
          Constant *C01 = ConstantInt::get(Ty, *C0 | *C1);
          return BinaryOperator::CreateAnd(A, C01);
        }
        // (A & C0) | ((X | A) & C1) --> (X | A) & (C0 | C1)
        // iff (C0 & C1) == 0 and (X & ~C1) == 0
        if (match(B, m_c_Or(m_Value(X), m_Specific(A))) &&
            MaskedValueIsZero(X, ~*C1, 0, &I)) {
          Constant *C01 = ConstantInt::get(Ty, *C0 | *C1);
          return BinaryOperator::CreateAnd(B, C01);
        }
        // ((X | C2) & C0) | ((X | C3) & C1) --> (X | C2 | C3) & (C0 | C1)
        // iff (C0 & C1) == 0 and (C2 & ~C0) == 0 and (C3 & ~C1) == 0.
        const APInt *C2, *C3;
        if (match(A, m_Or(m_Value(X), m_APInt(C2))) &&
            match(B, m_Or(m_Specific(X), m_APInt(C3))) &&
            (*C2 & ~*C0).isZero() && (*C3 & ~*C1).isZero()) {
          Value *Or = Builder.CreateOr(X, *C2 | *C3, "bitfield");
          Constant *C01 = ConstantInt::get(Ty, *C0 | *C1);
          return BinaryOperator::CreateAnd(Or, C01);
        }
      }
    }

    // Don't try to form a select if it's unlikely that we'll get rid of at
    // least one of the operands. A select is generally more expensive than the
    // 'or' that it is replacing.
    if (Op0->hasOneUse() || Op1->hasOneUse()) {
      // (Cond & C) | (~Cond & D) -> Cond ? C : D, and commuted variants.
      if (Value *V = matchSelectFromAndOr(A, C, B, D))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(A, C, D, B))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(C, A, B, D))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(C, A, D, B))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(B, D, A, C))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(B, D, C, A))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(D, B, A, C))
        return replaceInstUsesWith(I, V);
      if (Value *V = matchSelectFromAndOr(D, B, C, A))
        return replaceInstUsesWith(I, V);
    }
  }

  // (A ^ B) | ((B ^ C) ^ A) -> (A ^ B) | C
  if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
    if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
      return BinaryOperator::CreateOr(Op0, C);

  // ((A ^ C) ^ B) | (B ^ A) -> (B ^ A) | C
  if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
    if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
      return BinaryOperator::CreateOr(Op1, C);

  // ((A & B) ^ C) | B -> C | B
  if (match(Op0, m_c_Xor(m_c_And(m_Value(A), m_Specific(Op1)), m_Value(C))))
    return BinaryOperator::CreateOr(C, Op1);

  // B | ((A & B) ^ C) -> B | C
  if (match(Op1, m_c_Xor(m_c_And(m_Value(A), m_Specific(Op0)), m_Value(C))))
    return BinaryOperator::CreateOr(Op0, C);

  // ((B | C) & A) | B -> B | (A & C)
  if (match(Op0, m_And(m_Or(m_Specific(Op1), m_Value(C)), m_Value(A))))
    return BinaryOperator::CreateOr(Op1, Builder.CreateAnd(A, C));

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
  // ~A | (A ^ B) -> ~(A & B)
  // The swap above should always make Op0 the 'not' for the last case.
  if (match(Op1, m_Xor(m_Value(A), m_Value(B)))) {
    if (Op0 == A || Op0 == B)
      return BinaryOperator::CreateOr(A, B);

    if (match(Op0, m_And(m_Specific(A), m_Specific(B))) ||
        match(Op0, m_And(m_Specific(B), m_Specific(A))))
      return BinaryOperator::CreateOr(A, B);

    if ((Op0->hasOneUse() || Op1->hasOneUse()) &&
        (match(Op0, m_Not(m_Specific(A))) || match(Op0, m_Not(m_Specific(B)))))
      return BinaryOperator::CreateNot(Builder.CreateAnd(A, B));

    if (Op1->hasOneUse() && match(A, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder.CreateNot(B, B->getName() + ".not");
      return BinaryOperator::CreateOr(Not, Op0);
    }
    if (Op1->hasOneUse() && match(B, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder.CreateNot(A, A->getName() + ".not");
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
        Value *Not = Builder.CreateNot(NotOp, NotOp->getName() + ".not");
        return BinaryOperator::CreateOr(Not, Op0);
      }

  if (SwappedForXor)
    std::swap(Op0, Op1);

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = foldAndOrOfICmps(LHS, RHS, I, /* IsAnd */ false))
        return replaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'or' instructions might have to be created.
    Value *X, *Y;
    if (LHS && match(Op1, m_OneUse(m_LogicalOr(m_Value(X), m_Value(Y))))) {
      bool IsLogical = isa<SelectInst>(Op1);
      // LHS | (X || Y) --> (LHS || X) || Y
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res =
                foldAndOrOfICmps(LHS, Cmp, I, /* IsAnd */ false, IsLogical))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalOr(Res, Y)
                                            : Builder.CreateOr(Res, Y));
      // LHS | (X || Y) --> X || (LHS | Y)
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = foldAndOrOfICmps(LHS, Cmp, I, /* IsAnd */ false,
                                          /* IsLogical */ false))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalOr(X, Res)
                                            : Builder.CreateOr(X, Res));
    }
    if (RHS && match(Op0, m_OneUse(m_LogicalOr(m_Value(X), m_Value(Y))))) {
      bool IsLogical = isa<SelectInst>(Op0);
      // (X || Y) | RHS --> (X || RHS) || Y
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res =
                foldAndOrOfICmps(Cmp, RHS, I, /* IsAnd */ false, IsLogical))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalOr(Res, Y)
                                            : Builder.CreateOr(Res, Y));
      // (X || Y) | RHS --> X || (Y | RHS)
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = foldAndOrOfICmps(Cmp, RHS, I, /* IsAnd */ false,
                                          /* IsLogical */ false))
          return replaceInstUsesWith(I, IsLogical
                                            ? Builder.CreateLogicalOr(X, Res)
                                            : Builder.CreateOr(X, Res));
    }
  }

  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = foldLogicOfFCmps(LHS, RHS, /*IsAnd*/ false))
        return replaceInstUsesWith(I, Res);

  if (Instruction *FoldedFCmps = reassociateFCmps(I, Builder))
    return FoldedFCmps;

  if (Instruction *CastedOr = foldCastedBitwiseLogic(I))
    return CastedOr;

  if (Instruction *Sel = foldBinopOfSextBoolToSelect(I))
    return Sel;

  // or(sext(A), B) / or(B, sext(A)) --> A ? -1 : B, where A is i1 or <N x i1>.
  // TODO: Move this into foldBinopOfSextBoolToSelect as a more generalized fold
  //       with binop identity constant. But creating a select with non-constant
  //       arm may not be reversible due to poison semantics. Is that a good
  //       canonicalization?
  if (match(Op0, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(A, ConstantInt::getAllOnesValue(Ty), Op1);
  if (match(Op1, m_OneUse(m_SExt(m_Value(A)))) &&
      A->getType()->isIntOrIntVectorTy(1))
    return SelectInst::Create(A, ConstantInt::getAllOnesValue(Ty), Op0);

  // Note: If we've gotten to the point of visiting the outer OR, then the
  // inner one couldn't be simplified.  If it was a constant, then it won't
  // be simplified by a later pass either, so we try swapping the inner/outer
  // ORs in the hopes that we'll be able to simplify it this way.
  // (X|C) | V --> (X|V) | C
  ConstantInt *CI;
  if (Op0->hasOneUse() && !match(Op1, m_ConstantInt()) &&
      match(Op0, m_Or(m_Value(A), m_ConstantInt(CI)))) {
    Value *Inner = Builder.CreateOr(A, Op1);
    Inner->takeName(Op0);
    return BinaryOperator::CreateOr(Inner, CI);
  }

  // Change (or (bool?A:B),(bool?C:D)) --> (bool?(or A,C):(or B,D))
  // Since this OR statement hasn't been optimized further yet, we hope
  // that this transformation will allow the new ORs to be optimized.
  {
    Value *X = nullptr, *Y = nullptr;
    if (Op0->hasOneUse() && Op1->hasOneUse() &&
        match(Op0, m_Select(m_Value(X), m_Value(A), m_Value(B))) &&
        match(Op1, m_Select(m_Value(Y), m_Value(C), m_Value(D))) && X == Y) {
      Value *orTrue = Builder.CreateOr(A, C);
      Value *orFalse = Builder.CreateOr(B, D);
      return SelectInst::Create(X, orTrue, orFalse);
    }
  }

  // or(ashr(subNSW(Y, X), ScalarSizeInBits(Y) - 1), X)  --> X s> Y ? -1 : X.
  {
    Value *X, *Y;
    if (match(&I, m_c_Or(m_OneUse(m_AShr(
                             m_NSWSub(m_Value(Y), m_Value(X)),
                             m_SpecificInt(Ty->getScalarSizeInBits() - 1))),
                         m_Deferred(X)))) {
      Value *NewICmpInst = Builder.CreateICmpSGT(X, Y);
      Value *AllOnes = ConstantInt::getAllOnesValue(Ty);
      return SelectInst::Create(NewICmpInst, AllOnes, X);
    }
  }

  if (Instruction *V =
          canonicalizeCondSignextOfHighBitExtractToSignextHighBitExtract(I))
    return V;

  CmpInst::Predicate Pred;
  Value *Mul, *Ov, *MulIsNotZero, *UMulWithOv;
  // Check if the OR weakens the overflow condition for umul.with.overflow by
  // treating any non-zero result as overflow. In that case, we overflow if both
  // umul.with.overflow operands are != 0, as in that case the result can only
  // be 0, iff the multiplication overflows.
  if (match(&I,
            m_c_Or(m_CombineAnd(m_ExtractValue<1>(m_Value(UMulWithOv)),
                                m_Value(Ov)),
                   m_CombineAnd(m_ICmp(Pred,
                                       m_CombineAnd(m_ExtractValue<0>(
                                                        m_Deferred(UMulWithOv)),
                                                    m_Value(Mul)),
                                       m_ZeroInt()),
                                m_Value(MulIsNotZero)))) &&
      (Ov->hasOneUse() || (MulIsNotZero->hasOneUse() && Mul->hasOneUse())) &&
      Pred == CmpInst::ICMP_NE) {
    Value *A, *B;
    if (match(UMulWithOv, m_Intrinsic<Intrinsic::umul_with_overflow>(
                              m_Value(A), m_Value(B)))) {
      Value *NotNullA = Builder.CreateIsNotNull(A);
      Value *NotNullB = Builder.CreateIsNotNull(B);
      return BinaryOperator::CreateAnd(NotNullA, NotNullB);
    }
  }

  // (~x) | y  -->  ~(x & (~y))  iff that gets rid of inversions
  if (sinkNotIntoOtherHandOfAndOrOr(I))
    return &I;

  // Improve "get low bit mask up to and including bit X" pattern:
  //   (1 << X) | ((1 << X) + -1)  -->  -1 l>> (bitwidth(x) - 1 - X)
  if (match(&I, m_c_Or(m_Add(m_Shl(m_One(), m_Value(X)), m_AllOnes()),
                       m_Shl(m_One(), m_Deferred(X)))) &&
      match(&I, m_c_Or(m_OneUse(m_Value()), m_Value()))) {
    Value *Sub = Builder.CreateSub(
        ConstantInt::get(Ty, Ty->getScalarSizeInBits() - 1), X);
    return BinaryOperator::CreateLShr(Constant::getAllOnesValue(Ty), Sub);
  }

  // An or recurrence w/loop invariant step is equivelent to (or start, step)
  PHINode *PN = nullptr;
  Value *Start = nullptr, *Step = nullptr;
  if (matchSimpleRecurrence(&I, PN, Start, Step) && DT.dominates(Step, PN))
    return replaceInstUsesWith(I, Builder.CreateOr(Start, Step));

  return nullptr;
}

/// A ^ B can be specified using other logic ops in a variety of patterns. We
/// can fold these early and efficiently by morphing an existing instruction.
static Instruction *foldXorToXor(BinaryOperator &I,
                                 InstCombiner::BuilderTy &Builder) {
  assert(I.getOpcode() == Instruction::Xor);
  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);
  Value *A, *B;

  // There are 4 commuted variants for each of the basic patterns.

  // (A & B) ^ (A | B) -> A ^ B
  // (A & B) ^ (B | A) -> A ^ B
  // (A | B) ^ (A & B) -> A ^ B
  // (A | B) ^ (B & A) -> A ^ B
  if (match(&I, m_c_Xor(m_And(m_Value(A), m_Value(B)),
                        m_c_Or(m_Deferred(A), m_Deferred(B)))))
    return BinaryOperator::CreateXor(A, B);

  // (A | ~B) ^ (~A | B) -> A ^ B
  // (~B | A) ^ (~A | B) -> A ^ B
  // (~A | B) ^ (A | ~B) -> A ^ B
  // (B | ~A) ^ (A | ~B) -> A ^ B
  if (match(&I, m_Xor(m_c_Or(m_Value(A), m_Not(m_Value(B))),
                      m_c_Or(m_Not(m_Deferred(A)), m_Deferred(B)))))
    return BinaryOperator::CreateXor(A, B);

  // (A & ~B) ^ (~A & B) -> A ^ B
  // (~B & A) ^ (~A & B) -> A ^ B
  // (~A & B) ^ (A & ~B) -> A ^ B
  // (B & ~A) ^ (A & ~B) -> A ^ B
  if (match(&I, m_Xor(m_c_And(m_Value(A), m_Not(m_Value(B))),
                      m_c_And(m_Not(m_Deferred(A)), m_Deferred(B)))))
    return BinaryOperator::CreateXor(A, B);

  // For the remaining cases we need to get rid of one of the operands.
  if (!Op0->hasOneUse() && !Op1->hasOneUse())
    return nullptr;

  // (A | B) ^ ~(A & B) -> ~(A ^ B)
  // (A | B) ^ ~(B & A) -> ~(A ^ B)
  // (A & B) ^ ~(A | B) -> ~(A ^ B)
  // (A & B) ^ ~(B | A) -> ~(A ^ B)
  // Complexity sorting ensures the not will be on the right side.
  if ((match(Op0, m_Or(m_Value(A), m_Value(B))) &&
       match(Op1, m_Not(m_c_And(m_Specific(A), m_Specific(B))))) ||
      (match(Op0, m_And(m_Value(A), m_Value(B))) &&
       match(Op1, m_Not(m_c_Or(m_Specific(A), m_Specific(B))))))
    return BinaryOperator::CreateNot(Builder.CreateXor(A, B));

  return nullptr;
}

Value *InstCombinerImpl::foldXorOfICmps(ICmpInst *LHS, ICmpInst *RHS,
                                        BinaryOperator &I) {
  assert(I.getOpcode() == Instruction::Xor && I.getOperand(0) == LHS &&
         I.getOperand(1) == RHS && "Should be 'xor' with these operands");

  ICmpInst::Predicate PredL = LHS->getPredicate(), PredR = RHS->getPredicate();
  Value *LHS0 = LHS->getOperand(0), *LHS1 = LHS->getOperand(1);
  Value *RHS0 = RHS->getOperand(0), *RHS1 = RHS->getOperand(1);

  if (predicatesFoldable(PredL, PredR)) {
    if (LHS0 == RHS1 && LHS1 == RHS0) {
      std::swap(LHS0, LHS1);
      PredL = ICmpInst::getSwappedPredicate(PredL);
    }
    if (LHS0 == RHS0 && LHS1 == RHS1) {
      // (icmp1 A, B) ^ (icmp2 A, B) --> (icmp3 A, B)
      unsigned Code = getICmpCode(PredL) ^ getICmpCode(PredR);
      bool IsSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(Code, IsSigned, LHS0, LHS1, Builder);
    }
  }

  // TODO: This can be generalized to compares of non-signbits using
  // decomposeBitTestICmp(). It could be enhanced more by using (something like)
  // foldLogOpOfMaskedICmps().
  if ((LHS->hasOneUse() || RHS->hasOneUse()) &&
      LHS0->getType() == RHS0->getType() &&
      LHS0->getType()->isIntOrIntVectorTy()) {
    // (X > -1) ^ (Y > -1) --> (X ^ Y) < 0
    // (X <  0) ^ (Y <  0) --> (X ^ Y) < 0
    if ((PredL == CmpInst::ICMP_SGT && match(LHS1, m_AllOnes()) &&
         PredR == CmpInst::ICMP_SGT && match(RHS1, m_AllOnes())) ||
        (PredL == CmpInst::ICMP_SLT && match(LHS1, m_Zero()) &&
         PredR == CmpInst::ICMP_SLT && match(RHS1, m_Zero())))
      return Builder.CreateIsNeg(Builder.CreateXor(LHS0, RHS0));

    // (X > -1) ^ (Y <  0) --> (X ^ Y) > -1
    // (X <  0) ^ (Y > -1) --> (X ^ Y) > -1
    if ((PredL == CmpInst::ICMP_SGT && match(LHS1, m_AllOnes()) &&
         PredR == CmpInst::ICMP_SLT && match(RHS1, m_Zero())) ||
        (PredL == CmpInst::ICMP_SLT && match(LHS1, m_Zero()) &&
         PredR == CmpInst::ICMP_SGT && match(RHS1, m_AllOnes())))
      return Builder.CreateIsNotNeg(Builder.CreateXor(LHS0, RHS0));

  }

  // Instead of trying to imitate the folds for and/or, decompose this 'xor'
  // into those logic ops. That is, try to turn this into an and-of-icmps
  // because we have many folds for that pattern.
  //
  // This is based on a truth table definition of xor:
  // X ^ Y --> (X | Y) & !(X & Y)
  if (Value *OrICmp = SimplifyBinOp(Instruction::Or, LHS, RHS, SQ)) {
    // TODO: If OrICmp is true, then the definition of xor simplifies to !(X&Y).
    // TODO: If OrICmp is false, the whole thing is false (InstSimplify?).
    if (Value *AndICmp = SimplifyBinOp(Instruction::And, LHS, RHS, SQ)) {
      // TODO: Independently handle cases where the 'and' side is a constant.
      ICmpInst *X = nullptr, *Y = nullptr;
      if (OrICmp == LHS && AndICmp == RHS) {
        // (LHS | RHS) & !(LHS & RHS) --> LHS & !RHS  --> X & !Y
        X = LHS;
        Y = RHS;
      }
      if (OrICmp == RHS && AndICmp == LHS) {
        // !(LHS & RHS) & (LHS | RHS) --> !LHS & RHS  --> !Y & X
        X = RHS;
        Y = LHS;
      }
      if (X && Y && (Y->hasOneUse() || canFreelyInvertAllUsersOf(Y, &I))) {
        // Invert the predicate of 'Y', thus inverting its output.
        Y->setPredicate(Y->getInversePredicate());
        // So, are there other uses of Y?
        if (!Y->hasOneUse()) {
          // We need to adapt other uses of Y though. Get a value that matches
          // the original value of Y before inversion. While this increases
          // immediate instruction count, we have just ensured that all the
          // users are freely-invertible, so that 'not' *will* get folded away.
          BuilderTy::InsertPointGuard Guard(Builder);
          // Set insertion point to right after the Y.
          Builder.SetInsertPoint(Y->getParent(), ++(Y->getIterator()));
          Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
          // Replace all uses of Y (excluding the one in NotY!) with NotY.
          Worklist.pushUsersToWorkList(*Y);
          Y->replaceUsesWithIf(NotY,
                               [NotY](Use &U) { return U.getUser() != NotY; });
        }
        // All done.
        return Builder.CreateAnd(LHS, RHS);
      }
    }
  }

  return nullptr;
}

/// If we have a masked merge, in the canonical form of:
/// (assuming that A only has one use.)
///   |        A  |  |B|
///   ((x ^ y) & M) ^ y
///    |  D  |
/// * If M is inverted:
///      |  D  |
///     ((x ^ y) & ~M) ^ y
///   We can canonicalize by swapping the final xor operand
///   to eliminate the 'not' of the mask.
///     ((x ^ y) & M) ^ x
/// * If M is a constant, and D has one use, we transform to 'and' / 'or' ops
///   because that shortens the dependency chain and improves analysis:
///     (x & M) | (y & ~M)
static Instruction *visitMaskedMerge(BinaryOperator &I,
                                     InstCombiner::BuilderTy &Builder) {
  Value *B, *X, *D;
  Value *M;
  if (!match(&I, m_c_Xor(m_Value(B),
                         m_OneUse(m_c_And(
                             m_CombineAnd(m_c_Xor(m_Deferred(B), m_Value(X)),
                                          m_Value(D)),
                             m_Value(M))))))
    return nullptr;

  Value *NotM;
  if (match(M, m_Not(m_Value(NotM)))) {
    // De-invert the mask and swap the value in B part.
    Value *NewA = Builder.CreateAnd(D, NotM);
    return BinaryOperator::CreateXor(NewA, X);
  }

  Constant *C;
  if (D->hasOneUse() && match(M, m_Constant(C))) {
    // Propagating undef is unsafe. Clamp undef elements to -1.
    Type *EltTy = C->getType()->getScalarType();
    C = Constant::replaceUndefsWith(C, ConstantInt::getAllOnesValue(EltTy));
    // Unfold.
    Value *LHS = Builder.CreateAnd(X, C);
    Value *NotC = Builder.CreateNot(C);
    Value *RHS = Builder.CreateAnd(B, NotC);
    return BinaryOperator::CreateOr(LHS, RHS);
  }

  return nullptr;
}

// Transform
//   ~(x ^ y)
// into:
//   (~x) ^ y
// or into
//   x ^ (~y)
static Instruction *sinkNotIntoXor(BinaryOperator &I,
                                   InstCombiner::BuilderTy &Builder) {
  Value *X, *Y;
  // FIXME: one-use check is not needed in general, but currently we are unable
  // to fold 'not' into 'icmp', if that 'icmp' has multiple uses. (D35182)
  if (!match(&I, m_Not(m_OneUse(m_Xor(m_Value(X), m_Value(Y))))))
    return nullptr;

  // We only want to do the transform if it is free to do.
  if (InstCombiner::isFreeToInvert(X, X->hasOneUse())) {
    // Ok, good.
  } else if (InstCombiner::isFreeToInvert(Y, Y->hasOneUse())) {
    std::swap(X, Y);
  } else
    return nullptr;

  Value *NotX = Builder.CreateNot(X, X->getName() + ".not");
  return BinaryOperator::CreateXor(NotX, Y, I.getName() + ".demorgan");
}

/// Canonicalize a shifty way to code absolute value to the more common pattern
/// that uses negation and select.
static Instruction *canonicalizeAbs(BinaryOperator &Xor,
                                    InstCombiner::BuilderTy &Builder) {
  assert(Xor.getOpcode() == Instruction::Xor && "Expected an xor instruction.");

  // There are 4 potential commuted variants. Move the 'ashr' candidate to Op1.
  // We're relying on the fact that we only do this transform when the shift has
  // exactly 2 uses and the add has exactly 1 use (otherwise, we might increase
  // instructions).
  Value *Op0 = Xor.getOperand(0), *Op1 = Xor.getOperand(1);
  if (Op0->hasNUses(2))
    std::swap(Op0, Op1);

  Type *Ty = Xor.getType();
  Value *A;
  const APInt *ShAmt;
  if (match(Op1, m_AShr(m_Value(A), m_APInt(ShAmt))) &&
      Op1->hasNUses(2) && *ShAmt == Ty->getScalarSizeInBits() - 1 &&
      match(Op0, m_OneUse(m_c_Add(m_Specific(A), m_Specific(Op1))))) {
    // Op1 = ashr i32 A, 31   ; smear the sign bit
    // xor (add A, Op1), Op1  ; add -1 and flip bits if negative
    // --> (A < 0) ? -A : A
    Value *IsNeg = Builder.CreateIsNeg(A);
    // Copy the nuw/nsw flags from the add to the negate.
    auto *Add = cast<BinaryOperator>(Op0);
    Value *NegA = Builder.CreateNeg(A, "", Add->hasNoUnsignedWrap(),
                                   Add->hasNoSignedWrap());
    return SelectInst::Create(IsNeg, NegA, A);
  }
  return nullptr;
}

// Transform
//   z = (~x) &/| y
// into:
//   z = ~(x |/& (~y))
// iff y is free to invert and all uses of z can be freely updated.
bool InstCombinerImpl::sinkNotIntoOtherHandOfAndOrOr(BinaryOperator &I) {
  Instruction::BinaryOps NewOpc;
  switch (I.getOpcode()) {
  case Instruction::And:
    NewOpc = Instruction::Or;
    break;
  case Instruction::Or:
    NewOpc = Instruction::And;
    break;
  default:
    return false;
  };

  Value *X, *Y;
  if (!match(&I, m_c_BinOp(m_Not(m_Value(X)), m_Value(Y))))
    return false;

  // Will we be able to fold the `not` into Y eventually?
  if (!InstCombiner::isFreeToInvert(Y, Y->hasOneUse()))
    return false;

  // And can our users be adapted?
  if (!InstCombiner::canFreelyInvertAllUsersOf(&I, /*IgnoredUser=*/nullptr))
    return false;

  Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
  Value *NewBinOp =
      BinaryOperator::Create(NewOpc, X, NotY, I.getName() + ".not");
  Builder.Insert(NewBinOp);
  replaceInstUsesWith(I, NewBinOp);
  // We can not just create an outer `not`, it will most likely be immediately
  // folded back, reconstructing our initial pattern, and causing an
  // infinite combine loop, so immediately manually fold it away.
  freelyInvertAllUsersOf(NewBinOp);
  return true;
}

Instruction *InstCombinerImpl::foldNot(BinaryOperator &I) {
  Value *NotOp;
  if (!match(&I, m_Not(m_Value(NotOp))))
    return nullptr;

  // Apply DeMorgan's Law for 'nand' / 'nor' logic with an inverted operand.
  // We must eliminate the and/or (one-use) for these transforms to not increase
  // the instruction count.
  //
  // ~(~X & Y) --> (X | ~Y)
  // ~(Y & ~X) --> (X | ~Y)
  //
  // Note: The logical matches do not check for the commuted patterns because
  //       those are handled via SimplifySelectsFeedingBinaryOp().
  Type *Ty = I.getType();
  Value *X, *Y;
  if (match(NotOp, m_OneUse(m_c_And(m_Not(m_Value(X)), m_Value(Y))))) {
    Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
    return BinaryOperator::CreateOr(X, NotY);
  }
  if (match(NotOp, m_OneUse(m_LogicalAnd(m_Not(m_Value(X)), m_Value(Y))))) {
    Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
    return SelectInst::Create(X, ConstantInt::getTrue(Ty), NotY);
  }

  // ~(~X | Y) --> (X & ~Y)
  // ~(Y | ~X) --> (X & ~Y)
  if (match(NotOp, m_OneUse(m_c_Or(m_Not(m_Value(X)), m_Value(Y))))) {
    Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
    return BinaryOperator::CreateAnd(X, NotY);
  }
  if (match(NotOp, m_OneUse(m_LogicalOr(m_Not(m_Value(X)), m_Value(Y))))) {
    Value *NotY = Builder.CreateNot(Y, Y->getName() + ".not");
    return SelectInst::Create(X, NotY, ConstantInt::getFalse(Ty));
  }

  // Is this a 'not' (~) fed by a binary operator?
  BinaryOperator *NotVal;
  if (match(NotOp, m_BinOp(NotVal))) {
    if (NotVal->getOpcode() == Instruction::And ||
        NotVal->getOpcode() == Instruction::Or) {
      // Apply DeMorgan's Law when inverts are free:
      // ~(X & Y) --> (~X | ~Y)
      // ~(X | Y) --> (~X & ~Y)
      if (isFreeToInvert(NotVal->getOperand(0),
                         NotVal->getOperand(0)->hasOneUse()) &&
          isFreeToInvert(NotVal->getOperand(1),
                         NotVal->getOperand(1)->hasOneUse())) {
        Value *NotX = Builder.CreateNot(NotVal->getOperand(0), "notlhs");
        Value *NotY = Builder.CreateNot(NotVal->getOperand(1), "notrhs");
        if (NotVal->getOpcode() == Instruction::And)
          return BinaryOperator::CreateOr(NotX, NotY);
        return BinaryOperator::CreateAnd(NotX, NotY);
      }
    }

    // ~((-X) | Y) --> (X - 1) & (~Y)
    if (match(NotVal,
              m_OneUse(m_c_Or(m_OneUse(m_Neg(m_Value(X))), m_Value(Y))))) {
      Value *DecX = Builder.CreateAdd(X, ConstantInt::getAllOnesValue(Ty));
      Value *NotY = Builder.CreateNot(Y);
      return BinaryOperator::CreateAnd(DecX, NotY);
    }

    // ~(~X >>s Y) --> (X >>s Y)
    if (match(NotVal, m_AShr(m_Not(m_Value(X)), m_Value(Y))))
      return BinaryOperator::CreateAShr(X, Y);

    // If we are inverting a right-shifted constant, we may be able to eliminate
    // the 'not' by inverting the constant and using the opposite shift type.
    // Canonicalization rules ensure that only a negative constant uses 'ashr',
    // but we must check that in case that transform has not fired yet.

    // ~(C >>s Y) --> ~C >>u Y (when inverting the replicated sign bits)
    Constant *C;
    if (match(NotVal, m_AShr(m_Constant(C), m_Value(Y))) &&
        match(C, m_Negative())) {
      // We matched a negative constant, so propagating undef is unsafe.
      // Clamp undef elements to -1.
      Type *EltTy = Ty->getScalarType();
      C = Constant::replaceUndefsWith(C, ConstantInt::getAllOnesValue(EltTy));
      return BinaryOperator::CreateLShr(ConstantExpr::getNot(C), Y);
    }

    // ~(C >>u Y) --> ~C >>s Y (when inverting the replicated sign bits)
    if (match(NotVal, m_LShr(m_Constant(C), m_Value(Y))) &&
        match(C, m_NonNegative())) {
      // We matched a non-negative constant, so propagating undef is unsafe.
      // Clamp undef elements to 0.
      Type *EltTy = Ty->getScalarType();
      C = Constant::replaceUndefsWith(C, ConstantInt::getNullValue(EltTy));
      return BinaryOperator::CreateAShr(ConstantExpr::getNot(C), Y);
    }

    // ~(X + C) --> ~C - X
    if (match(NotVal, m_c_Add(m_Value(X), m_ImmConstant(C))))
      return BinaryOperator::CreateSub(ConstantExpr::getNot(C), X);

    // ~(X - Y) --> ~X + Y
    // FIXME: is it really beneficial to sink the `not` here?
    if (match(NotVal, m_Sub(m_Value(X), m_Value(Y))))
      if (isa<Constant>(X) || NotVal->hasOneUse())
        return BinaryOperator::CreateAdd(Builder.CreateNot(X), Y);

    // ~(~X + Y) --> X - Y
    if (match(NotVal, m_c_Add(m_Not(m_Value(X)), m_Value(Y))))
      return BinaryOperator::CreateWithCopiedFlags(Instruction::Sub, X, Y,
                                                   NotVal);
  }

  // not (cmp A, B) = !cmp A, B
  CmpInst::Predicate Pred;
  if (match(NotOp, m_OneUse(m_Cmp(Pred, m_Value(), m_Value())))) {
    cast<CmpInst>(NotOp)->setPredicate(CmpInst::getInversePredicate(Pred));
    return replaceInstUsesWith(I, NotOp);
  }

  // Eliminate a bitwise 'not' op of 'not' min/max by inverting the min/max:
  // ~min(~X, ~Y) --> max(X, Y)
  // ~max(~X, Y) --> min(X, ~Y)
  auto *II = dyn_cast<IntrinsicInst>(NotOp);
  if (II && II->hasOneUse()) {
    if (match(NotOp, m_MaxOrMin(m_Value(X), m_Value(Y))) &&
        isFreeToInvert(X, X->hasOneUse()) &&
        isFreeToInvert(Y, Y->hasOneUse())) {
      Intrinsic::ID InvID = getInverseMinMaxIntrinsic(II->getIntrinsicID());
      Value *NotX = Builder.CreateNot(X);
      Value *NotY = Builder.CreateNot(Y);
      Value *InvMaxMin = Builder.CreateBinaryIntrinsic(InvID, NotX, NotY);
      return replaceInstUsesWith(I, InvMaxMin);
    }
    if (match(NotOp, m_c_MaxOrMin(m_Not(m_Value(X)), m_Value(Y)))) {
      Intrinsic::ID InvID = getInverseMinMaxIntrinsic(II->getIntrinsicID());
      Value *NotY = Builder.CreateNot(Y);
      Value *InvMaxMin = Builder.CreateBinaryIntrinsic(InvID, X, NotY);
      return replaceInstUsesWith(I, InvMaxMin);
    }
  }

  if (NotOp->hasOneUse()) {
    // Pull 'not' into operands of select if both operands are one-use compares
    // or one is one-use compare and the other one is a constant.
    // Inverting the predicates eliminates the 'not' operation.
    // Example:
    //   not (select ?, (cmp TPred, ?, ?), (cmp FPred, ?, ?) -->
    //     select ?, (cmp InvTPred, ?, ?), (cmp InvFPred, ?, ?)
    //   not (select ?, (cmp TPred, ?, ?), true -->
    //     select ?, (cmp InvTPred, ?, ?), false
    if (auto *Sel = dyn_cast<SelectInst>(NotOp)) {
      Value *TV = Sel->getTrueValue();
      Value *FV = Sel->getFalseValue();
      auto *CmpT = dyn_cast<CmpInst>(TV);
      auto *CmpF = dyn_cast<CmpInst>(FV);
      bool InvertibleT = (CmpT && CmpT->hasOneUse()) || isa<Constant>(TV);
      bool InvertibleF = (CmpF && CmpF->hasOneUse()) || isa<Constant>(FV);
      if (InvertibleT && InvertibleF) {
        if (CmpT)
          CmpT->setPredicate(CmpT->getInversePredicate());
        else
          Sel->setTrueValue(ConstantExpr::getNot(cast<Constant>(TV)));
        if (CmpF)
          CmpF->setPredicate(CmpF->getInversePredicate());
        else
          Sel->setFalseValue(ConstantExpr::getNot(cast<Constant>(FV)));
        return replaceInstUsesWith(I, Sel);
      }
    }
  }

  if (Instruction *NewXor = sinkNotIntoXor(I, Builder))
    return NewXor;

  return nullptr;
}

// FIXME: We use commutative matchers (m_c_*) for some, but not all, matches
// here. We should standardize that construct where it is needed or choose some
// other way to ensure that commutated variants of patterns are not missed.
Instruction *InstCombinerImpl::visitXor(BinaryOperator &I) {
  if (Value *V = SimplifyXorInst(I.getOperand(0), I.getOperand(1),
                                 SQ.getWithInstruction(&I)))
    return replaceInstUsesWith(I, V);

  if (SimplifyAssociativeOrCommutative(I))
    return &I;

  if (Instruction *X = foldVectorBinop(I))
    return X;

  if (Instruction *Phi = foldBinopWithPhiOperands(I))
    return Phi;

  if (Instruction *NewXor = foldXorToXor(I, Builder))
    return NewXor;

  // (A&B)^(A&C) -> A&(B^C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return replaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I, Builder))
    return replaceInstUsesWith(I, V);

  if (Instruction *R = foldNot(I))
    return R;

  // Fold (X & M) ^ (Y & ~M) -> (X & M) | (Y & ~M)
  // This it a special case in haveNoCommonBitsSet, but the computeKnownBits
  // calls in there are unnecessary as SimplifyDemandedInstructionBits should
  // have already taken care of those cases.
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  Value *M;
  if (match(&I, m_c_Xor(m_c_And(m_Not(m_Value(M)), m_Value()),
                        m_c_And(m_Deferred(M), m_Value()))))
    return BinaryOperator::CreateOr(Op0, Op1);

  if (Instruction *Xor = visitMaskedMerge(I, Builder))
    return Xor;

  Value *X, *Y;
  Constant *C1;
  if (match(Op1, m_Constant(C1))) {
    Constant *C2;

    if (match(Op0, m_OneUse(m_Or(m_Value(X), m_ImmConstant(C2)))) &&
        match(C1, m_ImmConstant())) {
      // (X | C2) ^ C1 --> (X & ~C2) ^ (C1^C2)
      C2 = Constant::replaceUndefsWith(
          C2, Constant::getAllOnesValue(C2->getType()->getScalarType()));
      Value *And = Builder.CreateAnd(
          X, Constant::mergeUndefsWith(ConstantExpr::getNot(C2), C1));
      return BinaryOperator::CreateXor(
          And, Constant::mergeUndefsWith(ConstantExpr::getXor(C1, C2), C1));
    }

    // Use DeMorgan and reassociation to eliminate a 'not' op.
    if (match(Op0, m_OneUse(m_Or(m_Not(m_Value(X)), m_Constant(C2))))) {
      // (~X | C2) ^ C1 --> ((X & ~C2) ^ -1) ^ C1 --> (X & ~C2) ^ ~C1
      Value *And = Builder.CreateAnd(X, ConstantExpr::getNot(C2));
      return BinaryOperator::CreateXor(And, ConstantExpr::getNot(C1));
    }
    if (match(Op0, m_OneUse(m_And(m_Not(m_Value(X)), m_Constant(C2))))) {
      // (~X & C2) ^ C1 --> ((X | ~C2) ^ -1) ^ C1 --> (X | ~C2) ^ ~C1
      Value *Or = Builder.CreateOr(X, ConstantExpr::getNot(C2));
      return BinaryOperator::CreateXor(Or, ConstantExpr::getNot(C1));
    }

    // Convert xor ([trunc] (ashr X, BW-1)), C =>
    //   select(X >s -1, C, ~C)
    // The ashr creates "AllZeroOrAllOne's", which then optionally inverses the
    // constant depending on whether this input is less than 0.
    const APInt *CA;
    if (match(Op0, m_OneUse(m_TruncOrSelf(
                       m_AShr(m_Value(X), m_APIntAllowUndef(CA))))) &&
        *CA == X->getType()->getScalarSizeInBits() - 1 &&
        !match(C1, m_AllOnes())) {
      assert(!C1->isZeroValue() && "Unexpected xor with 0");
      Value *IsNotNeg = Builder.CreateIsNotNeg(X);
      return SelectInst::Create(IsNotNeg, Op1, Builder.CreateNot(Op1));
    }
  }

  Type *Ty = I.getType();
  {
    const APInt *RHSC;
    if (match(Op1, m_APInt(RHSC))) {
      Value *X;
      const APInt *C;
      // (C - X) ^ signmaskC --> (C + signmaskC) - X
      if (RHSC->isSignMask() && match(Op0, m_Sub(m_APInt(C), m_Value(X))))
        return BinaryOperator::CreateSub(ConstantInt::get(Ty, *C + *RHSC), X);

      // (X + C) ^ signmaskC --> X + (C + signmaskC)
      if (RHSC->isSignMask() && match(Op0, m_Add(m_Value(X), m_APInt(C))))
        return BinaryOperator::CreateAdd(X, ConstantInt::get(Ty, *C + *RHSC));

      // (X | C) ^ RHSC --> X ^ (C ^ RHSC) iff X & C == 0
      if (match(Op0, m_Or(m_Value(X), m_APInt(C))) &&
          MaskedValueIsZero(X, *C, 0, &I))
        return BinaryOperator::CreateXor(X, ConstantInt::get(Ty, *C ^ *RHSC));

      // If RHSC is inverting the remaining bits of shifted X,
      // canonicalize to a 'not' before the shift to help SCEV and codegen:
      // (X << C) ^ RHSC --> ~X << C
      if (match(Op0, m_OneUse(m_Shl(m_Value(X), m_APInt(C)))) &&
          *RHSC == APInt::getAllOnes(Ty->getScalarSizeInBits()).shl(*C)) {
        Value *NotX = Builder.CreateNot(X);
        return BinaryOperator::CreateShl(NotX, ConstantInt::get(Ty, *C));
      }
      // (X >>u C) ^ RHSC --> ~X >>u C
      if (match(Op0, m_OneUse(m_LShr(m_Value(X), m_APInt(C)))) &&
          *RHSC == APInt::getAllOnes(Ty->getScalarSizeInBits()).lshr(*C)) {
        Value *NotX = Builder.CreateNot(X);
        return BinaryOperator::CreateLShr(NotX, ConstantInt::get(Ty, *C));
      }
      // TODO: We could handle 'ashr' here as well. That would be matching
      //       a 'not' op and moving it before the shift. Doing that requires
      //       preventing the inverse fold in canShiftBinOpWithConstantRHS().
    }
  }

  // FIXME: This should not be limited to scalar (pull into APInt match above).
  {
    Value *X;
    ConstantInt *C1, *C2, *C3;
    // ((X^C1) >> C2) ^ C3 -> (X>>C2) ^ ((C1>>C2)^C3)
    if (match(Op1, m_ConstantInt(C3)) &&
        match(Op0, m_LShr(m_Xor(m_Value(X), m_ConstantInt(C1)),
                          m_ConstantInt(C2))) &&
        Op0->hasOneUse()) {
      // fold (C1 >> C2) ^ C3
      APInt FoldConst = C1->getValue().lshr(C2->getValue());
      FoldConst ^= C3->getValue();
      // Prepare the two operands.
      auto *Opnd0 = Builder.CreateLShr(X, C2);
      Opnd0->takeName(Op0);
      return BinaryOperator::CreateXor(Opnd0, ConstantInt::get(Ty, FoldConst));
    }
  }

  if (Instruction *FoldedLogic = foldBinOpIntoSelectOrPhi(I))
    return FoldedLogic;

  // Y ^ (X | Y) --> X & ~Y
  // Y ^ (Y | X) --> X & ~Y
  if (match(Op1, m_OneUse(m_c_Or(m_Value(X), m_Specific(Op0)))))
    return BinaryOperator::CreateAnd(X, Builder.CreateNot(Op0));
  // (X | Y) ^ Y --> X & ~Y
  // (Y | X) ^ Y --> X & ~Y
  if (match(Op0, m_OneUse(m_c_Or(m_Value(X), m_Specific(Op1)))))
    return BinaryOperator::CreateAnd(X, Builder.CreateNot(Op1));

  // Y ^ (X & Y) --> ~X & Y
  // Y ^ (Y & X) --> ~X & Y
  if (match(Op1, m_OneUse(m_c_And(m_Value(X), m_Specific(Op0)))))
    return BinaryOperator::CreateAnd(Op0, Builder.CreateNot(X));
  // (X & Y) ^ Y --> ~X & Y
  // (Y & X) ^ Y --> ~X & Y
  // Canonical form is (X & C) ^ C; don't touch that.
  // TODO: A 'not' op is better for analysis and codegen, but demanded bits must
  //       be fixed to prefer that (otherwise we get infinite looping).
  if (!match(Op1, m_Constant()) &&
      match(Op0, m_OneUse(m_c_And(m_Value(X), m_Specific(Op1)))))
    return BinaryOperator::CreateAnd(Op1, Builder.CreateNot(X));

  Value *A, *B, *C;
  // (A ^ B) ^ (A | C) --> (~A & C) ^ B -- There are 4 commuted variants.
  if (match(&I, m_c_Xor(m_OneUse(m_Xor(m_Value(A), m_Value(B))),
                        m_OneUse(m_c_Or(m_Deferred(A), m_Value(C))))))
      return BinaryOperator::CreateXor(
          Builder.CreateAnd(Builder.CreateNot(A), C), B);

  // (A ^ B) ^ (B | C) --> (~B & C) ^ A -- There are 4 commuted variants.
  if (match(&I, m_c_Xor(m_OneUse(m_Xor(m_Value(A), m_Value(B))),
                        m_OneUse(m_c_Or(m_Deferred(B), m_Value(C))))))
      return BinaryOperator::CreateXor(
          Builder.CreateAnd(Builder.CreateNot(B), C), A);

  // (A & B) ^ (A ^ B) -> (A | B)
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      match(Op1, m_c_Xor(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateOr(A, B);
  // (A ^ B) ^ (A & B) -> (A | B)
  if (match(Op0, m_Xor(m_Value(A), m_Value(B))) &&
      match(Op1, m_c_And(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateOr(A, B);

  // (A & ~B) ^ ~A -> ~(A & B)
  // (~B & A) ^ ~A -> ~(A & B)
  if (match(Op0, m_c_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_Not(m_Specific(A))))
    return BinaryOperator::CreateNot(Builder.CreateAnd(A, B));

  // (~A & B) ^ A --> A | B -- There are 4 commuted variants.
  if (match(&I, m_c_Xor(m_c_And(m_Not(m_Value(A)), m_Value(B)), m_Deferred(A))))
    return BinaryOperator::CreateOr(A, B);

  // (~A | B) ^ A --> ~(A & B)
  if (match(Op0, m_OneUse(m_c_Or(m_Not(m_Specific(Op1)), m_Value(B)))))
    return BinaryOperator::CreateNot(Builder.CreateAnd(Op1, B));

  // A ^ (~A | B) --> ~(A & B)
  if (match(Op1, m_OneUse(m_c_Or(m_Not(m_Specific(Op0)), m_Value(B)))))
    return BinaryOperator::CreateNot(Builder.CreateAnd(Op0, B));

  // (A | B) ^ (A | C) --> (B ^ C) & ~A -- There are 4 commuted variants.
  // TODO: Loosen one-use restriction if common operand is a constant.
  Value *D;
  if (match(Op0, m_OneUse(m_Or(m_Value(A), m_Value(B)))) &&
      match(Op1, m_OneUse(m_Or(m_Value(C), m_Value(D))))) {
    if (B == C || B == D)
      std::swap(A, B);
    if (A == C)
      std::swap(C, D);
    if (A == D) {
      Value *NotA = Builder.CreateNot(A);
      return BinaryOperator::CreateAnd(Builder.CreateXor(B, C), NotA);
    }
  }

  if (auto *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
    if (auto *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
      if (Value *V = foldXorOfICmps(LHS, RHS, I))
        return replaceInstUsesWith(I, V);

  if (Instruction *CastedXor = foldCastedBitwiseLogic(I))
    return CastedXor;

  if (Instruction *Abs = canonicalizeAbs(I, Builder))
    return Abs;

  // Otherwise, if all else failed, try to hoist the xor-by-constant:
  //   (X ^ C) ^ Y --> (X ^ Y) ^ C
  // Just like we do in other places, we completely avoid the fold
  // for constantexprs, at least to avoid endless combine loop.
  if (match(&I, m_c_Xor(m_OneUse(m_Xor(m_CombineAnd(m_Value(X),
                                                    m_Unless(m_ConstantExpr())),
                                       m_ImmConstant(C1))),
                        m_Value(Y))))
    return BinaryOperator::CreateXor(Builder.CreateXor(X, Y), C1);

  return nullptr;
}
