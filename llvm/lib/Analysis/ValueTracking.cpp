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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <cstring>
using namespace llvm;
using namespace llvm::PatternMatch;

const unsigned MaxDepth = 6;

// Controls the number of uses of the value searched for possible
// dominating comparisons.
static cl::opt<unsigned> DomConditionsMaxUses("dom-conditions-max-uses",
                                              cl::Hidden, cl::init(20));

/// Returns the bitwidth of the given scalar or pointer type (if unknown returns
/// 0). For vector types, returns the element type's bitwidth.
static unsigned getBitWidth(Type *Ty, const DataLayout &DL) {
  if (unsigned BitWidth = Ty->getScalarSizeInBits())
    return BitWidth;

  return DL.getPointerTypeSizeInBits(Ty);
}

namespace {
// Simplifying using an assume can only be done in a particular control-flow
// context (the context instruction provides that context). If an assume and
// the context instruction are not in the same block then the DT helps in
// figuring out if we can use it.
struct Query {
  const DataLayout &DL;
  AssumptionCache *AC;
  const Instruction *CxtI;
  const DominatorTree *DT;

  /// Set of assumptions that should be excluded from further queries.
  /// This is because of the potential for mutual recursion to cause
  /// computeKnownBits to repeatedly visit the same assume intrinsic. The
  /// classic case of this is assume(x = y), which will attempt to determine
  /// bits in x from bits in y, which will attempt to determine bits in y from
  /// bits in x, etc. Regarding the mutual recursion, computeKnownBits can call
  /// isKnownNonZero, which calls computeKnownBits and ComputeSignBit and
  /// isKnownToBeAPowerOfTwo (all of which can call computeKnownBits), and so
  /// on.
  std::array<const Value*, MaxDepth> Excluded;
  unsigned NumExcluded;

  Query(const DataLayout &DL, AssumptionCache *AC, const Instruction *CxtI,
        const DominatorTree *DT)
      : DL(DL), AC(AC), CxtI(CxtI), DT(DT), NumExcluded(0) {}

  Query(const Query &Q, const Value *NewExcl)
      : DL(Q.DL), AC(Q.AC), CxtI(Q.CxtI), DT(Q.DT), NumExcluded(Q.NumExcluded) {
    Excluded = Q.Excluded;
    Excluded[NumExcluded++] = NewExcl;
    assert(NumExcluded <= Excluded.size());
  }

  bool isExcluded(const Value *Value) const {
    if (NumExcluded == 0)
      return false;
    auto End = Excluded.begin() + NumExcluded;
    return std::find(Excluded.begin(), End, Value) != End;
  }
};
} // end anonymous namespace

// Given the provided Value and, potentially, a context instruction, return
// the preferred context instruction (if any).
static const Instruction *safeCxtI(const Value *V, const Instruction *CxtI) {
  // If we've been provided with a context instruction, then use that (provided
  // it has been inserted).
  if (CxtI && CxtI->getParent())
    return CxtI;

  // If the value is really an already-inserted instruction, then use that.
  CxtI = dyn_cast<Instruction>(V);
  if (CxtI && CxtI->getParent())
    return CxtI;

  return nullptr;
}

static void computeKnownBits(const Value *V, APInt &KnownZero, APInt &KnownOne,
                             unsigned Depth, const Query &Q);

void llvm::computeKnownBits(const Value *V, APInt &KnownZero, APInt &KnownOne,
                            const DataLayout &DL, unsigned Depth,
                            AssumptionCache *AC, const Instruction *CxtI,
                            const DominatorTree *DT) {
  ::computeKnownBits(V, KnownZero, KnownOne, Depth,
                     Query(DL, AC, safeCxtI(V, CxtI), DT));
}

bool llvm::haveNoCommonBitsSet(const Value *LHS, const Value *RHS,
                               const DataLayout &DL,
                               AssumptionCache *AC, const Instruction *CxtI,
                               const DominatorTree *DT) {
  assert(LHS->getType() == RHS->getType() &&
         "LHS and RHS should have the same type");
  assert(LHS->getType()->isIntOrIntVectorTy() &&
         "LHS and RHS should be integers");
  IntegerType *IT = cast<IntegerType>(LHS->getType()->getScalarType());
  APInt LHSKnownZero(IT->getBitWidth(), 0), LHSKnownOne(IT->getBitWidth(), 0);
  APInt RHSKnownZero(IT->getBitWidth(), 0), RHSKnownOne(IT->getBitWidth(), 0);
  computeKnownBits(LHS, LHSKnownZero, LHSKnownOne, DL, 0, AC, CxtI, DT);
  computeKnownBits(RHS, RHSKnownZero, RHSKnownOne, DL, 0, AC, CxtI, DT);
  return (LHSKnownZero | RHSKnownZero).isAllOnesValue();
}

static void ComputeSignBit(const Value *V, bool &KnownZero, bool &KnownOne,
                           unsigned Depth, const Query &Q);

void llvm::ComputeSignBit(const Value *V, bool &KnownZero, bool &KnownOne,
                          const DataLayout &DL, unsigned Depth,
                          AssumptionCache *AC, const Instruction *CxtI,
                          const DominatorTree *DT) {
  ::ComputeSignBit(V, KnownZero, KnownOne, Depth,
                   Query(DL, AC, safeCxtI(V, CxtI), DT));
}

static bool isKnownToBeAPowerOfTwo(const Value *V, bool OrZero, unsigned Depth,
                                   const Query &Q);

bool llvm::isKnownToBeAPowerOfTwo(const Value *V, const DataLayout &DL,
                                  bool OrZero,
                                  unsigned Depth, AssumptionCache *AC,
                                  const Instruction *CxtI,
                                  const DominatorTree *DT) {
  return ::isKnownToBeAPowerOfTwo(V, OrZero, Depth,
                                  Query(DL, AC, safeCxtI(V, CxtI), DT));
}

static bool isKnownNonZero(const Value *V, unsigned Depth, const Query &Q);

bool llvm::isKnownNonZero(const Value *V, const DataLayout &DL, unsigned Depth,
                          AssumptionCache *AC, const Instruction *CxtI,
                          const DominatorTree *DT) {
  return ::isKnownNonZero(V, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT));
}

bool llvm::isKnownNonNegative(const Value *V, const DataLayout &DL,
                              unsigned Depth,
                              AssumptionCache *AC, const Instruction *CxtI,
                              const DominatorTree *DT) {
  bool NonNegative, Negative;
  ComputeSignBit(V, NonNegative, Negative, DL, Depth, AC, CxtI, DT);
  return NonNegative;
}

bool llvm::isKnownPositive(const Value *V, const DataLayout &DL, unsigned Depth,
                           AssumptionCache *AC, const Instruction *CxtI,
                           const DominatorTree *DT) {
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return CI->getValue().isStrictlyPositive();

  // TODO: We'd doing two recursive queries here.  We should factor this such
  // that only a single query is needed.
  return isKnownNonNegative(V, DL, Depth, AC, CxtI, DT) &&
    isKnownNonZero(V, DL, Depth, AC, CxtI, DT);
}

bool llvm::isKnownNegative(const Value *V, const DataLayout &DL, unsigned Depth,
                           AssumptionCache *AC, const Instruction *CxtI,
                           const DominatorTree *DT) {
  bool NonNegative, Negative;
  ComputeSignBit(V, NonNegative, Negative, DL, Depth, AC, CxtI, DT);
  return Negative;
}

static bool isKnownNonEqual(const Value *V1, const Value *V2, const Query &Q);

bool llvm::isKnownNonEqual(const Value *V1, const Value *V2,
                           const DataLayout &DL,
                           AssumptionCache *AC, const Instruction *CxtI,
                           const DominatorTree *DT) {
  return ::isKnownNonEqual(V1, V2, Query(DL, AC,
                                         safeCxtI(V1, safeCxtI(V2, CxtI)),
                                         DT));
}

static bool MaskedValueIsZero(const Value *V, const APInt &Mask, unsigned Depth,
                              const Query &Q);

bool llvm::MaskedValueIsZero(const Value *V, const APInt &Mask,
                             const DataLayout &DL,
                             unsigned Depth, AssumptionCache *AC,
                             const Instruction *CxtI, const DominatorTree *DT) {
  return ::MaskedValueIsZero(V, Mask, Depth,
                             Query(DL, AC, safeCxtI(V, CxtI), DT));
}

static unsigned ComputeNumSignBits(const Value *V, unsigned Depth,
                                   const Query &Q);

unsigned llvm::ComputeNumSignBits(const Value *V, const DataLayout &DL,
                                  unsigned Depth, AssumptionCache *AC,
                                  const Instruction *CxtI,
                                  const DominatorTree *DT) {
  return ::ComputeNumSignBits(V, Depth, Query(DL, AC, safeCxtI(V, CxtI), DT));
}

static void computeKnownBitsAddSub(bool Add, const Value *Op0, const Value *Op1,
                                   bool NSW,
                                   APInt &KnownZero, APInt &KnownOne,
                                   APInt &KnownZero2, APInt &KnownOne2,
                                   unsigned Depth, const Query &Q) {
  if (!Add) {
    if (const ConstantInt *CLHS = dyn_cast<ConstantInt>(Op0)) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (!CLHS->getValue().isNegative()) {
        unsigned BitWidth = KnownZero.getBitWidth();
        unsigned NLZ = (CLHS->getValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        computeKnownBits(Op1, KnownZero2, KnownOne2, Depth + 1, Q);

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

  // If an initial sequence of bits in the result is not needed, the
  // corresponding bits in the operands are not needed.
  APInt LHSKnownZero(BitWidth, 0), LHSKnownOne(BitWidth, 0);
  computeKnownBits(Op0, LHSKnownZero, LHSKnownOne, Depth + 1, Q);
  computeKnownBits(Op1, KnownZero2, KnownOne2, Depth + 1, Q);

  // Carry in a 1 for a subtract, rather than a 0.
  APInt CarryIn(BitWidth, 0);
  if (!Add) {
    // Sum = LHS + ~RHS + 1
    std::swap(KnownZero2, KnownOne2);
    CarryIn.setBit(0);
  }

  APInt PossibleSumZero = ~LHSKnownZero + ~KnownZero2 + CarryIn;
  APInt PossibleSumOne = LHSKnownOne + KnownOne2 + CarryIn;

  // Compute known bits of the carry.
  APInt CarryKnownZero = ~(PossibleSumZero ^ LHSKnownZero ^ KnownZero2);
  APInt CarryKnownOne = PossibleSumOne ^ LHSKnownOne ^ KnownOne2;

  // Compute set of known bits (where all three relevant bits are known).
  APInt LHSKnown = LHSKnownZero | LHSKnownOne;
  APInt RHSKnown = KnownZero2 | KnownOne2;
  APInt CarryKnown = CarryKnownZero | CarryKnownOne;
  APInt Known = LHSKnown & RHSKnown & CarryKnown;

  assert((PossibleSumZero & Known) == (PossibleSumOne & Known) &&
         "known bits of sum differ");

  // Compute known bits of the result.
  KnownZero = ~PossibleSumOne & Known;
  KnownOne = PossibleSumOne & Known;

  // Are we still trying to solve for the sign bit?
  if (!Known.isNegative()) {
    if (NSW) {
      // Adding two non-negative numbers, or subtracting a negative number from
      // a non-negative one, can't wrap into negative.
      if (LHSKnownZero.isNegative() && KnownZero2.isNegative())
        KnownZero |= APInt::getSignBit(BitWidth);
      // Adding two negative numbers, or subtracting a non-negative number from
      // a negative one, can't wrap into non-negative.
      else if (LHSKnownOne.isNegative() && KnownOne2.isNegative())
        KnownOne |= APInt::getSignBit(BitWidth);
    }
  }
}

static void computeKnownBitsMul(const Value *Op0, const Value *Op1, bool NSW,
                                APInt &KnownZero, APInt &KnownOne,
                                APInt &KnownZero2, APInt &KnownOne2,
                                unsigned Depth, const Query &Q) {
  unsigned BitWidth = KnownZero.getBitWidth();
  computeKnownBits(Op1, KnownZero, KnownOne, Depth + 1, Q);
  computeKnownBits(Op0, KnownZero2, KnownOne2, Depth + 1, Q);

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
                           isKnownNonZero(Op0, Depth, Q)) ||
                          (isKnownNegativeOp0 && isKnownNonNegativeOp1 &&
                           isKnownNonZero(Op1, Depth, Q));
    }
  }

  // If low bits are zero in either operand, output low known-0 bits.
  // Also compute a conservative estimate for high known-0 bits.
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

void llvm::computeKnownBitsFromRangeMetadata(const MDNode &Ranges,
                                             APInt &KnownZero,
                                             APInt &KnownOne) {
  unsigned BitWidth = KnownZero.getBitWidth();
  unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1);

  KnownZero.setAllBits();
  KnownOne.setAllBits();

  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());

    // The first CommonPrefixBits of all values in Range are equal.
    unsigned CommonPrefixBits =
        (Range.getUnsignedMax() ^ Range.getUnsignedMin()).countLeadingZeros();

    APInt Mask = APInt::getHighBitsSet(BitWidth, CommonPrefixBits);
    KnownOne &= Range.getUnsignedMax() & Mask;
    KnownZero &= ~Range.getUnsignedMax() & Mask;
  }
}

static bool isEphemeralValueOf(const Instruction *I, const Value *E) {
  SmallVector<const Value *, 16> WorkSet(1, I);
  SmallPtrSet<const Value *, 32> Visited;
  SmallPtrSet<const Value *, 16> EphValues;

  // The instruction defining an assumption's condition itself is always
  // considered ephemeral to that assumption (even if it has other
  // non-ephemeral users). See r246696's test case for an example.
  if (is_contained(I->operands(), E))
    return true;

  while (!WorkSet.empty()) {
    const Value *V = WorkSet.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    // If all uses of this value are ephemeral, then so is this value.
    if (all_of(V->users(), [&](const User *U) { return EphValues.count(U); })) {
      if (V == E)
        return true;

      EphValues.insert(V);
      if (const User *U = dyn_cast<User>(V))
        for (User::const_op_iterator J = U->op_begin(), JE = U->op_end();
             J != JE; ++J) {
          if (isSafeToSpeculativelyExecute(*J))
            WorkSet.push_back(*J);
        }
    }
  }

  return false;
}

// Is this an intrinsic that cannot be speculated but also cannot trap?
static bool isAssumeLikeIntrinsic(const Instruction *I) {
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      switch (F->getIntrinsicID()) {
      default: break;
      // FIXME: This list is repeated from NoTTI::getIntrinsicCost.
      case Intrinsic::assume:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::objectsize:
      case Intrinsic::ptr_annotation:
      case Intrinsic::var_annotation:
        return true;
      }

  return false;
}

bool llvm::isValidAssumeForContext(const Instruction *Inv,
                                   const Instruction *CxtI,
                                   const DominatorTree *DT) {

  // There are two restrictions on the use of an assume:
  //  1. The assume must dominate the context (or the control flow must
  //     reach the assume whenever it reaches the context).
  //  2. The context must not be in the assume's set of ephemeral values
  //     (otherwise we will use the assume to prove that the condition
  //     feeding the assume is trivially true, thus causing the removal of
  //     the assume).

  if (DT) {
    if (DT->dominates(Inv, CxtI))
      return true;
  } else if (Inv->getParent() == CxtI->getParent()->getSinglePredecessor()) {
    // We don't have a DT, but this trivially dominates.
    return true;
  }

  // With or without a DT, the only remaining case we will check is if the
  // instructions are in the same BB.  Give up if that is not the case.
  if (Inv->getParent() != CxtI->getParent())
    return false;

  // If we have a dom tree, then we now know that the assume doens't dominate
  // the other instruction.  If we don't have a dom tree then we can check if
  // the assume is first in the BB.
  if (!DT) {
    // Search forward from the assume until we reach the context (or the end
    // of the block); the common case is that the assume will come first.
    for (auto I = std::next(BasicBlock::const_iterator(Inv)),
         IE = Inv->getParent()->end(); I != IE; ++I)
      if (&*I == CxtI)
        return true;
  }

  // The context comes first, but they're both in the same block. Make sure
  // there is nothing in between that might interrupt the control flow.
  for (BasicBlock::const_iterator I =
         std::next(BasicBlock::const_iterator(CxtI)), IE(Inv);
       I != IE; ++I)
    if (!isSafeToSpeculativelyExecute(&*I) && !isAssumeLikeIntrinsic(&*I))
      return false;

  return !isEphemeralValueOf(Inv, CxtI);
}

static void computeKnownBitsFromAssume(const Value *V, APInt &KnownZero,
                                       APInt &KnownOne, unsigned Depth,
                                       const Query &Q) {
  // Use of assumptions is context-sensitive. If we don't have a context, we
  // cannot use them!
  if (!Q.AC || !Q.CxtI)
    return;

  unsigned BitWidth = KnownZero.getBitWidth();

  for (auto &AssumeVH : Q.AC->assumptions()) {
    if (!AssumeVH)
      continue;
    CallInst *I = cast<CallInst>(AssumeVH);
    assert(I->getParent()->getParent() == Q.CxtI->getParent()->getParent() &&
           "Got assumption for the wrong function!");
    if (Q.isExcluded(I))
      continue;

    // Warning: This loop can end up being somewhat performance sensetive.
    // We're running this loop for once for each value queried resulting in a
    // runtime of ~O(#assumes * #values).

    assert(I->getCalledFunction()->getIntrinsicID() == Intrinsic::assume &&
           "must be an assume intrinsic");

    Value *Arg = I->getArgOperand(0);

    if (Arg == V && isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      assert(BitWidth == 1 && "assume operand is not i1?");
      KnownZero.clearAllBits();
      KnownOne.setAllBits();
      return;
    }

    // The remaining tests are all recursive, so bail out if we hit the limit.
    if (Depth == MaxDepth)
      continue;

    Value *A, *B;
    auto m_V = m_CombineOr(m_Specific(V),
                           m_CombineOr(m_PtrToInt(m_Specific(V)),
                           m_BitCast(m_Specific(V))));

    CmpInst::Predicate Pred;
    ConstantInt *C;
    // assume(v = a)
    if (match(Arg, m_c_ICmp(Pred, m_V, m_Value(A))) &&
        Pred == ICmpInst::ICMP_EQ && isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      KnownZero |= RHSKnownZero;
      KnownOne  |= RHSKnownOne;
    // assume(v & b = a)
    } else if (match(Arg,
                     m_c_ICmp(Pred, m_c_And(m_V, m_Value(B)), m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt MaskKnownZero(BitWidth, 0), MaskKnownOne(BitWidth, 0);
      computeKnownBits(B, MaskKnownZero, MaskKnownOne, Depth+1, Query(Q, I));

      // For those bits in the mask that are known to be one, we can propagate
      // known bits from the RHS to V.
      KnownZero |= RHSKnownZero & MaskKnownOne;
      KnownOne  |= RHSKnownOne  & MaskKnownOne;
    // assume(~(v & b) = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Not(m_c_And(m_V, m_Value(B))),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt MaskKnownZero(BitWidth, 0), MaskKnownOne(BitWidth, 0);
      computeKnownBits(B, MaskKnownZero, MaskKnownOne, Depth+1, Query(Q, I));

      // For those bits in the mask that are known to be one, we can propagate
      // inverted known bits from the RHS to V.
      KnownZero |= RHSKnownOne  & MaskKnownOne;
      KnownOne  |= RHSKnownZero & MaskKnownOne;
    // assume(v | b = a)
    } else if (match(Arg,
                     m_c_ICmp(Pred, m_c_Or(m_V, m_Value(B)), m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt BKnownZero(BitWidth, 0), BKnownOne(BitWidth, 0);
      computeKnownBits(B, BKnownZero, BKnownOne, Depth+1, Query(Q, I));

      // For those bits in B that are known to be zero, we can propagate known
      // bits from the RHS to V.
      KnownZero |= RHSKnownZero & BKnownZero;
      KnownOne  |= RHSKnownOne  & BKnownZero;
    // assume(~(v | b) = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Not(m_c_Or(m_V, m_Value(B))),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt BKnownZero(BitWidth, 0), BKnownOne(BitWidth, 0);
      computeKnownBits(B, BKnownZero, BKnownOne, Depth+1, Query(Q, I));

      // For those bits in B that are known to be zero, we can propagate
      // inverted known bits from the RHS to V.
      KnownZero |= RHSKnownOne  & BKnownZero;
      KnownOne  |= RHSKnownZero & BKnownZero;
    // assume(v ^ b = a)
    } else if (match(Arg,
                     m_c_ICmp(Pred, m_c_Xor(m_V, m_Value(B)), m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt BKnownZero(BitWidth, 0), BKnownOne(BitWidth, 0);
      computeKnownBits(B, BKnownZero, BKnownOne, Depth+1, Query(Q, I));

      // For those bits in B that are known to be zero, we can propagate known
      // bits from the RHS to V. For those bits in B that are known to be one,
      // we can propagate inverted known bits from the RHS to V.
      KnownZero |= RHSKnownZero & BKnownZero;
      KnownOne  |= RHSKnownOne  & BKnownZero;
      KnownZero |= RHSKnownOne  & BKnownOne;
      KnownOne  |= RHSKnownZero & BKnownOne;
    // assume(~(v ^ b) = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Not(m_c_Xor(m_V, m_Value(B))),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      APInt BKnownZero(BitWidth, 0), BKnownOne(BitWidth, 0);
      computeKnownBits(B, BKnownZero, BKnownOne, Depth+1, Query(Q, I));

      // For those bits in B that are known to be zero, we can propagate
      // inverted known bits from the RHS to V. For those bits in B that are
      // known to be one, we can propagate known bits from the RHS to V.
      KnownZero |= RHSKnownOne  & BKnownZero;
      KnownOne  |= RHSKnownZero & BKnownZero;
      KnownZero |= RHSKnownZero & BKnownOne;
      KnownOne  |= RHSKnownOne  & BKnownOne;
    // assume(v << c = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Shl(m_V, m_ConstantInt(C)),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      // For those bits in RHS that are known, we can propagate them to known
      // bits in V shifted to the right by C.
      KnownZero |= RHSKnownZero.lshr(C->getZExtValue());
      KnownOne  |= RHSKnownOne.lshr(C->getZExtValue());
    // assume(~(v << c) = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Not(m_Shl(m_V, m_ConstantInt(C))),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      // For those bits in RHS that are known, we can propagate them inverted
      // to known bits in V shifted to the right by C.
      KnownZero |= RHSKnownOne.lshr(C->getZExtValue());
      KnownOne  |= RHSKnownZero.lshr(C->getZExtValue());
    // assume(v >> c = a)
    } else if (match(Arg,
                     m_c_ICmp(Pred, m_CombineOr(m_LShr(m_V, m_ConstantInt(C)),
                                                m_AShr(m_V, m_ConstantInt(C))),
                              m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      // For those bits in RHS that are known, we can propagate them to known
      // bits in V shifted to the right by C.
      KnownZero |= RHSKnownZero << C->getZExtValue();
      KnownOne  |= RHSKnownOne  << C->getZExtValue();
    // assume(~(v >> c) = a)
    } else if (match(Arg, m_c_ICmp(Pred, m_Not(m_CombineOr(
                                             m_LShr(m_V, m_ConstantInt(C)),
                                             m_AShr(m_V, m_ConstantInt(C)))),
                                   m_Value(A))) &&
               Pred == ICmpInst::ICMP_EQ &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));
      // For those bits in RHS that are known, we can propagate them inverted
      // to known bits in V shifted to the right by C.
      KnownZero |= RHSKnownOne  << C->getZExtValue();
      KnownOne  |= RHSKnownZero << C->getZExtValue();
    // assume(v >=_s c) where c is non-negative
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_SGE &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      if (RHSKnownZero.isNegative()) {
        // We know that the sign bit is zero.
        KnownZero |= APInt::getSignBit(BitWidth);
      }
    // assume(v >_s c) where c is at least -1.
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_SGT &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      if (RHSKnownOne.isAllOnesValue() || RHSKnownZero.isNegative()) {
        // We know that the sign bit is zero.
        KnownZero |= APInt::getSignBit(BitWidth);
      }
    // assume(v <=_s c) where c is negative
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_SLE &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      if (RHSKnownOne.isNegative()) {
        // We know that the sign bit is one.
        KnownOne |= APInt::getSignBit(BitWidth);
      }
    // assume(v <_s c) where c is non-positive
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_SLT &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      if (RHSKnownZero.isAllOnesValue() || RHSKnownOne.isNegative()) {
        // We know that the sign bit is one.
        KnownOne |= APInt::getSignBit(BitWidth);
      }
    // assume(v <=_u c)
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_ULE &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      // Whatever high bits in c are zero are known to be zero.
      KnownZero |=
        APInt::getHighBitsSet(BitWidth, RHSKnownZero.countLeadingOnes());
    // assume(v <_u c)
    } else if (match(Arg, m_ICmp(Pred, m_V, m_Value(A))) &&
               Pred == ICmpInst::ICMP_ULT &&
               isValidAssumeForContext(I, Q.CxtI, Q.DT)) {
      APInt RHSKnownZero(BitWidth, 0), RHSKnownOne(BitWidth, 0);
      computeKnownBits(A, RHSKnownZero, RHSKnownOne, Depth+1, Query(Q, I));

      // Whatever high bits in c are zero are known to be zero (if c is a power
      // of 2, then one more).
      if (isKnownToBeAPowerOfTwo(A, false, Depth + 1, Query(Q, I)))
        KnownZero |=
          APInt::getHighBitsSet(BitWidth, RHSKnownZero.countLeadingOnes()+1);
      else
        KnownZero |=
          APInt::getHighBitsSet(BitWidth, RHSKnownZero.countLeadingOnes());
    }
  }
}

// Compute known bits from a shift operator, including those with a
// non-constant shift amount. KnownZero and KnownOne are the outputs of this
// function. KnownZero2 and KnownOne2 are pre-allocated temporaries with the
// same bit width as KnownZero and KnownOne. KZF and KOF are operator-specific
// functors that, given the known-zero or known-one bits respectively, and a
// shift amount, compute the implied known-zero or known-one bits of the shift
// operator's result respectively for that shift amount. The results from calling
// KZF and KOF are conservatively combined for all permitted shift amounts.
static void computeKnownBitsFromShiftOperator(
    const Operator *I, APInt &KnownZero, APInt &KnownOne, APInt &KnownZero2,
    APInt &KnownOne2, unsigned Depth, const Query &Q,
    function_ref<APInt(const APInt &, unsigned)> KZF,
    function_ref<APInt(const APInt &, unsigned)> KOF) {
  unsigned BitWidth = KnownZero.getBitWidth();

  if (auto *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
    unsigned ShiftAmt = SA->getLimitedValue(BitWidth-1);

    computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
    KnownZero = KZF(KnownZero, ShiftAmt);
    KnownOne  = KOF(KnownOne, ShiftAmt);
    // If there is conflict between KnownZero and KnownOne, this must be an
    // overflowing left shift, so the shift result is undefined. Clear KnownZero
    // and KnownOne bits so that other code could propagate this undef.
    if ((KnownZero & KnownOne) != 0) {
      KnownZero.clearAllBits();
      KnownOne.clearAllBits();
    }

    return;
  }

  computeKnownBits(I->getOperand(1), KnownZero, KnownOne, Depth + 1, Q);

  // Note: We cannot use KnownZero.getLimitedValue() here, because if
  // BitWidth > 64 and any upper bits are known, we'll end up returning the
  // limit value (which implies all bits are known).
  uint64_t ShiftAmtKZ = KnownZero.zextOrTrunc(64).getZExtValue();
  uint64_t ShiftAmtKO = KnownOne.zextOrTrunc(64).getZExtValue();

  // It would be more-clearly correct to use the two temporaries for this
  // calculation. Reusing the APInts here to prevent unnecessary allocations.
  KnownZero.clearAllBits();
  KnownOne.clearAllBits();

  // If we know the shifter operand is nonzero, we can sometimes infer more
  // known bits. However this is expensive to compute, so be lazy about it and
  // only compute it when absolutely necessary.
  Optional<bool> ShifterOperandIsNonZero;

  // Early exit if we can't constrain any well-defined shift amount.
  if (!(ShiftAmtKZ & (BitWidth - 1)) && !(ShiftAmtKO & (BitWidth - 1))) {
    ShifterOperandIsNonZero =
        isKnownNonZero(I->getOperand(1), Depth + 1, Q);
    if (!*ShifterOperandIsNonZero)
      return;
  }

  computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);

  KnownZero = KnownOne = APInt::getAllOnesValue(BitWidth);
  for (unsigned ShiftAmt = 0; ShiftAmt < BitWidth; ++ShiftAmt) {
    // Combine the shifted known input bits only for those shift amounts
    // compatible with its known constraints.
    if ((ShiftAmt & ~ShiftAmtKZ) != ShiftAmt)
      continue;
    if ((ShiftAmt | ShiftAmtKO) != ShiftAmt)
      continue;
    // If we know the shifter is nonzero, we may be able to infer more known
    // bits. This check is sunk down as far as possible to avoid the expensive
    // call to isKnownNonZero if the cheaper checks above fail.
    if (ShiftAmt == 0) {
      if (!ShifterOperandIsNonZero.hasValue())
        ShifterOperandIsNonZero =
            isKnownNonZero(I->getOperand(1), Depth + 1, Q);
      if (*ShifterOperandIsNonZero)
        continue;
    }

    KnownZero &= KZF(KnownZero2, ShiftAmt);
    KnownOne  &= KOF(KnownOne2, ShiftAmt);
  }

  // If there are no compatible shift amounts, then we've proven that the shift
  // amount must be >= the BitWidth, and the result is undefined. We could
  // return anything we'd like, but we need to make sure the sets of known bits
  // stay disjoint (it should be better for some other code to actually
  // propagate the undef than to pick a value here using known bits).
  if ((KnownZero & KnownOne) != 0) {
    KnownZero.clearAllBits();
    KnownOne.clearAllBits();
  }
}

static void computeKnownBitsFromOperator(const Operator *I, APInt &KnownZero,
                                         APInt &KnownOne, unsigned Depth,
                                         const Query &Q) {
  unsigned BitWidth = KnownZero.getBitWidth();

  APInt KnownZero2(KnownZero), KnownOne2(KnownOne);
  switch (I->getOpcode()) {
  default: break;
  case Instruction::Load:
    if (MDNode *MD = cast<LoadInst>(I)->getMetadata(LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, KnownZero, KnownOne);
    break;
  case Instruction::And: {
    // If either the LHS or the RHS are Zero, the result is zero.
    computeKnownBits(I->getOperand(1), KnownZero, KnownOne, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;

    // and(x, add (x, -1)) is a common idiom that always clears the low bit;
    // here we handle the more general case of adding any odd number by
    // matching the form add(x, add(x, y)) where y is odd.
    // TODO: This could be generalized to clearing any bit set in y where the
    // following bit is known to be unset in y.
    Value *Y = nullptr;
    if (match(I->getOperand(0), m_Add(m_Specific(I->getOperand(1)),
                                      m_Value(Y))) ||
        match(I->getOperand(1), m_Add(m_Specific(I->getOperand(0)),
                                      m_Value(Y)))) {
      APInt KnownZero3(BitWidth, 0), KnownOne3(BitWidth, 0);
      computeKnownBits(Y, KnownZero3, KnownOne3, Depth + 1, Q);
      if (KnownOne3.countTrailingOnes() > 0)
        KnownZero |= APInt::getLowBitsSet(BitWidth, 1);
    }
    break;
  }
  case Instruction::Or: {
    computeKnownBits(I->getOperand(1), KnownZero, KnownOne, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  }
  case Instruction::Xor: {
    computeKnownBits(I->getOperand(1), KnownZero, KnownOne, Depth + 1, Q);
    computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    break;
  }
  case Instruction::Mul: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    computeKnownBitsMul(I->getOperand(0), I->getOperand(1), NSW, KnownZero,
                        KnownOne, KnownZero2, KnownOne2, Depth, Q);
    break;
  }
  case Instruction::UDiv: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clearAllBits();
    KnownZero2.clearAllBits();
    computeKnownBits(I->getOperand(1), KnownZero2, KnownOne2, Depth + 1, Q);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ);
    break;
  }
  case Instruction::Select: {
    computeKnownBits(I->getOperand(2), KnownZero, KnownOne, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), KnownZero2, KnownOne2, Depth + 1, Q);

    const Value *LHS;
    const Value *RHS;
    SelectPatternFlavor SPF = matchSelectPattern(I, LHS, RHS).Flavor;
    if (SelectPatternResult::isMinOrMax(SPF)) {
      computeKnownBits(RHS, KnownZero, KnownOne, Depth + 1, Q);
      computeKnownBits(LHS, KnownZero2, KnownOne2, Depth + 1, Q);
    } else {
      computeKnownBits(I->getOperand(2), KnownZero, KnownOne, Depth + 1, Q);
      computeKnownBits(I->getOperand(1), KnownZero2, KnownOne2, Depth + 1, Q);
    }

    unsigned MaxHighOnes = 0;
    unsigned MaxHighZeros = 0;
    if (SPF == SPF_SMAX) {
      // If both sides are negative, the result is negative.
      if (KnownOne[BitWidth - 1] && KnownOne2[BitWidth - 1])
        // We can derive a lower bound on the result by taking the max of the
        // leading one bits.
        MaxHighOnes =
            std::max(KnownOne.countLeadingOnes(), KnownOne2.countLeadingOnes());
      // If either side is non-negative, the result is non-negative.
      else if (KnownZero[BitWidth - 1] || KnownZero2[BitWidth - 1])
        MaxHighZeros = 1;
    } else if (SPF == SPF_SMIN) {
      // If both sides are non-negative, the result is non-negative.
      if (KnownZero[BitWidth - 1] && KnownZero2[BitWidth - 1])
        // We can derive an upper bound on the result by taking the max of the
        // leading zero bits.
        MaxHighZeros = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
      // If either side is negative, the result is negative.
      else if (KnownOne[BitWidth - 1] || KnownOne2[BitWidth - 1])
        MaxHighOnes = 1;
    } else if (SPF == SPF_UMAX) {
      // We can derive a lower bound on the result by taking the max of the
      // leading one bits.
      MaxHighOnes =
          std::max(KnownOne.countLeadingOnes(), KnownOne2.countLeadingOnes());
    } else if (SPF == SPF_UMIN) {
      // We can derive an upper bound on the result by taking the max of the
      // leading zero bits.
      MaxHighZeros =
          std::max(KnownZero.countLeadingOnes(), KnownZero2.countLeadingOnes());
    }

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    if (MaxHighOnes > 0)
      KnownOne |= APInt::getHighBitsSet(BitWidth, MaxHighOnes);
    if (MaxHighZeros > 0)
      KnownZero |= APInt::getHighBitsSet(BitWidth, MaxHighZeros);
    break;
  }
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    break; // Can't work with floating point.
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::AddrSpaceCast: // Pointers could be different sizes.
    // Fall through and handle them the same as zext/trunc.
    LLVM_FALLTHROUGH;
  case Instruction::ZExt:
  case Instruction::Trunc: {
    Type *SrcTy = I->getOperand(0)->getType();

    unsigned SrcBitWidth;
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    SrcBitWidth = Q.DL.getTypeSizeInBits(SrcTy->getScalarType());

    assert(SrcBitWidth && "SrcBitWidth can't be zero");
    KnownZero = KnownZero.zextOrTrunc(SrcBitWidth);
    KnownOne = KnownOne.zextOrTrunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
    KnownZero = KnownZero.zextOrTrunc(BitWidth);
    KnownOne = KnownOne.zextOrTrunc(BitWidth);
    // Any top bits are known to be zero.
    if (BitWidth > SrcBitWidth)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    break;
  }
  case Instruction::BitCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    if ((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
        // TODO: For now, not handling conversions like:
        // (bitcast i64 %x to <2 x i32>)
        !I->getType()->isVectorTy()) {
      computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
      break;
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    unsigned SrcBitWidth = I->getOperand(0)->getType()->getScalarSizeInBits();

    KnownZero = KnownZero.trunc(SrcBitWidth);
    KnownOne = KnownOne.trunc(SrcBitWidth);
    computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero[SrcBitWidth-1])             // Input sign bit known zero
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    else if (KnownOne[SrcBitWidth-1])           // Input sign bit known set
      KnownOne |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    break;
  }
  case Instruction::Shl: {
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    auto KZF = [BitWidth, NSW](const APInt &KnownZero, unsigned ShiftAmt) {
      APInt KZResult =
          (KnownZero << ShiftAmt) |
          APInt::getLowBitsSet(BitWidth, ShiftAmt); // Low bits known 0.
      // If this shift has "nsw" keyword, then the result is either a poison
      // value or has the same sign bit as the first operand.
      if (NSW && KnownZero.isNegative())
        KZResult.setBit(BitWidth - 1);
      return KZResult;
    };

    auto KOF = [BitWidth, NSW](const APInt &KnownOne, unsigned ShiftAmt) {
      APInt KOResult = KnownOne << ShiftAmt;
      if (NSW && KnownOne.isNegative())
        KOResult.setBit(BitWidth - 1);
      return KOResult;
    };

    computeKnownBitsFromShiftOperator(I, KnownZero, KnownOne,
                                      KnownZero2, KnownOne2, Depth, Q, KZF,
                                      KOF);
    break;
  }
  case Instruction::LShr: {
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    auto KZF = [BitWidth](const APInt &KnownZero, unsigned ShiftAmt) {
      return APIntOps::lshr(KnownZero, ShiftAmt) |
             // High bits known zero.
             APInt::getHighBitsSet(BitWidth, ShiftAmt);
    };

    auto KOF = [BitWidth](const APInt &KnownOne, unsigned ShiftAmt) {
      return APIntOps::lshr(KnownOne, ShiftAmt);
    };

    computeKnownBitsFromShiftOperator(I, KnownZero, KnownOne,
                                      KnownZero2, KnownOne2, Depth, Q, KZF,
                                      KOF);
    break;
  }
  case Instruction::AShr: {
    // (ashr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    auto KZF = [BitWidth](const APInt &KnownZero, unsigned ShiftAmt) {
      return APIntOps::ashr(KnownZero, ShiftAmt);
    };

    auto KOF = [BitWidth](const APInt &KnownOne, unsigned ShiftAmt) {
      return APIntOps::ashr(KnownOne, ShiftAmt);
    };

    computeKnownBitsFromShiftOperator(I, KnownZero, KnownOne,
                                      KnownZero2, KnownOne2, Depth, Q, KZF,
                                      KOF);
    break;
  }
  case Instruction::Sub: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    computeKnownBitsAddSub(false, I->getOperand(0), I->getOperand(1), NSW,
                           KnownZero, KnownOne, KnownZero2, KnownOne2, Depth,
                           Q);
    break;
  }
  case Instruction::Add: {
    bool NSW = cast<OverflowingBinaryOperator>(I)->hasNoSignedWrap();
    computeKnownBitsAddSub(true, I->getOperand(0), I->getOperand(1), NSW,
                           KnownZero, KnownOne, KnownZero2, KnownOne2, Depth,
                           Q);
    break;
  }
  case Instruction::SRem:
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue().abs();
      if (RA.isPowerOf2()) {
        APInt LowBits = RA - 1;
        computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1,
                         Q);

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
      computeKnownBits(I->getOperand(0), LHSKnownZero, LHSKnownOne, Depth + 1,
                       Q);
      // If it's known zero, our sign bit is also zero.
      if (LHSKnownZero.isNegative())
        KnownZero.setBit(BitWidth - 1);
    }

    break;
  case Instruction::URem: {
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      const APInt &RA = Rem->getValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
        KnownZero |= ~LowBits;
        KnownOne &= LowBits;
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    computeKnownBits(I->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);
    computeKnownBits(I->getOperand(1), KnownZero2, KnownOne2, Depth + 1, Q);

    unsigned Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clearAllBits();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders);
    break;
  }

  case Instruction::Alloca: {
    const AllocaInst *AI = cast<AllocaInst>(I);
    unsigned Align = AI->getAlignment();
    if (Align == 0)
      Align = Q.DL.getABITypeAlignment(AI->getAllocatedType());

    if (Align > 0)
      KnownZero = APInt::getLowBitsSet(BitWidth, countTrailingZeros(Align));
    break;
  }
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    APInt LocalKnownZero(BitWidth, 0), LocalKnownOne(BitWidth, 0);
    computeKnownBits(I->getOperand(0), LocalKnownZero, LocalKnownOne, Depth + 1,
                     Q);
    unsigned TrailZ = LocalKnownZero.countTrailingOnes();

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      Value *Index = I->getOperand(i);
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        // Handle struct member offset arithmetic.

        // Handle case when index is vector zeroinitializer
        Constant *CIndex = cast<Constant>(Index);
        if (CIndex->isZeroValue())
          continue;

        if (CIndex->getType()->isVectorTy())
          Index = CIndex->getSplatValue();

        unsigned Idx = cast<ConstantInt>(Index)->getZExtValue();
        const StructLayout *SL = Q.DL.getStructLayout(STy);
        uint64_t Offset = SL->getElementOffset(Idx);
        TrailZ = std::min<unsigned>(TrailZ,
                                    countTrailingZeros(Offset));
      } else {
        // Handle array index arithmetic.
        Type *IndexedTy = GTI.getIndexedType();
        if (!IndexedTy->isSized()) {
          TrailZ = 0;
          break;
        }
        unsigned GEPOpiBits = Index->getType()->getScalarSizeInBits();
        uint64_t TypeSize = Q.DL.getTypeAllocSize(IndexedTy);
        LocalKnownZero = LocalKnownOne = APInt(GEPOpiBits, 0);
        computeKnownBits(Index, LocalKnownZero, LocalKnownOne, Depth + 1, Q);
        TrailZ = std::min(TrailZ,
                          unsigned(countTrailingZeros(TypeSize) +
                                   LocalKnownZero.countTrailingOnes()));
      }
    }

    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ);
    break;
  }
  case Instruction::PHI: {
    const PHINode *P = cast<PHINode>(I);
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
          computeKnownBits(R, KnownZero2, KnownOne2, Depth + 1, Q);

          // We need to take the minimum number of known bits
          APInt KnownZero3(KnownZero), KnownOne3(KnownOne);
          computeKnownBits(L, KnownZero3, KnownOne3, Depth + 1, Q);

          KnownZero = APInt::getLowBitsSet(BitWidth,
                                           std::min(KnownZero2.countTrailingOnes(),
                                                    KnownZero3.countTrailingOnes()));
          break;
        }
      }
    }

    // Unreachable blocks may have zero-operand PHI nodes.
    if (P->getNumIncomingValues() == 0)
      break;

    // Otherwise take the unions of the known bit sets of the operands,
    // taking conservative care to avoid excessive recursion.
    if (Depth < MaxDepth - 1 && !KnownZero && !KnownOne) {
      // Skip if every incoming value references to ourself.
      if (dyn_cast_or_null<UndefValue>(P->hasConstantValue()))
        break;

      KnownZero = APInt::getAllOnesValue(BitWidth);
      KnownOne = APInt::getAllOnesValue(BitWidth);
      for (Value *IncValue : P->incoming_values()) {
        // Skip direct self references.
        if (IncValue == P) continue;

        KnownZero2 = APInt(BitWidth, 0);
        KnownOne2 = APInt(BitWidth, 0);
        // Recurse, but cap the recursion to one level, because we don't
        // want to waste time spinning around in loops.
        computeKnownBits(IncValue, KnownZero2, KnownOne2, MaxDepth - 1, Q);
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
  case Instruction::Invoke:
    // If range metadata is attached to this call, set known bits from that,
    // and then intersect with known bits based on other properties of the
    // function.
    if (MDNode *MD = cast<Instruction>(I)->getMetadata(LLVMContext::MD_range))
      computeKnownBitsFromRangeMetadata(*MD, KnownZero, KnownOne);
    if (const Value *RV = ImmutableCallSite(I).getReturnedArgOperand()) {
      computeKnownBits(RV, KnownZero2, KnownOne2, Depth + 1, Q);
      KnownZero |= KnownZero2;
      KnownOne |= KnownOne2;
    }
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::bswap:
        computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);
        KnownZero |= KnownZero2.byteSwap();
        KnownOne |= KnownOne2.byteSwap();
        break;
      case Intrinsic::ctlz:
      case Intrinsic::cttz: {
        unsigned LowBits = Log2_32(BitWidth)+1;
        // If this call is undefined for 0, the result will be less than 2^n.
        if (II->getArgOperand(1) == ConstantInt::getTrue(II->getContext()))
          LowBits -= 1;
        KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
        break;
      }
      case Intrinsic::ctpop: {
        computeKnownBits(I->getOperand(0), KnownZero2, KnownOne2, Depth + 1, Q);
        // We can bound the space the count needs.  Also, bits known to be zero
        // can't contribute to the population.
        unsigned BitsPossiblySet = BitWidth - KnownZero2.countPopulation();
        unsigned LeadingZeros =
          APInt(BitWidth, BitsPossiblySet).countLeadingZeros();
        assert(LeadingZeros <= BitWidth);
        KnownZero |= APInt::getHighBitsSet(BitWidth, LeadingZeros);
        KnownOne &= ~KnownZero;
        // TODO: we could bound KnownOne using the lower bound on the number
        // of bits which might be set provided by popcnt KnownOne2.
        break;
      }
      case Intrinsic::x86_sse42_crc32_64_64:
        KnownZero |= APInt::getHighBitsSet(64, 32);
        break;
      }
    }
    break;
  case Instruction::ExtractValue:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I->getOperand(0))) {
      const ExtractValueInst *EVI = cast<ExtractValueInst>(I);
      if (EVI->getNumIndices() != 1) break;
      if (EVI->getIndices()[0] == 0) {
        switch (II->getIntrinsicID()) {
        default: break;
        case Intrinsic::uadd_with_overflow:
        case Intrinsic::sadd_with_overflow:
          computeKnownBitsAddSub(true, II->getArgOperand(0),
                                 II->getArgOperand(1), false, KnownZero,
                                 KnownOne, KnownZero2, KnownOne2, Depth, Q);
          break;
        case Intrinsic::usub_with_overflow:
        case Intrinsic::ssub_with_overflow:
          computeKnownBitsAddSub(false, II->getArgOperand(0),
                                 II->getArgOperand(1), false, KnownZero,
                                 KnownOne, KnownZero2, KnownOne2, Depth, Q);
          break;
        case Intrinsic::umul_with_overflow:
        case Intrinsic::smul_with_overflow:
          computeKnownBitsMul(II->getArgOperand(0), II->getArgOperand(1), false,
                              KnownZero, KnownOne, KnownZero2, KnownOne2, Depth,
                              Q);
          break;
        }
      }
    }
  }
}

/// Determine which bits of V are known to be either zero or one and return
/// them in the KnownZero/KnownOne bit sets.
///
/// NOTE: we cannot consider 'undef' to be "IsZero" here.  The problem is that
/// we cannot optimize based on the assumption that it is zero without changing
/// it to be an explicit zero.  If we don't change it to zero, other code could
/// optimized based on the contradictory assumption that it is non-zero.
/// Because instcombine aggressively folds operations with undef args anyway,
/// this won't lose us code quality.
///
/// This function is defined on values with integer type, values with pointer
/// type, and vectors of integers.  In the case
/// where V is a vector, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the elements in the vector.
void computeKnownBits(const Value *V, APInt &KnownZero, APInt &KnownOne,
                      unsigned Depth, const Query &Q) {
  assert(V && "No Value?");
  assert(Depth <= MaxDepth && "Limit Search Depth");
  unsigned BitWidth = KnownZero.getBitWidth();

  assert((V->getType()->isIntOrIntVectorTy() ||
          V->getType()->getScalarType()->isPointerTy()) &&
         "Not integer or pointer type!");
  assert((Q.DL.getTypeSizeInBits(V->getType()->getScalarType()) == BitWidth) &&
         (!V->getType()->isIntOrIntVectorTy() ||
          V->getType()->getScalarSizeInBits() == BitWidth) &&
         KnownZero.getBitWidth() == BitWidth &&
         KnownOne.getBitWidth() == BitWidth &&
         "V, KnownOne and KnownZero should have same BitWidth");

  const APInt *C;
  if (match(V, m_APInt(C))) {
    // We know all of the bits for a scalar constant or a splat vector constant!
    KnownOne = *C;
    KnownZero = ~KnownOne;
    return;
  }
  // Null and aggregate-zero are all-zeros.
  if (isa<ConstantPointerNull>(V) || isa<ConstantAggregateZero>(V)) {
    KnownOne.clearAllBits();
    KnownZero = APInt::getAllOnesValue(BitWidth);
    return;
  }
  // Handle a constant vector by taking the intersection of the known bits of
  // each element.
  if (const ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(V)) {
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

  if (const auto *CV = dyn_cast<ConstantVector>(V)) {
    // We know that CV must be a vector of integers. Take the intersection of
    // each element.
    KnownZero.setAllBits(); KnownOne.setAllBits();
    APInt Elt(KnownZero.getBitWidth(), 0);
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      Constant *Element = CV->getAggregateElement(i);
      auto *ElementCI = dyn_cast_or_null<ConstantInt>(Element);
      if (!ElementCI) {
        KnownZero.clearAllBits();
        KnownOne.clearAllBits();
        return;
      }
      Elt = ElementCI->getValue();
      KnownZero &= ~Elt;
      KnownOne &= Elt;
    }
    return;
  }

  // Start out not knowing anything.
  KnownZero.clearAllBits(); KnownOne.clearAllBits();

  // We can't imply anything about undefs.
  if (isa<UndefValue>(V))
    return;

  // There's no point in looking through other users of ConstantData for
  // assumptions.  Confirm that we've handled them all.
  assert(!isa<ConstantData>(V) && "Unhandled constant data!");

  // Limit search depth.
  // All recursive calls that increase depth must come after this.
  if (Depth == MaxDepth)
    return;

  // A weak GlobalAlias is totally unknown. A non-weak GlobalAlias has
  // the bits of its aliasee.
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
    if (!GA->isInterposable())
      computeKnownBits(GA->getAliasee(), KnownZero, KnownOne, Depth + 1, Q);
    return;
  }

  if (const Operator *I = dyn_cast<Operator>(V))
    computeKnownBitsFromOperator(I, KnownZero, KnownOne, Depth, Q);

  // Aligned pointers have trailing zeros - refine KnownZero set
  if (V->getType()->isPointerTy()) {
    unsigned Align = V->getPointerAlignment(Q.DL);
    if (Align)
      KnownZero |= APInt::getLowBitsSet(BitWidth, countTrailingZeros(Align));
  }

  // computeKnownBitsFromAssume strictly refines KnownZero and
  // KnownOne. Therefore, we run them after computeKnownBitsFromOperator.

  // Check whether a nearby assume intrinsic can determine some known bits.
  computeKnownBitsFromAssume(V, KnownZero, KnownOne, Depth, Q);

  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
}

/// Determine whether the sign bit is known to be zero or one.
/// Convenience wrapper around computeKnownBits.
void ComputeSignBit(const Value *V, bool &KnownZero, bool &KnownOne,
                    unsigned Depth, const Query &Q) {
  unsigned BitWidth = getBitWidth(V->getType(), Q.DL);
  if (!BitWidth) {
    KnownZero = false;
    KnownOne = false;
    return;
  }
  APInt ZeroBits(BitWidth, 0);
  APInt OneBits(BitWidth, 0);
  computeKnownBits(V, ZeroBits, OneBits, Depth, Q);
  KnownOne = OneBits[BitWidth - 1];
  KnownZero = ZeroBits[BitWidth - 1];
}

/// Return true if the given value is known to have exactly one
/// bit set when defined. For vectors return true if every element is known to
/// be a power of two when defined. Supports values with integer or pointer
/// types and vectors of integers.
bool isKnownToBeAPowerOfTwo(const Value *V, bool OrZero, unsigned Depth,
                            const Query &Q) {
  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return OrZero;

    const APInt *ConstIntOrConstSplatInt;
    if (match(C, m_APInt(ConstIntOrConstSplatInt)))
      return ConstIntOrConstSplatInt->isPowerOf2();
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

  Value *X = nullptr, *Y = nullptr;
  // A shift left or a logical shift right of a power of two is a power of two
  // or zero.
  if (OrZero && (match(V, m_Shl(m_Value(X), m_Value())) ||
                 match(V, m_LShr(m_Value(X), m_Value()))))
    return isKnownToBeAPowerOfTwo(X, /*OrZero*/ true, Depth, Q);

  if (const ZExtInst *ZI = dyn_cast<ZExtInst>(V))
    return isKnownToBeAPowerOfTwo(ZI->getOperand(0), OrZero, Depth, Q);

  if (const SelectInst *SI = dyn_cast<SelectInst>(V))
    return isKnownToBeAPowerOfTwo(SI->getTrueValue(), OrZero, Depth, Q) &&
           isKnownToBeAPowerOfTwo(SI->getFalseValue(), OrZero, Depth, Q);

  if (OrZero && match(V, m_And(m_Value(X), m_Value(Y)))) {
    // A power of two and'd with anything is a power of two or zero.
    if (isKnownToBeAPowerOfTwo(X, /*OrZero*/ true, Depth, Q) ||
        isKnownToBeAPowerOfTwo(Y, /*OrZero*/ true, Depth, Q))
      return true;
    // X & (-X) is always a power of two or zero.
    if (match(X, m_Neg(m_Specific(Y))) || match(Y, m_Neg(m_Specific(X))))
      return true;
    return false;
  }

  // Adding a power-of-two or zero to the same power-of-two or zero yields
  // either the original power-of-two, a larger power-of-two or zero.
  if (match(V, m_Add(m_Value(X), m_Value(Y)))) {
    const OverflowingBinaryOperator *VOBO = cast<OverflowingBinaryOperator>(V);
    if (OrZero || VOBO->hasNoUnsignedWrap() || VOBO->hasNoSignedWrap()) {
      if (match(X, m_And(m_Specific(Y), m_Value())) ||
          match(X, m_And(m_Value(), m_Specific(Y))))
        if (isKnownToBeAPowerOfTwo(Y, OrZero, Depth, Q))
          return true;
      if (match(Y, m_And(m_Specific(X), m_Value())) ||
          match(Y, m_And(m_Value(), m_Specific(X))))
        if (isKnownToBeAPowerOfTwo(X, OrZero, Depth, Q))
          return true;

      unsigned BitWidth = V->getType()->getScalarSizeInBits();
      APInt LHSZeroBits(BitWidth, 0), LHSOneBits(BitWidth, 0);
      computeKnownBits(X, LHSZeroBits, LHSOneBits, Depth, Q);

      APInt RHSZeroBits(BitWidth, 0), RHSOneBits(BitWidth, 0);
      computeKnownBits(Y, RHSZeroBits, RHSOneBits, Depth, Q);
      // If i8 V is a power of two or zero:
      //  ZeroBits: 1 1 1 0 1 1 1 1
      // ~ZeroBits: 0 0 0 1 0 0 0 0
      if ((~(LHSZeroBits & RHSZeroBits)).isPowerOf2())
        // If OrZero isn't set, we cannot give back a zero result.
        // Make sure either the LHS or RHS has a bit set.
        if (OrZero || RHSOneBits.getBoolValue() || LHSOneBits.getBoolValue())
          return true;
    }
  }

  // An exact divide or right shift can only shift off zero bits, so the result
  // is a power of two only if the first operand is a power of two and not
  // copying a sign bit (sdiv int_min, 2).
  if (match(V, m_Exact(m_LShr(m_Value(), m_Value()))) ||
      match(V, m_Exact(m_UDiv(m_Value(), m_Value())))) {
    return isKnownToBeAPowerOfTwo(cast<Operator>(V)->getOperand(0), OrZero,
                                  Depth, Q);
  }

  return false;
}

/// \brief Test whether a GEP's result is known to be non-null.
///
/// Uses properties inherent in a GEP to try to determine whether it is known
/// to be non-null.
///
/// Currently this routine does not support vector GEPs.
static bool isGEPKnownNonNull(const GEPOperator *GEP, unsigned Depth,
                              const Query &Q) {
  if (!GEP->isInBounds() || GEP->getPointerAddressSpace() != 0)
    return false;

  // FIXME: Support vector-GEPs.
  assert(GEP->getType()->isPointerTy() && "We only support plain pointer GEP");

  // If the base pointer is non-null, we cannot walk to a null address with an
  // inbounds GEP in address space zero.
  if (isKnownNonZero(GEP->getPointerOperand(), Depth, Q))
    return true;

  // Walk the GEP operands and see if any operand introduces a non-zero offset.
  // If so, then the GEP cannot produce a null pointer, as doing so would
  // inherently violate the inbounds contract within address space zero.
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    // Struct types are easy -- they must always be indexed by a constant.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      ConstantInt *OpC = cast<ConstantInt>(GTI.getOperand());
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = Q.DL.getStructLayout(STy);
      uint64_t ElementOffset = SL->getElementOffset(ElementIdx);
      if (ElementOffset > 0)
        return true;
      continue;
    }

    // If we have a zero-sized type, the index doesn't matter. Keep looping.
    if (Q.DL.getTypeAllocSize(GTI.getIndexedType()) == 0)
      continue;

    // Fast path the constant operand case both for efficiency and so we don't
    // increment Depth when just zipping down an all-constant GEP.
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand())) {
      if (!OpC->isZero())
        return true;
      continue;
    }

    // We post-increment Depth here because while isKnownNonZero increments it
    // as well, when we pop back up that increment won't persist. We don't want
    // to recurse 10k times just because we have 10k GEP operands. We don't
    // bail completely out because we want to handle constant GEPs regardless
    // of depth.
    if (Depth++ >= MaxDepth)
      continue;

    if (isKnownNonZero(GTI.getOperand(), Depth, Q))
      return true;
  }

  return false;
}

/// Does the 'Range' metadata (which must be a valid MD_range operand list)
/// ensure that the value it's attached to is never Value?  'RangeType' is
/// is the type of the value described by the range.
static bool rangeMetadataExcludesValue(const MDNode* Ranges, const APInt& Value) {
  const unsigned NumRanges = Ranges->getNumOperands() / 2;
  assert(NumRanges >= 1);
  for (unsigned i = 0; i < NumRanges; ++i) {
    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(Ranges->getOperand(2 * i + 1));
    ConstantRange Range(Lower->getValue(), Upper->getValue());
    if (Range.contains(Value))
      return false;
  }
  return true;
}

/// Return true if the given value is known to be non-zero when defined.
/// For vectors return true if every element is known to be non-zero when
/// defined. Supports values with integer or pointer type and vectors of
/// integers.
bool isKnownNonZero(const Value *V, unsigned Depth, const Query &Q) {
  if (auto *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue())
      return false;
    if (isa<ConstantInt>(C))
      // Must be non-zero due to null test above.
      return true;

    // For constant vectors, check that all elements are undefined or known
    // non-zero to determine that the whole vector is known non-zero.
    if (auto *VecTy = dyn_cast<VectorType>(C->getType())) {
      for (unsigned i = 0, e = VecTy->getNumElements(); i != e; ++i) {
        Constant *Elt = C->getAggregateElement(i);
        if (!Elt || Elt->isNullValue())
          return false;
        if (!isa<UndefValue>(Elt) && !isa<ConstantInt>(Elt))
          return false;
      }
      return true;
    }

    return false;
  }

  if (auto *I = dyn_cast<Instruction>(V)) {
    if (MDNode *Ranges = I->getMetadata(LLVMContext::MD_range)) {
      // If the possible ranges don't contain zero, then the value is
      // definitely non-zero.
      if (auto *Ty = dyn_cast<IntegerType>(V->getType())) {
        const APInt ZeroValue(Ty->getBitWidth(), 0);
        if (rangeMetadataExcludesValue(Ranges, ZeroValue))
          return true;
      }
    }
  }

  // The remaining tests are all recursive, so bail out if we hit the limit.
  if (Depth++ >= MaxDepth)
    return false;

  // Check for pointer simplifications.
  if (V->getType()->isPointerTy()) {
    if (isKnownNonNull(V))
      return true;
    if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      if (isGEPKnownNonNull(GEP, Depth, Q))
        return true;
  }

  unsigned BitWidth = getBitWidth(V->getType()->getScalarType(), Q.DL);

  // X | Y != 0 if X != 0 or Y != 0.
  Value *X = nullptr, *Y = nullptr;
  if (match(V, m_Or(m_Value(X), m_Value(Y))))
    return isKnownNonZero(X, Depth, Q) || isKnownNonZero(Y, Depth, Q);

  // ext X != 0 if X != 0.
  if (isa<SExtInst>(V) || isa<ZExtInst>(V))
    return isKnownNonZero(cast<Instruction>(V)->getOperand(0), Depth, Q);

  // shl X, Y != 0 if X is odd.  Note that the value of the shift is undefined
  // if the lowest bit is shifted off the end.
  if (BitWidth && match(V, m_Shl(m_Value(X), m_Value(Y)))) {
    // shl nuw can't remove any non-zero bits.
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    if (BO->hasNoUnsignedWrap())
      return isKnownNonZero(X, Depth, Q);

    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    computeKnownBits(X, KnownZero, KnownOne, Depth, Q);
    if (KnownOne[0])
      return true;
  }
  // shr X, Y != 0 if X is negative.  Note that the value of the shift is not
  // defined if the sign bit is shifted off the end.
  else if (match(V, m_Shr(m_Value(X), m_Value(Y)))) {
    // shr exact can only shift out zero bits.
    const PossiblyExactOperator *BO = cast<PossiblyExactOperator>(V);
    if (BO->isExact())
      return isKnownNonZero(X, Depth, Q);

    bool XKnownNonNegative, XKnownNegative;
    ComputeSignBit(X, XKnownNonNegative, XKnownNegative, Depth, Q);
    if (XKnownNegative)
      return true;

    // If the shifter operand is a constant, and all of the bits shifted
    // out are known to be zero, and X is known non-zero then at least one
    // non-zero bit must remain.
    if (ConstantInt *Shift = dyn_cast<ConstantInt>(Y)) {
      APInt KnownZero(BitWidth, 0);
      APInt KnownOne(BitWidth, 0);
      computeKnownBits(X, KnownZero, KnownOne, Depth, Q);

      auto ShiftVal = Shift->getLimitedValue(BitWidth - 1);
      // Is there a known one in the portion not shifted out?
      if (KnownOne.countLeadingZeros() < BitWidth - ShiftVal)
        return true;
      // Are all the bits to be shifted out known zero?
      if (KnownZero.countTrailingOnes() >= ShiftVal)
        return isKnownNonZero(X, Depth, Q);
    }
  }
  // div exact can only produce a zero if the dividend is zero.
  else if (match(V, m_Exact(m_IDiv(m_Value(X), m_Value())))) {
    return isKnownNonZero(X, Depth, Q);
  }
  // X + Y.
  else if (match(V, m_Add(m_Value(X), m_Value(Y)))) {
    bool XKnownNonNegative, XKnownNegative;
    bool YKnownNonNegative, YKnownNegative;
    ComputeSignBit(X, XKnownNonNegative, XKnownNegative, Depth, Q);
    ComputeSignBit(Y, YKnownNonNegative, YKnownNegative, Depth, Q);

    // If X and Y are both non-negative (as signed values) then their sum is not
    // zero unless both X and Y are zero.
    if (XKnownNonNegative && YKnownNonNegative)
      if (isKnownNonZero(X, Depth, Q) || isKnownNonZero(Y, Depth, Q))
        return true;

    // If X and Y are both negative (as signed values) then their sum is not
    // zero unless both X and Y equal INT_MIN.
    if (BitWidth && XKnownNegative && YKnownNegative) {
      APInt KnownZero(BitWidth, 0);
      APInt KnownOne(BitWidth, 0);
      APInt Mask = APInt::getSignedMaxValue(BitWidth);
      // The sign bit of X is set.  If some other bit is set then X is not equal
      // to INT_MIN.
      computeKnownBits(X, KnownZero, KnownOne, Depth, Q);
      if ((KnownOne & Mask) != 0)
        return true;
      // The sign bit of Y is set.  If some other bit is set then Y is not equal
      // to INT_MIN.
      computeKnownBits(Y, KnownZero, KnownOne, Depth, Q);
      if ((KnownOne & Mask) != 0)
        return true;
    }

    // The sum of a non-negative number and a power of two is not zero.
    if (XKnownNonNegative &&
        isKnownToBeAPowerOfTwo(Y, /*OrZero*/ false, Depth, Q))
      return true;
    if (YKnownNonNegative &&
        isKnownToBeAPowerOfTwo(X, /*OrZero*/ false, Depth, Q))
      return true;
  }
  // X * Y.
  else if (match(V, m_Mul(m_Value(X), m_Value(Y)))) {
    const OverflowingBinaryOperator *BO = cast<OverflowingBinaryOperator>(V);
    // If X and Y are non-zero then so is X * Y as long as the multiplication
    // does not overflow.
    if ((BO->hasNoSignedWrap() || BO->hasNoUnsignedWrap()) &&
        isKnownNonZero(X, Depth, Q) && isKnownNonZero(Y, Depth, Q))
      return true;
  }
  // (C ? X : Y) != 0 if X != 0 and Y != 0.
  else if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
    if (isKnownNonZero(SI->getTrueValue(), Depth, Q) &&
        isKnownNonZero(SI->getFalseValue(), Depth, Q))
      return true;
  }
  // PHI
  else if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    // Try and detect a recurrence that monotonically increases from a
    // starting value, as these are common as induction variables.
    if (PN->getNumIncomingValues() == 2) {
      Value *Start = PN->getIncomingValue(0);
      Value *Induction = PN->getIncomingValue(1);
      if (isa<ConstantInt>(Induction) && !isa<ConstantInt>(Start))
        std::swap(Start, Induction);
      if (ConstantInt *C = dyn_cast<ConstantInt>(Start)) {
        if (!C->isZero() && !C->isNegative()) {
          ConstantInt *X;
          if ((match(Induction, m_NSWAdd(m_Specific(PN), m_ConstantInt(X))) ||
               match(Induction, m_NUWAdd(m_Specific(PN), m_ConstantInt(X)))) &&
              !X->isNegative())
            return true;
        }
      }
    }
    // Check if all incoming values are non-zero constant.
    bool AllNonZeroConstants = all_of(PN->operands(), [](Value *V) {
      return isa<ConstantInt>(V) && !cast<ConstantInt>(V)->isZeroValue();
    });
    if (AllNonZeroConstants)
      return true;
  }

  if (!BitWidth) return false;
  APInt KnownZero(BitWidth, 0);
  APInt KnownOne(BitWidth, 0);
  computeKnownBits(V, KnownZero, KnownOne, Depth, Q);
  return KnownOne != 0;
}

/// Return true if V2 == V1 + X, where X is known non-zero.
static bool isAddOfNonZero(const Value *V1, const Value *V2, const Query &Q) {
  const BinaryOperator *BO = dyn_cast<BinaryOperator>(V1);
  if (!BO || BO->getOpcode() != Instruction::Add)
    return false;
  Value *Op = nullptr;
  if (V2 == BO->getOperand(0))
    Op = BO->getOperand(1);
  else if (V2 == BO->getOperand(1))
    Op = BO->getOperand(0);
  else
    return false;
  return isKnownNonZero(Op, 0, Q);
}

/// Return true if it is known that V1 != V2.
static bool isKnownNonEqual(const Value *V1, const Value *V2, const Query &Q) {
  if (V1->getType()->isVectorTy() || V1 == V2)
    return false;
  if (V1->getType() != V2->getType())
    // We can't look through casts yet.
    return false;
  if (isAddOfNonZero(V1, V2, Q) || isAddOfNonZero(V2, V1, Q))
    return true;

  if (IntegerType *Ty = dyn_cast<IntegerType>(V1->getType())) {
    // Are any known bits in V1 contradictory to known bits in V2? If V1
    // has a known zero where V2 has a known one, they must not be equal.
    auto BitWidth = Ty->getBitWidth();
    APInt KnownZero1(BitWidth, 0);
    APInt KnownOne1(BitWidth, 0);
    computeKnownBits(V1, KnownZero1, KnownOne1, 0, Q);
    APInt KnownZero2(BitWidth, 0);
    APInt KnownOne2(BitWidth, 0);
    computeKnownBits(V2, KnownZero2, KnownOne2, 0, Q);

    auto OppositeBits = (KnownZero1 & KnownOne2) | (KnownZero2 & KnownOne1);
    if (OppositeBits.getBoolValue())
      return true;
  }
  return false;
}

/// Return true if 'V & Mask' is known to be zero.  We use this predicate to
/// simplify operations downstream. Mask is known to be zero for bits that V
/// cannot have.
///
/// This function is defined on values with integer type, values with pointer
/// type, and vectors of integers.  In the case
/// where V is a vector, the mask, known zero, and known one values are the
/// same width as the vector element, and the bit is set only if it is true
/// for all of the elements in the vector.
bool MaskedValueIsZero(const Value *V, const APInt &Mask, unsigned Depth,
                       const Query &Q) {
  APInt KnownZero(Mask.getBitWidth(), 0), KnownOne(Mask.getBitWidth(), 0);
  computeKnownBits(V, KnownZero, KnownOne, Depth, Q);
  return (KnownZero & Mask) == Mask;
}

/// For vector constants, loop over the elements and find the constant with the
/// minimum number of sign bits. Return 0 if the value is not a vector constant
/// or if any element was not analyzed; otherwise, return the count for the
/// element with the minimum number of sign bits.
static unsigned computeNumSignBitsVectorConstant(const Value *V,
                                                 unsigned TyBits) {
  const auto *CV = dyn_cast<Constant>(V);
  if (!CV || !CV->getType()->isVectorTy())
    return 0;

  unsigned MinSignBits = TyBits;
  unsigned NumElts = CV->getType()->getVectorNumElements();
  for (unsigned i = 0; i != NumElts; ++i) {
    // If we find a non-ConstantInt, bail out.
    auto *Elt = dyn_cast_or_null<ConstantInt>(CV->getAggregateElement(i));
    if (!Elt)
      return 0;

    // If the sign bit is 1, flip the bits, so we always count leading zeros.
    APInt EltVal = Elt->getValue();
    if (EltVal.isNegative())
      EltVal = ~EltVal;
    MinSignBits = std::min(MinSignBits, EltVal.countLeadingZeros());
  }

  return MinSignBits;
}

/// Return the number of times the sign bit of the register is replicated into
/// the other bits. We know that at least 1 bit is always equal to the sign bit
/// (itself), but other cases can give us information. For example, immediately
/// after an "ashr X, 2", we know that the top 3 bits are all equal to each
/// other, so we return 3. For vectors, return the number of sign bits for the
/// vector element with the mininum number of known sign bits.
unsigned ComputeNumSignBits(const Value *V, unsigned Depth, const Query &Q) {
  unsigned TyBits = Q.DL.getTypeSizeInBits(V->getType()->getScalarType());
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  // Note that ConstantInt is handled by the general computeKnownBits case
  // below.

  if (Depth == 6)
    return 1;  // Limit search depth.

  const Operator *U = dyn_cast<Operator>(V);
  switch (Operator::getOpcode(V)) {
  default: break;
  case Instruction::SExt:
    Tmp = TyBits - U->getOperand(0)->getType()->getScalarSizeInBits();
    return ComputeNumSignBits(U->getOperand(0), Depth + 1, Q) + Tmp;

  case Instruction::SDiv: {
    const APInt *Denominator;
    // sdiv X, C -> adds log(C) sign bits.
    if (match(U->getOperand(1), m_APInt(Denominator))) {

      // Ignore non-positive denominator.
      if (!Denominator->isStrictlyPositive())
        break;

      // Calculate the incoming numerator bits.
      unsigned NumBits = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);

      // Add floor(log(C)) bits to the numerator bits.
      return std::min(TyBits, NumBits + Denominator->logBase2());
    }
    break;
  }

  case Instruction::SRem: {
    const APInt *Denominator;
    // srem X, C -> we know that the result is within [-C+1,C) when C is a
    // positive constant.  This let us put a lower bound on the number of sign
    // bits.
    if (match(U->getOperand(1), m_APInt(Denominator))) {

      // Ignore non-positive denominator.
      if (!Denominator->isStrictlyPositive())
        break;

      // Calculate the incoming numerator bits. SRem by a positive constant
      // can't lower the number of sign bits.
      unsigned NumrBits =
          ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);

      // Calculate the leading sign bit constraints by examining the
      // denominator.  Given that the denominator is positive, there are two
      // cases:
      //
      //  1. the numerator is positive.  The result range is [0,C) and [0,C) u<
      //     (1 << ceilLogBase2(C)).
      //
      //  2. the numerator is negative.  Then the result range is (-C,0] and
      //     integers in (-C,0] are either 0 or >u (-1 << ceilLogBase2(C)).
      //
      // Thus a lower bound on the number of sign bits is `TyBits -
      // ceilLogBase2(C)`.

      unsigned ResBits = TyBits - Denominator->ceilLogBase2();
      return std::max(NumrBits, ResBits);
    }
    break;
  }

  case Instruction::AShr: {
    Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
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
      Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
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
    Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // computeKnownBits, and pick whichever answer is better.
    }
    break;

  case Instruction::Select:
    Tmp = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(U->getOperand(2), Depth + 1, Q);
    return std::min(Tmp, Tmp2);

  case Instruction::Add:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
    if (Tmp == 1) return 1;  // Early out.

    // Special case decrementing a value (ADD X, -1):
    if (const auto *CRHS = dyn_cast<Constant>(U->getOperand(1)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        computeKnownBits(U->getOperand(0), KnownZero, KnownOne, Depth + 1, Q);

        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(TyBits, 1)).isAllOnesValue())
          return TyBits;

        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }

    Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
    if (Tmp2 == 1) return 1;
    return std::min(Tmp, Tmp2)-1;

  case Instruction::Sub:
    Tmp2 = ComputeNumSignBits(U->getOperand(1), Depth + 1, Q);
    if (Tmp2 == 1) return 1;

    // Handle NEG.
    if (const auto *CLHS = dyn_cast<Constant>(U->getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        computeKnownBits(U->getOperand(1), KnownZero, KnownOne, Depth + 1, Q);
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
    Tmp = ComputeNumSignBits(U->getOperand(0), Depth + 1, Q);
    if (Tmp == 1) return 1;  // Early out.
    return std::min(Tmp, Tmp2)-1;

  case Instruction::PHI: {
    const PHINode *PN = cast<PHINode>(U);
    unsigned NumIncomingValues = PN->getNumIncomingValues();
    // Don't analyze large in-degree PHIs.
    if (NumIncomingValues > 4) break;
    // Unreachable blocks may have zero-operand PHI nodes.
    if (NumIncomingValues == 0) break;

    // Take the minimum of all incoming values.  This can't infinitely loop
    // because of our depth threshold.
    Tmp = ComputeNumSignBits(PN->getIncomingValue(0), Depth + 1, Q);
    for (unsigned i = 1, e = NumIncomingValues; i != e; ++i) {
      if (Tmp == 1) return Tmp;
      Tmp = std::min(
          Tmp, ComputeNumSignBits(PN->getIncomingValue(i), Depth + 1, Q));
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

  // If we can examine all elements of a vector constant successfully, we're
  // done (we can't do any better than that). If not, keep trying.
  if (unsigned VecSignBits = computeNumSignBitsVectorConstant(V, TyBits))
    return VecSignBits;

  APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
  computeKnownBits(V, KnownZero, KnownOne, Depth, Q);

  // If we know that the sign bit is either zero or one, determine the number of
  // identical bits in the top of the input value.
  if (KnownZero.isNegative())
    return std::max(FirstAnswer, KnownZero.countLeadingOnes());

  if (KnownOne.isNegative())
    return std::max(FirstAnswer, KnownOne.countLeadingOnes());

  // computeKnownBits gave us no extra information about the top bits.
  return FirstAnswer;
}

/// This function computes the integer multiple of Base that equals V.
/// If successful, it returns true and returns the multiple in
/// Multiple. If unsuccessful, it returns false. It looks
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

    Value *Mul0 = nullptr;
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

    Value *Mul1 = nullptr;
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

Intrinsic::ID llvm::getIntrinsicForCallSite(ImmutableCallSite ICS,
                                            const TargetLibraryInfo *TLI) {
  const Function *F = ICS.getCalledFunction();
  if (!F)
    return Intrinsic::not_intrinsic;

  if (F->isIntrinsic())
    return F->getIntrinsicID();

  if (!TLI)
    return Intrinsic::not_intrinsic;

  LibFunc::Func Func;
  // We're going to make assumptions on the semantics of the functions, check
  // that the target knows that it's available in this environment and it does
  // not have local linkage.
  if (!F || F->hasLocalLinkage() || !TLI->getLibFunc(*F, Func))
    return Intrinsic::not_intrinsic;

  if (!ICS.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  // Otherwise check if we have a call to a function that can be turned into a
  // vector intrinsic.
  switch (Func) {
  default:
    break;
  case LibFunc::sin:
  case LibFunc::sinf:
  case LibFunc::sinl:
    return Intrinsic::sin;
  case LibFunc::cos:
  case LibFunc::cosf:
  case LibFunc::cosl:
    return Intrinsic::cos;
  case LibFunc::exp:
  case LibFunc::expf:
  case LibFunc::expl:
    return Intrinsic::exp;
  case LibFunc::exp2:
  case LibFunc::exp2f:
  case LibFunc::exp2l:
    return Intrinsic::exp2;
  case LibFunc::log:
  case LibFunc::logf:
  case LibFunc::logl:
    return Intrinsic::log;
  case LibFunc::log10:
  case LibFunc::log10f:
  case LibFunc::log10l:
    return Intrinsic::log10;
  case LibFunc::log2:
  case LibFunc::log2f:
  case LibFunc::log2l:
    return Intrinsic::log2;
  case LibFunc::fabs:
  case LibFunc::fabsf:
  case LibFunc::fabsl:
    return Intrinsic::fabs;
  case LibFunc::fmin:
  case LibFunc::fminf:
  case LibFunc::fminl:
    return Intrinsic::minnum;
  case LibFunc::fmax:
  case LibFunc::fmaxf:
  case LibFunc::fmaxl:
    return Intrinsic::maxnum;
  case LibFunc::copysign:
  case LibFunc::copysignf:
  case LibFunc::copysignl:
    return Intrinsic::copysign;
  case LibFunc::floor:
  case LibFunc::floorf:
  case LibFunc::floorl:
    return Intrinsic::floor;
  case LibFunc::ceil:
  case LibFunc::ceilf:
  case LibFunc::ceill:
    return Intrinsic::ceil;
  case LibFunc::trunc:
  case LibFunc::truncf:
  case LibFunc::truncl:
    return Intrinsic::trunc;
  case LibFunc::rint:
  case LibFunc::rintf:
  case LibFunc::rintl:
    return Intrinsic::rint;
  case LibFunc::nearbyint:
  case LibFunc::nearbyintf:
  case LibFunc::nearbyintl:
    return Intrinsic::nearbyint;
  case LibFunc::round:
  case LibFunc::roundf:
  case LibFunc::roundl:
    return Intrinsic::round;
  case LibFunc::pow:
  case LibFunc::powf:
  case LibFunc::powl:
    return Intrinsic::pow;
  case LibFunc::sqrt:
  case LibFunc::sqrtf:
  case LibFunc::sqrtl:
    if (ICS->hasNoNaNs())
      return Intrinsic::sqrt;
    return Intrinsic::not_intrinsic;
  }

  return Intrinsic::not_intrinsic;
}

/// Return true if we can prove that the specified FP value is never equal to
/// -0.0.
///
/// NOTE: this function will need to be revisited when we support non-default
/// rounding modes!
///
bool llvm::CannotBeNegativeZero(const Value *V, const TargetLibraryInfo *TLI,
                                unsigned Depth) {
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->getValueAPF().isNegZero();

  // FIXME: Magic number! At the least, this should be given a name because it's
  // used similarly in CannotBeOrderedLessThanZero(). A better fix may be to
  // expose it as a parameter, so it can be used for testing / experimenting.
  if (Depth == 6)
    return false;  // Limit search depth.

  const Operator *I = dyn_cast<Operator>(V);
  if (!I) return false;

  // Check if the nsz fast-math flag is set
  if (const FPMathOperator *FPO = dyn_cast<FPMathOperator>(I))
    if (FPO->hasNoSignedZeros())
      return true;

  // (add x, 0.0) is guaranteed to return +0.0, not -0.0.
  if (I->getOpcode() == Instruction::FAdd)
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(I->getOperand(1)))
      if (CFP->isNullValue())
        return true;

  // sitofp and uitofp turn into +0.0 for zero.
  if (isa<SIToFPInst>(I) || isa<UIToFPInst>(I))
    return true;

  if (const CallInst *CI = dyn_cast<CallInst>(I)) {
    Intrinsic::ID IID = getIntrinsicForCallSite(CI, TLI);
    switch (IID) {
    default:
      break;
    // sqrt(-0.0) = -0.0, no other negative results are possible.
    case Intrinsic::sqrt:
      return CannotBeNegativeZero(CI->getArgOperand(0), TLI, Depth + 1);
    // fabs(x) != -0.0
    case Intrinsic::fabs:
      return true;
    }
  }

  return false;
}

bool llvm::CannotBeOrderedLessThanZero(const Value *V,
                                       const TargetLibraryInfo *TLI,
                                       unsigned Depth) {
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->getValueAPF().isNegative() || CFP->getValueAPF().isZero();

  // FIXME: Magic number! At the least, this should be given a name because it's
  // used similarly in CannotBeNegativeZero(). A better fix may be to
  // expose it as a parameter, so it can be used for testing / experimenting.
  if (Depth == 6)
    return false;  // Limit search depth.

  const Operator *I = dyn_cast<Operator>(V);
  if (!I) return false;

  switch (I->getOpcode()) {
  default: break;
  // Unsigned integers are always nonnegative.
  case Instruction::UIToFP:
    return true;
  case Instruction::FMul:
    // x*x is always non-negative or a NaN.
    if (I->getOperand(0) == I->getOperand(1))
      return true;
    LLVM_FALLTHROUGH;
  case Instruction::FAdd:
  case Instruction::FDiv:
  case Instruction::FRem:
    return CannotBeOrderedLessThanZero(I->getOperand(0), TLI, Depth + 1) &&
           CannotBeOrderedLessThanZero(I->getOperand(1), TLI, Depth + 1);
  case Instruction::Select:
    return CannotBeOrderedLessThanZero(I->getOperand(1), TLI, Depth + 1) &&
           CannotBeOrderedLessThanZero(I->getOperand(2), TLI, Depth + 1);
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    // Widening/narrowing never change sign.
    return CannotBeOrderedLessThanZero(I->getOperand(0), TLI, Depth + 1);
  case Instruction::Call:
    Intrinsic::ID IID = getIntrinsicForCallSite(cast<CallInst>(I), TLI);
    switch (IID) {
    default:
      break;
    case Intrinsic::maxnum:
      return CannotBeOrderedLessThanZero(I->getOperand(0), TLI, Depth + 1) ||
             CannotBeOrderedLessThanZero(I->getOperand(1), TLI, Depth + 1);
    case Intrinsic::minnum:
      return CannotBeOrderedLessThanZero(I->getOperand(0), TLI, Depth + 1) &&
             CannotBeOrderedLessThanZero(I->getOperand(1), TLI, Depth + 1);
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::fabs:
    case Intrinsic::sqrt:
      return true;
    case Intrinsic::powi:
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // powi(x,n) is non-negative if n is even.
        if (CI->getBitWidth() <= 64 && CI->getSExtValue() % 2u == 0)
          return true;
      }
      return CannotBeOrderedLessThanZero(I->getOperand(0), TLI, Depth + 1);
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
      // x*x+y is non-negative if y is non-negative.
      return I->getOperand(0) == I->getOperand(1) &&
             CannotBeOrderedLessThanZero(I->getOperand(2), TLI, Depth + 1);
    }
    break;
  }
  return false;
}

/// If the specified value can be set by repeating the same byte in memory,
/// return the i8 value that it is represented with.  This is
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

  // We can handle constant integers that are multiple of 8 bits.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getBitWidth() % 8 == 0) {
      assert(CI->getBitWidth() > 8 && "8 bits should be handled above!");

      if (!CI->getValue().isSplat(8))
        return nullptr;
      return ConstantInt::get(V->getContext(), CI->getValue().trunc(8));
    }
  }

  // A ConstantDataArray/Vector is splatable if all its members are equal and
  // also splatable.
  if (ConstantDataSequential *CA = dyn_cast<ConstantDataSequential>(V)) {
    Value *Elt = CA->getElementAsConstant(0);
    Value *Val = isBytewiseValue(Elt);
    if (!Val)
      return nullptr;

    for (unsigned I = 1, E = CA->getNumElements(); I != E; ++I)
      if (CA->getElementAsConstant(I) != Elt)
        return nullptr;

    return Val;
  }

  // Conceptually, we could handle things like:
  //   %a = zext i8 %X to i16
  //   %b = shl i16 %a, 8
  //   %c = or i16 %a, %b
  // but until there is an example that actually needs this, it doesn't seem
  // worth worrying about.
  return nullptr;
}


// This is the recursive version of BuildSubAggregate. It takes a few different
// arguments. Idxs is the index within the nested struct From that we are
// looking at now (which is of type IndexedType). IdxSkip is the number of
// indices from Idxs that should be left out when inserting into the resulting
// struct. To is the result struct built so far, new insertvalue instructions
// build on that.
static Value *BuildSubAggregate(Value *From, Value* To, Type *IndexedType,
                                SmallVectorImpl<unsigned> &Idxs,
                                unsigned IdxSkip,
                                Instruction *InsertBefore) {
  llvm::StructType *STy = dyn_cast<llvm::StructType>(IndexedType);
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
    return nullptr;

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

/// Given an aggregrate and an sequence of indices, see if
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
    if (!C) return nullptr;
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
          return nullptr;

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
      // See if the (aggregate) value inserted into has the value we are
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
    // If we're extracting a value from an aggregate that was extracted from
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
  return nullptr;
}

/// Analyze the specified pointer to see if it can be expressed as a base
/// pointer plus a constant offset. Return the base and offset to the caller.
Value *llvm::GetPointerBaseWithConstantOffset(Value *Ptr, int64_t &Offset,
                                              const DataLayout &DL) {
  unsigned BitWidth = DL.getPointerTypeSizeInBits(Ptr->getType());
  APInt ByteOffset(BitWidth, 0);

  // We walk up the defs but use a visited set to handle unreachable code. In
  // that case, we stop after accumulating the cycle once (not that it
  // matters).
  SmallPtrSet<Value *, 16> Visited;
  while (Visited.insert(Ptr).second) {
    if (Ptr->getType()->isVectorTy())
      break;

    if (GEPOperator *GEP = dyn_cast<GEPOperator>(Ptr)) {
      APInt GEPOffset(BitWidth, 0);
      if (!GEP->accumulateConstantOffset(DL, GEPOffset))
        break;

      ByteOffset += GEPOffset;

      Ptr = GEP->getPointerOperand();
    } else if (Operator::getOpcode(Ptr) == Instruction::BitCast ||
               Operator::getOpcode(Ptr) == Instruction::AddrSpaceCast) {
      Ptr = cast<Operator>(Ptr)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(Ptr)) {
      if (GA->isInterposable())
        break;
      Ptr = GA->getAliasee();
    } else {
      break;
    }
  }
  Offset = ByteOffset.getSExtValue();
  return Ptr;
}

bool llvm::isGEPBasedOnPointerToString(const GEPOperator *GEP) {
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return false;

  // Make sure the index-ee is a pointer to array of i8.
  ArrayType *AT = dyn_cast<ArrayType>(GEP->getSourceElementType());
  if (!AT || !AT->getElementType()->isIntegerTy(8))
    return false;

  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  const ConstantInt *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!FirstIdx || !FirstIdx->isZero())
    return false;

  return true;
}

/// This function computes the length of a null-terminated C string pointed to
/// by V. If successful, it returns true and returns the string in Str.
/// If unsuccessful, it returns false.
bool llvm::getConstantStringInfo(const Value *V, StringRef &Str,
                                 uint64_t Offset, bool TrimAtNul) {
  assert(V);

  // Look through bitcast instructions and geps.
  V = V->stripPointerCasts();

  // If the value is a GEP instruction or constant expression, treat it as an
  // offset.
  if (const GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    // The GEP operator should be based on a pointer to string constant, and is
    // indexing into the string constant.
    if (!isGEPBasedOnPointerToString(GEP))
      return false;

    // If the second index isn't a ConstantInt, then this is a variable index
    // into the array.  If this occurs, we can't say anything meaningful about
    // the string.
    uint64_t StartIdx = 0;
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
      StartIdx = CI->getZExtValue();
    else
      return false;
    return getConstantStringInfo(GEP->getOperand(0), Str, StartIdx + Offset,
                                 TrimAtNul);
  }

  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return false;

  // Handle the all-zeros case.
  if (GV->getInitializer()->isNullValue()) {
    // This is a degenerate case. The initializer is constant zero so the
    // length of the string must be zero.
    Str = "";
    return true;
  }

  // This must be a ConstantDataArray.
  const auto *Array = dyn_cast<ConstantDataArray>(GV->getInitializer());
  if (!Array || !Array->isString())
    return false;

  // Get the number of elements in the array.
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

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(const Value *V,
                                 SmallPtrSetImpl<const PHINode*> &PHIs) {
  // Look through noop bitcast instructions.
  V = V->stripPointerCasts();

  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (const PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN).second)
      return ~0ULL;  // already in the set.

    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (Value *IncValue : PN->incoming_values()) {
      uint64_t Len = GetStringLengthH(IncValue, PHIs);
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
  if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
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

/// If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
uint64_t llvm::GetStringLength(const Value *V) {
  if (!V->getType()->isPointerTy()) return 0;

  SmallPtrSet<const PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

/// \brief \p PN defines a loop-variant pointer to an object.  Check if the
/// previous iteration of the loop was referring to the same object as \p PN.
static bool isSameUnderlyingObjectInLoop(const PHINode *PN,
                                         const LoopInfo *LI) {
  // Find the loop-defined value.
  Loop *L = LI->getLoopFor(PN->getParent());
  if (PN->getNumIncomingValues() != 2)
    return true;

  // Find the value from previous iteration.
  auto *PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(0));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    PrevValue = dyn_cast<Instruction>(PN->getIncomingValue(1));
  if (!PrevValue || LI->getLoopFor(PrevValue->getParent()) != L)
    return true;

  // If a new pointer is loaded in the loop, the pointer references a different
  // object in every iteration.  E.g.:
  //    for (i)
  //       int *p = a[i];
  //       ...
  if (auto *Load = dyn_cast<LoadInst>(PrevValue))
    if (!L->isLoopInvariant(Load->getPointerOperand()))
      return false;
  return true;
}

Value *llvm::GetUnderlyingObject(Value *V, const DataLayout &DL,
                                 unsigned MaxLookup) {
  if (!V->getType()->isPointerTy())
    return V;
  for (unsigned Count = 0; MaxLookup == 0 || Count < MaxLookup; ++Count) {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
    } else if (Operator::getOpcode(V) == Instruction::BitCast ||
               Operator::getOpcode(V) == Instruction::AddrSpaceCast) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
      if (GA->isInterposable())
        return V;
      V = GA->getAliasee();
    } else {
      if (auto CS = CallSite(V))
        if (Value *RV = CS.getReturnedArgOperand()) {
          V = RV;
          continue;
        }

      // See if InstructionSimplify knows any relevant tricks.
      if (Instruction *I = dyn_cast<Instruction>(V))
        // TODO: Acquire a DominatorTree and AssumptionCache and use them.
        if (Value *Simplified = SimplifyInstruction(I, DL, nullptr)) {
          V = Simplified;
          continue;
        }

      return V;
    }
    assert(V->getType()->isPointerTy() && "Unexpected operand type!");
  }
  return V;
}

void llvm::GetUnderlyingObjects(Value *V, SmallVectorImpl<Value *> &Objects,
                                const DataLayout &DL, LoopInfo *LI,
                                unsigned MaxLookup) {
  SmallPtrSet<Value *, 4> Visited;
  SmallVector<Value *, 4> Worklist;
  Worklist.push_back(V);
  do {
    Value *P = Worklist.pop_back_val();
    P = GetUnderlyingObject(P, DL, MaxLookup);

    if (!Visited.insert(P).second)
      continue;

    if (SelectInst *SI = dyn_cast<SelectInst>(P)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    if (PHINode *PN = dyn_cast<PHINode>(P)) {
      // If this PHI changes the underlying object in every iteration of the
      // loop, don't look through it.  Consider:
      //   int **A;
      //   for (i) {
      //     Prev = Curr;     // Prev = PHI (Prev_0, Curr)
      //     Curr = A[i];
      //     *Prev, *Curr;
      //
      // Prev is tracking Curr one iteration behind so they refer to different
      // underlying objects.
      if (!LI || !LI->isLoopHeader(PN->getParent()) ||
          isSameUnderlyingObjectInLoop(PN, LI))
        for (Value *IncValue : PN->incoming_values())
          Worklist.push_back(IncValue);
      continue;
    }

    Objects.push_back(P);
  } while (!Worklist.empty());
}

/// Return true if the only users of this pointer are lifetime markers.
bool llvm::onlyUsedByLifetimeMarkers(const Value *V) {
  for (const User *U : V->users()) {
    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    if (!II) return false;

    if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
        II->getIntrinsicID() != Intrinsic::lifetime_end)
      return false;
  }
  return true;
}

bool llvm::isSafeToSpeculativelyExecute(const Value *V,
                                        const Instruction *CtxI,
                                        const DominatorTree *DT) {
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
  case Instruction::URem: {
    // x / y is undefined if y == 0.
    const APInt *V;
    if (match(Inst->getOperand(1), m_APInt(V)))
      return *V != 0;
    return false;
  }
  case Instruction::SDiv:
  case Instruction::SRem: {
    // x / y is undefined if y == 0 or x == INT_MIN and y == -1
    const APInt *Numerator, *Denominator;
    if (!match(Inst->getOperand(1), m_APInt(Denominator)))
      return false;
    // We cannot hoist this division if the denominator is 0.
    if (*Denominator == 0)
      return false;
    // It's safe to hoist if the denominator is not 0 or -1.
    if (*Denominator != -1)
      return true;
    // At this point we know that the denominator is -1.  It is safe to hoist as
    // long we know that the numerator is not INT_MIN.
    if (match(Inst->getOperand(0), m_APInt(Numerator)))
      return !Numerator->isMinSignedValue();
    // The numerator *might* be MinSignedValue.
    return false;
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(Inst);
    if (!LI->isUnordered() ||
        // Speculative load may create a race that did not exist in the source.
        LI->getFunction()->hasFnAttribute(Attribute::SanitizeThread) ||
        // Speculative load may load data from dirty regions.
        LI->getFunction()->hasFnAttribute(Attribute::SanitizeAddress))
      return false;
    const DataLayout &DL = LI->getModule()->getDataLayout();
    return isDereferenceableAndAlignedPointer(LI->getPointerOperand(),
                                              LI->getAlignment(), DL, CtxI, DT);
  }
  case Instruction::Call: {
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
      switch (II->getIntrinsicID()) {
      // These synthetic intrinsics have no side-effects and just mark
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
      // These intrinsics are defined to have the same behavior as libm
      // functions except for setting errno.
      case Intrinsic::sqrt:
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
        return true;
      // These intrinsics are defined to have the same behavior as libm
      // functions, and the corresponding libm functions never set errno.
      case Intrinsic::trunc:
      case Intrinsic::copysign:
      case Intrinsic::fabs:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
        return true;
      // These intrinsics are defined to have the same behavior as libm
      // functions, which never overflow when operating on the IEEE754 types
      // that we support, and never set errno otherwise.
      case Intrinsic::ceil:
      case Intrinsic::floor:
      case Intrinsic::nearbyint:
      case Intrinsic::rint:
      case Intrinsic::round:
        return true;
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
  case Instruction::AtomicRMW:
  case Instruction::AtomicCmpXchg:
  case Instruction::LandingPad:
  case Instruction::Resume:
  case Instruction::CatchSwitch:
  case Instruction::CatchPad:
  case Instruction::CatchRet:
  case Instruction::CleanupPad:
  case Instruction::CleanupRet:
    return false; // Misc instructions which have effects
  }
}

bool llvm::mayBeMemoryDependent(const Instruction &I) {
  return I.mayReadOrWriteMemory() || !isSafeToSpeculativelyExecute(&I);
}

/// Return true if we know that the specified value is never null.
bool llvm::isKnownNonNull(const Value *V) {
  assert(V->getType()->isPointerTy() && "V must be pointer type");

  // Alloca never returns null, malloc might.
  if (isa<AllocaInst>(V)) return true;

  // A byval, inalloca, or nonnull argument is never null.
  if (const Argument *A = dyn_cast<Argument>(V))
    return A->hasByValOrInAllocaAttr() || A->hasNonNullAttr();

  // A global variable in address space 0 is non null unless extern weak.
  // Other address spaces may have null as a valid address for a global,
  // so we can't assume anything.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return !GV->hasExternalWeakLinkage() &&
           GV->getType()->getAddressSpace() == 0;

  // A Load tagged with nonnull metadata is never null.
  if (const LoadInst *LI = dyn_cast<LoadInst>(V))
    return LI->getMetadata(LLVMContext::MD_nonnull);

  if (auto CS = ImmutableCallSite(V))
    if (CS.isReturnNonNull())
      return true;

  return false;
}

static bool isKnownNonNullFromDominatingCondition(const Value *V,
                                                  const Instruction *CtxI,
                                                  const DominatorTree *DT) {
  assert(V->getType()->isPointerTy() && "V must be pointer type");
  assert(!isa<ConstantData>(V) && "Did not expect ConstantPointerNull");

  unsigned NumUsesExplored = 0;
  for (auto *U : V->users()) {
    // Avoid massive lists
    if (NumUsesExplored >= DomConditionsMaxUses)
      break;
    NumUsesExplored++;
    // Consider only compare instructions uniquely controlling a branch
    CmpInst::Predicate Pred;
    if (!match(const_cast<User *>(U),
               m_c_ICmp(Pred, m_Specific(V), m_Zero())) ||
        (Pred != ICmpInst::ICMP_EQ && Pred != ICmpInst::ICMP_NE))
      continue;

    for (auto *CmpU : U->users()) {
      if (const BranchInst *BI = dyn_cast<BranchInst>(CmpU)) {
        assert(BI->isConditional() && "uses a comparison!");

        BasicBlock *NonNullSuccessor =
            BI->getSuccessor(Pred == ICmpInst::ICMP_EQ ? 1 : 0);
        BasicBlockEdge Edge(BI->getParent(), NonNullSuccessor);
        if (Edge.isSingleEdge() && DT->dominates(Edge, CtxI->getParent()))
          return true;
      } else if (Pred == ICmpInst::ICMP_NE &&
                 match(CmpU, m_Intrinsic<Intrinsic::experimental_guard>()) &&
                 DT->dominates(cast<Instruction>(CmpU), CtxI)) {
        return true;
      }
    }
  }

  return false;
}

bool llvm::isKnownNonNullAt(const Value *V, const Instruction *CtxI,
                            const DominatorTree *DT) {
  if (isa<ConstantPointerNull>(V) || isa<UndefValue>(V))
    return false;

  if (isKnownNonNull(V))
    return true;

  return CtxI ? ::isKnownNonNullFromDominatingCondition(V, CtxI, DT) : false;
}

OverflowResult llvm::computeOverflowForUnsignedMul(const Value *LHS,
                                                   const Value *RHS,
                                                   const DataLayout &DL,
                                                   AssumptionCache *AC,
                                                   const Instruction *CxtI,
                                                   const DominatorTree *DT) {
  // Multiplying n * m significant bits yields a result of n + m significant
  // bits. If the total number of significant bits does not exceed the
  // result bit width (minus 1), there is no overflow.
  // This means if we have enough leading zero bits in the operands
  // we can guarantee that the result does not overflow.
  // Ref: "Hacker's Delight" by Henry Warren
  unsigned BitWidth = LHS->getType()->getScalarSizeInBits();
  APInt LHSKnownZero(BitWidth, 0);
  APInt LHSKnownOne(BitWidth, 0);
  APInt RHSKnownZero(BitWidth, 0);
  APInt RHSKnownOne(BitWidth, 0);
  computeKnownBits(LHS, LHSKnownZero, LHSKnownOne, DL, /*Depth=*/0, AC, CxtI,
                   DT);
  computeKnownBits(RHS, RHSKnownZero, RHSKnownOne, DL, /*Depth=*/0, AC, CxtI,
                   DT);
  // Note that underestimating the number of zero bits gives a more
  // conservative answer.
  unsigned ZeroBits = LHSKnownZero.countLeadingOnes() +
                      RHSKnownZero.countLeadingOnes();
  // First handle the easy case: if we have enough zero bits there's
  // definitely no overflow.
  if (ZeroBits >= BitWidth)
    return OverflowResult::NeverOverflows;

  // Get the largest possible values for each operand.
  APInt LHSMax = ~LHSKnownZero;
  APInt RHSMax = ~RHSKnownZero;

  // We know the multiply operation doesn't overflow if the maximum values for
  // each operand will not overflow after we multiply them together.
  bool MaxOverflow;
  LHSMax.umul_ov(RHSMax, MaxOverflow);
  if (!MaxOverflow)
    return OverflowResult::NeverOverflows;

  // We know it always overflows if multiplying the smallest possible values for
  // the operands also results in overflow.
  bool MinOverflow;
  LHSKnownOne.umul_ov(RHSKnownOne, MinOverflow);
  if (MinOverflow)
    return OverflowResult::AlwaysOverflows;

  return OverflowResult::MayOverflow;
}

OverflowResult llvm::computeOverflowForUnsignedAdd(const Value *LHS,
                                                   const Value *RHS,
                                                   const DataLayout &DL,
                                                   AssumptionCache *AC,
                                                   const Instruction *CxtI,
                                                   const DominatorTree *DT) {
  bool LHSKnownNonNegative, LHSKnownNegative;
  ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, DL, /*Depth=*/0,
                 AC, CxtI, DT);
  if (LHSKnownNonNegative || LHSKnownNegative) {
    bool RHSKnownNonNegative, RHSKnownNegative;
    ComputeSignBit(RHS, RHSKnownNonNegative, RHSKnownNegative, DL, /*Depth=*/0,
                   AC, CxtI, DT);

    if (LHSKnownNegative && RHSKnownNegative) {
      // The sign bit is set in both cases: this MUST overflow.
      // Create a simple add instruction, and insert it into the struct.
      return OverflowResult::AlwaysOverflows;
    }

    if (LHSKnownNonNegative && RHSKnownNonNegative) {
      // The sign bit is clear in both cases: this CANNOT overflow.
      // Create a simple add instruction, and insert it into the struct.
      return OverflowResult::NeverOverflows;
    }
  }

  return OverflowResult::MayOverflow;
}

static OverflowResult computeOverflowForSignedAdd(const Value *LHS,
                                                  const Value *RHS,
                                                  const AddOperator *Add,
                                                  const DataLayout &DL,
                                                  AssumptionCache *AC,
                                                  const Instruction *CxtI,
                                                  const DominatorTree *DT) {
  if (Add && Add->hasNoSignedWrap()) {
    return OverflowResult::NeverOverflows;
  }

  bool LHSKnownNonNegative, LHSKnownNegative;
  bool RHSKnownNonNegative, RHSKnownNegative;
  ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, DL, /*Depth=*/0,
                 AC, CxtI, DT);
  ComputeSignBit(RHS, RHSKnownNonNegative, RHSKnownNegative, DL, /*Depth=*/0,
                 AC, CxtI, DT);

  if ((LHSKnownNonNegative && RHSKnownNegative) ||
      (LHSKnownNegative && RHSKnownNonNegative)) {
    // The sign bits are opposite: this CANNOT overflow.
    return OverflowResult::NeverOverflows;
  }

  // The remaining code needs Add to be available. Early returns if not so.
  if (!Add)
    return OverflowResult::MayOverflow;

  // If the sign of Add is the same as at least one of the operands, this add
  // CANNOT overflow. This is particularly useful when the sum is
  // @llvm.assume'ed non-negative rather than proved so from analyzing its
  // operands.
  bool LHSOrRHSKnownNonNegative =
      (LHSKnownNonNegative || RHSKnownNonNegative);
  bool LHSOrRHSKnownNegative = (LHSKnownNegative || RHSKnownNegative);
  if (LHSOrRHSKnownNonNegative || LHSOrRHSKnownNegative) {
    bool AddKnownNonNegative, AddKnownNegative;
    ComputeSignBit(Add, AddKnownNonNegative, AddKnownNegative, DL,
                   /*Depth=*/0, AC, CxtI, DT);
    if ((AddKnownNonNegative && LHSOrRHSKnownNonNegative) ||
        (AddKnownNegative && LHSOrRHSKnownNegative)) {
      return OverflowResult::NeverOverflows;
    }
  }

  return OverflowResult::MayOverflow;
}

bool llvm::isOverflowIntrinsicNoWrap(const IntrinsicInst *II,
                                     const DominatorTree &DT) {
#ifndef NDEBUG
  auto IID = II->getIntrinsicID();
  assert((IID == Intrinsic::sadd_with_overflow ||
          IID == Intrinsic::uadd_with_overflow ||
          IID == Intrinsic::ssub_with_overflow ||
          IID == Intrinsic::usub_with_overflow ||
          IID == Intrinsic::smul_with_overflow ||
          IID == Intrinsic::umul_with_overflow) &&
         "Not an overflow intrinsic!");
#endif

  SmallVector<const BranchInst *, 2> GuardingBranches;
  SmallVector<const ExtractValueInst *, 2> Results;

  for (const User *U : II->users()) {
    if (const auto *EVI = dyn_cast<ExtractValueInst>(U)) {
      assert(EVI->getNumIndices() == 1 && "Obvious from CI's type");

      if (EVI->getIndices()[0] == 0)
        Results.push_back(EVI);
      else {
        assert(EVI->getIndices()[0] == 1 && "Obvious from CI's type");

        for (const auto *U : EVI->users())
          if (const auto *B = dyn_cast<BranchInst>(U)) {
            assert(B->isConditional() && "How else is it using an i1?");
            GuardingBranches.push_back(B);
          }
      }
    } else {
      // We are using the aggregate directly in a way we don't want to analyze
      // here (storing it to a global, say).
      return false;
    }
  }

  auto AllUsesGuardedByBranch = [&](const BranchInst *BI) {
    BasicBlockEdge NoWrapEdge(BI->getParent(), BI->getSuccessor(1));
    if (!NoWrapEdge.isSingleEdge())
      return false;

    // Check if all users of the add are provably no-wrap.
    for (const auto *Result : Results) {
      // If the extractvalue itself is not executed on overflow, the we don't
      // need to check each use separately, since domination is transitive.
      if (DT.dominates(NoWrapEdge, Result->getParent()))
        continue;

      for (auto &RU : Result->uses())
        if (!DT.dominates(NoWrapEdge, RU))
          return false;
    }

    return true;
  };

  return any_of(GuardingBranches, AllUsesGuardedByBranch);
}


OverflowResult llvm::computeOverflowForSignedAdd(const AddOperator *Add,
                                                 const DataLayout &DL,
                                                 AssumptionCache *AC,
                                                 const Instruction *CxtI,
                                                 const DominatorTree *DT) {
  return ::computeOverflowForSignedAdd(Add->getOperand(0), Add->getOperand(1),
                                       Add, DL, AC, CxtI, DT);
}

OverflowResult llvm::computeOverflowForSignedAdd(const Value *LHS,
                                                 const Value *RHS,
                                                 const DataLayout &DL,
                                                 AssumptionCache *AC,
                                                 const Instruction *CxtI,
                                                 const DominatorTree *DT) {
  return ::computeOverflowForSignedAdd(LHS, RHS, nullptr, DL, AC, CxtI, DT);
}

bool llvm::isGuaranteedToTransferExecutionToSuccessor(const Instruction *I) {
  // A memory operation returns normally if it isn't volatile. A volatile
  // operation is allowed to trap.
  //
  // An atomic operation isn't guaranteed to return in a reasonable amount of
  // time because it's possible for another thread to interfere with it for an
  // arbitrary length of time, but programs aren't allowed to rely on that.
  if (const LoadInst *LI = dyn_cast<LoadInst>(I))
    return !LI->isVolatile();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I))
    return !SI->isVolatile();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I))
    return !CXI->isVolatile();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I))
    return !RMWI->isVolatile();
  if (const MemIntrinsic *MII = dyn_cast<MemIntrinsic>(I))
    return !MII->isVolatile();

  // If there is no successor, then execution can't transfer to it.
  if (const auto *CRI = dyn_cast<CleanupReturnInst>(I))
    return !CRI->unwindsToCaller();
  if (const auto *CatchSwitch = dyn_cast<CatchSwitchInst>(I))
    return !CatchSwitch->unwindsToCaller();
  if (isa<ResumeInst>(I))
    return false;
  if (isa<ReturnInst>(I))
    return false;

  // Calls can throw, or contain an infinite loop, or kill the process.
  if (CallSite CS = CallSite(const_cast<Instruction*>(I))) {
    // Calls which don't write to arbitrary memory are safe.
    // FIXME: Ignoring infinite loops without any side-effects is too aggressive,
    // but it's consistent with other passes. See http://llvm.org/PR965 .
    // FIXME: This isn't aggressive enough; a call which only writes to a
    // global is guaranteed to return.
    return CS.onlyReadsMemory() || CS.onlyAccessesArgMemory() ||
           match(I, m_Intrinsic<Intrinsic::assume>());
  }

  // Other instructions return normally.
  return true;
}

bool llvm::isGuaranteedToExecuteForEveryIteration(const Instruction *I,
                                                  const Loop *L) {
  // The loop header is guaranteed to be executed for every iteration.
  //
  // FIXME: Relax this constraint to cover all basic blocks that are
  // guaranteed to be executed at every iteration.
  if (I->getParent() != L->getHeader()) return false;

  for (const Instruction &LI : *L->getHeader()) {
    if (&LI == I) return true;
    if (!isGuaranteedToTransferExecutionToSuccessor(&LI)) return false;
  }
  llvm_unreachable("Instruction not contained in its own parent basic block.");
}

bool llvm::propagatesFullPoison(const Instruction *I) {
  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Xor:
    case Instruction::Trunc:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      // These operations all propagate poison unconditionally. Note that poison
      // is not any particular value, so xor or subtraction of poison with
      // itself still yields poison, not zero.
      return true;

    case Instruction::AShr:
    case Instruction::SExt:
      // For these operations, one bit of the input is replicated across
      // multiple output bits. A replicated poison bit is still poison.
      return true;

    case Instruction::Shl: {
      // Left shift *by* a poison value is poison. The number of
      // positions to shift is unsigned, so no negative values are
      // possible there. Left shift by zero places preserves poison. So
      // it only remains to consider left shift of poison by a positive
      // number of places.
      //
      // A left shift by a positive number of places leaves the lowest order bit
      // non-poisoned. However, if such a shift has a no-wrap flag, then we can
      // make the poison operand violate that flag, yielding a fresh full-poison
      // value.
      auto *OBO = cast<OverflowingBinaryOperator>(I);
      return OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap();
    }

    case Instruction::Mul: {
      // A multiplication by zero yields a non-poison zero result, so we need to
      // rule out zero as an operand. Conservatively, multiplication by a
      // non-zero constant is not multiplication by zero.
      //
      // Multiplication by a non-zero constant can leave some bits
      // non-poisoned. For example, a multiplication by 2 leaves the lowest
      // order bit unpoisoned. So we need to consider that.
      //
      // Multiplication by 1 preserves poison. If the multiplication has a
      // no-wrap flag, then we can make the poison operand violate that flag
      // when multiplied by any integer other than 0 and 1.
      auto *OBO = cast<OverflowingBinaryOperator>(I);
      if (OBO->hasNoUnsignedWrap() || OBO->hasNoSignedWrap()) {
        for (Value *V : OBO->operands()) {
          if (auto *CI = dyn_cast<ConstantInt>(V)) {
            // A ConstantInt cannot yield poison, so we can assume that it is
            // the other operand that is poison.
            return !CI->isZero();
          }
        }
      }
      return false;
    }

    case Instruction::ICmp:
      // Comparing poison with any value yields poison.  This is why, for
      // instance, x s< (x +nsw 1) can be folded to true.
      return true;

    case Instruction::GetElementPtr:
      // A GEP implicitly represents a sequence of additions, subtractions,
      // truncations, sign extensions and multiplications. The multiplications
      // are by the non-zero sizes of some set of types, so we do not have to be
      // concerned with multiplication by zero. If the GEP is in-bounds, then
      // these operations are implicitly no-signed-wrap so poison is propagated
      // by the arguments above for Add, Sub, Trunc, SExt and Mul.
      return cast<GEPOperator>(I)->isInBounds();

    default:
      return false;
  }
}

const Value *llvm::getGuaranteedNonFullPoisonOp(const Instruction *I) {
  switch (I->getOpcode()) {
    case Instruction::Store:
      return cast<StoreInst>(I)->getPointerOperand();

    case Instruction::Load:
      return cast<LoadInst>(I)->getPointerOperand();

    case Instruction::AtomicCmpXchg:
      return cast<AtomicCmpXchgInst>(I)->getPointerOperand();

    case Instruction::AtomicRMW:
      return cast<AtomicRMWInst>(I)->getPointerOperand();

    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      return I->getOperand(1);

    default:
      return nullptr;
  }
}

bool llvm::isKnownNotFullPoison(const Instruction *PoisonI) {
  // We currently only look for uses of poison values within the same basic
  // block, as that makes it easier to guarantee that the uses will be
  // executed given that PoisonI is executed.
  //
  // FIXME: Expand this to consider uses beyond the same basic block. To do
  // this, look out for the distinction between post-dominance and strong
  // post-dominance.
  const BasicBlock *BB = PoisonI->getParent();

  // Set of instructions that we have proved will yield poison if PoisonI
  // does.
  SmallSet<const Value *, 16> YieldsPoison;
  SmallSet<const BasicBlock *, 4> Visited;
  YieldsPoison.insert(PoisonI);
  Visited.insert(PoisonI->getParent());

  BasicBlock::const_iterator Begin = PoisonI->getIterator(), End = BB->end();

  unsigned Iter = 0;
  while (Iter++ < MaxDepth) {
    for (auto &I : make_range(Begin, End)) {
      if (&I != PoisonI) {
        const Value *NotPoison = getGuaranteedNonFullPoisonOp(&I);
        if (NotPoison != nullptr && YieldsPoison.count(NotPoison))
          return true;
        if (!isGuaranteedToTransferExecutionToSuccessor(&I))
          return false;
      }

      // Mark poison that propagates from I through uses of I.
      if (YieldsPoison.count(&I)) {
        for (const User *User : I.users()) {
          const Instruction *UserI = cast<Instruction>(User);
          if (propagatesFullPoison(UserI))
            YieldsPoison.insert(User);
        }
      }
    }

    if (auto *NextBB = BB->getSingleSuccessor()) {
      if (Visited.insert(NextBB).second) {
        BB = NextBB;
        Begin = BB->getFirstNonPHI()->getIterator();
        End = BB->end();
        continue;
      }
    }

    break;
  };
  return false;
}

static bool isKnownNonNaN(const Value *V, FastMathFlags FMF) {
  if (FMF.noNaNs())
    return true;

  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isNaN();
  return false;
}

static bool isKnownNonZero(const Value *V) {
  if (auto *C = dyn_cast<ConstantFP>(V))
    return !C->isZero();
  return false;
}

static SelectPatternResult matchSelectPattern(CmpInst::Predicate Pred,
                                              FastMathFlags FMF,
                                              Value *CmpLHS, Value *CmpRHS,
                                              Value *TrueVal, Value *FalseVal,
                                              Value *&LHS, Value *&RHS) {
  LHS = CmpLHS;
  RHS = CmpRHS;

  // If the predicate is an "or-equal"  (FP) predicate, then signed zeroes may
  // return inconsistent results between implementations.
  //   (0.0 <= -0.0) ? 0.0 : -0.0 // Returns 0.0
  //   minNum(0.0, -0.0)          // May return -0.0 or 0.0 (IEEE 754-2008 5.3.1)
  // Therefore we behave conservatively and only proceed if at least one of the
  // operands is known to not be zero, or if we don't care about signed zeroes.
  switch (Pred) {
  default: break;
  case CmpInst::FCMP_OGE: case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_UGE: case CmpInst::FCMP_ULE:
    if (!FMF.noSignedZeros() && !isKnownNonZero(CmpLHS) &&
        !isKnownNonZero(CmpRHS))
      return {SPF_UNKNOWN, SPNB_NA, false};
  }

  SelectPatternNaNBehavior NaNBehavior = SPNB_NA;
  bool Ordered = false;

  // When given one NaN and one non-NaN input:
  //   - maxnum/minnum (C99 fmaxf()/fminf()) return the non-NaN input.
  //   - A simple C99 (a < b ? a : b) construction will return 'b' (as the
  //     ordered comparison fails), which could be NaN or non-NaN.
  // so here we discover exactly what NaN behavior is required/accepted.
  if (CmpInst::isFPPredicate(Pred)) {
    bool LHSSafe = isKnownNonNaN(CmpLHS, FMF);
    bool RHSSafe = isKnownNonNaN(CmpRHS, FMF);

    if (LHSSafe && RHSSafe) {
      // Both operands are known non-NaN.
      NaNBehavior = SPNB_RETURNS_ANY;
    } else if (CmpInst::isOrdered(Pred)) {
      // An ordered comparison will return false when given a NaN, so it
      // returns the RHS.
      Ordered = true;
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then NaN will be returned.
        NaNBehavior = SPNB_RETURNS_NAN;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_OTHER;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    } else {
      Ordered = false;
      // An unordered comparison will return true when given a NaN, so it
      // returns the LHS.
      if (LHSSafe)
        // LHS is non-NaN, so if RHS is NaN then non-NaN will be returned.
        NaNBehavior = SPNB_RETURNS_OTHER;
      else if (RHSSafe)
        NaNBehavior = SPNB_RETURNS_NAN;
      else
        // Completely unsafe.
        return {SPF_UNKNOWN, SPNB_NA, false};
    }
  }

  if (TrueVal == CmpRHS && FalseVal == CmpLHS) {
    std::swap(CmpLHS, CmpRHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
    if (NaNBehavior == SPNB_RETURNS_NAN)
      NaNBehavior = SPNB_RETURNS_OTHER;
    else if (NaNBehavior == SPNB_RETURNS_OTHER)
      NaNBehavior = SPNB_RETURNS_NAN;
    Ordered = !Ordered;
  }

  // ([if]cmp X, Y) ? X : Y
  if (TrueVal == CmpLHS && FalseVal == CmpRHS) {
    switch (Pred) {
    default: return {SPF_UNKNOWN, SPNB_NA, false}; // Equality.
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return {SPF_UMAX, SPNB_NA, false};
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return {SPF_SMAX, SPNB_NA, false};
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE: return {SPF_UMIN, SPNB_NA, false};
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: return {SPF_SMIN, SPNB_NA, false};
    case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_OGT:
    case FCmpInst::FCMP_OGE: return {SPF_FMAXNUM, NaNBehavior, Ordered};
    case FCmpInst::FCMP_ULT:
    case FCmpInst::FCMP_ULE:
    case FCmpInst::FCMP_OLT:
    case FCmpInst::FCMP_OLE: return {SPF_FMINNUM, NaNBehavior, Ordered};
    }
  }

  if (ConstantInt *C1 = dyn_cast<ConstantInt>(CmpRHS)) {
    if ((CmpLHS == TrueVal && match(FalseVal, m_Neg(m_Specific(CmpLHS)))) ||
        (CmpLHS == FalseVal && match(TrueVal, m_Neg(m_Specific(CmpLHS))))) {

      // ABS(X) ==> (X >s 0) ? X : -X and (X >s -1) ? X : -X
      // NABS(X) ==> (X >s 0) ? -X : X and (X >s -1) ? -X : X
      if (Pred == ICmpInst::ICMP_SGT && (C1->isZero() || C1->isMinusOne())) {
        return {(CmpLHS == TrueVal) ? SPF_ABS : SPF_NABS, SPNB_NA, false};
      }

      // ABS(X) ==> (X <s 0) ? -X : X and (X <s 1) ? -X : X
      // NABS(X) ==> (X <s 0) ? X : -X and (X <s 1) ? X : -X
      if (Pred == ICmpInst::ICMP_SLT && (C1->isZero() || C1->isOne())) {
        return {(CmpLHS == FalseVal) ? SPF_ABS : SPF_NABS, SPNB_NA, false};
      }
    }

    // Y >s C ? ~Y : ~C == ~Y <s ~C ? ~Y : ~C = SMIN(~Y, ~C)
    if (const auto *C2 = dyn_cast<ConstantInt>(FalseVal)) {
      if (Pred == ICmpInst::ICMP_SGT && C1->getType() == C2->getType() &&
          ~C1->getValue() == C2->getValue() &&
          (match(TrueVal, m_Not(m_Specific(CmpLHS))) ||
           match(CmpLHS, m_Not(m_Specific(TrueVal))))) {
        LHS = TrueVal;
        RHS = FalseVal;
        return {SPF_SMIN, SPNB_NA, false};
      }
    }
  }

  // TODO: (X > 4) ? X : 5   -->  (X >= 5) ? X : 5  -->  MAX(X, 5)

  return {SPF_UNKNOWN, SPNB_NA, false};
}

static Value *lookThroughCast(CmpInst *CmpI, Value *V1, Value *V2,
                              Instruction::CastOps *CastOp) {
  CastInst *CI = dyn_cast<CastInst>(V1);
  Constant *C = dyn_cast<Constant>(V2);
  if (!CI)
    return nullptr;
  *CastOp = CI->getOpcode();

  if (auto *CI2 = dyn_cast<CastInst>(V2)) {
    // If V1 and V2 are both the same cast from the same type, we can look
    // through V1.
    if (CI2->getOpcode() == CI->getOpcode() &&
        CI2->getSrcTy() == CI->getSrcTy())
      return CI2->getOperand(0);
    return nullptr;
  } else if (!C) {
    return nullptr;
  }

  Constant *CastedTo = nullptr;

  if (isa<ZExtInst>(CI) && CmpI->isUnsigned())
    CastedTo = ConstantExpr::getTrunc(C, CI->getSrcTy());

  if (isa<SExtInst>(CI) && CmpI->isSigned())
    CastedTo = ConstantExpr::getTrunc(C, CI->getSrcTy(), true);

  if (isa<TruncInst>(CI))
    CastedTo = ConstantExpr::getIntegerCast(C, CI->getSrcTy(), CmpI->isSigned());

  if (isa<FPTruncInst>(CI))
    CastedTo = ConstantExpr::getFPExtend(C, CI->getSrcTy(), true);

  if (isa<FPExtInst>(CI))
    CastedTo = ConstantExpr::getFPTrunc(C, CI->getSrcTy(), true);

  if (isa<FPToUIInst>(CI))
    CastedTo = ConstantExpr::getUIToFP(C, CI->getSrcTy(), true);

  if (isa<FPToSIInst>(CI))
    CastedTo = ConstantExpr::getSIToFP(C, CI->getSrcTy(), true);

  if (isa<UIToFPInst>(CI))
    CastedTo = ConstantExpr::getFPToUI(C, CI->getSrcTy(), true);

  if (isa<SIToFPInst>(CI))
    CastedTo = ConstantExpr::getFPToSI(C, CI->getSrcTy(), true);

  if (!CastedTo)
    return nullptr;

  Constant *CastedBack =
      ConstantExpr::getCast(CI->getOpcode(), CastedTo, C->getType(), true);
  // Make sure the cast doesn't lose any information.
  if (CastedBack != C)
    return nullptr;

  return CastedTo;
}

SelectPatternResult llvm::matchSelectPattern(Value *V, Value *&LHS, Value *&RHS,
                                             Instruction::CastOps *CastOp) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI) return {SPF_UNKNOWN, SPNB_NA, false};

  CmpInst *CmpI = dyn_cast<CmpInst>(SI->getCondition());
  if (!CmpI) return {SPF_UNKNOWN, SPNB_NA, false};

  CmpInst::Predicate Pred = CmpI->getPredicate();
  Value *CmpLHS = CmpI->getOperand(0);
  Value *CmpRHS = CmpI->getOperand(1);
  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();
  FastMathFlags FMF;
  if (isa<FPMathOperator>(CmpI))
    FMF = CmpI->getFastMathFlags();

  // Bail out early.
  if (CmpI->isEquality())
    return {SPF_UNKNOWN, SPNB_NA, false};

  // Deal with type mismatches.
  if (CastOp && CmpLHS->getType() != TrueVal->getType()) {
    if (Value *C = lookThroughCast(CmpI, TrueVal, FalseVal, CastOp))
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS,
                                  cast<CastInst>(TrueVal)->getOperand(0), C,
                                  LHS, RHS);
    if (Value *C = lookThroughCast(CmpI, FalseVal, TrueVal, CastOp))
      return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS,
                                  C, cast<CastInst>(FalseVal)->getOperand(0),
                                  LHS, RHS);
  }
  return ::matchSelectPattern(Pred, FMF, CmpLHS, CmpRHS, TrueVal, FalseVal,
                              LHS, RHS);
}

ConstantRange llvm::getConstantRangeFromMetadata(const MDNode &Ranges) {
  const unsigned NumRanges = Ranges.getNumOperands() / 2;
  assert(NumRanges >= 1 && "Must have at least one range!");
  assert(Ranges.getNumOperands() % 2 == 0 && "Must be a sequence of pairs");

  auto *FirstLow = mdconst::extract<ConstantInt>(Ranges.getOperand(0));
  auto *FirstHigh = mdconst::extract<ConstantInt>(Ranges.getOperand(1));

  ConstantRange CR(FirstLow->getValue(), FirstHigh->getValue());

  for (unsigned i = 1; i < NumRanges; ++i) {
    auto *Low = mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 0));
    auto *High = mdconst::extract<ConstantInt>(Ranges.getOperand(2 * i + 1));

    // Note: unionWith will potentially create a range that contains values not
    // contained in any of the original N ranges.
    CR = CR.unionWith(ConstantRange(Low->getValue(), High->getValue()));
  }

  return CR;
}

/// Return true if "icmp Pred LHS RHS" is always true.
static bool isTruePredicate(CmpInst::Predicate Pred,
                            const Value *LHS, const Value *RHS,
                            const DataLayout &DL, unsigned Depth,
                            AssumptionCache *AC, const Instruction *CxtI,
                            const DominatorTree *DT) {
  assert(!LHS->getType()->isVectorTy() && "TODO: extend to handle vectors!");
  if (ICmpInst::isTrueWhenEqual(Pred) && LHS == RHS)
    return true;

  switch (Pred) {
  default:
    return false;

  case CmpInst::ICMP_SLE: {
    const APInt *C;

    // LHS s<= LHS +_{nsw} C   if C >= 0
    if (match(RHS, m_NSWAdd(m_Specific(LHS), m_APInt(C))))
      return !C->isNegative();
    return false;
  }

  case CmpInst::ICMP_ULE: {
    const APInt *C;

    // LHS u<= LHS +_{nuw} C   for any C
    if (match(RHS, m_NUWAdd(m_Specific(LHS), m_APInt(C))))
      return true;

    // Match A to (X +_{nuw} CA) and B to (X +_{nuw} CB)
    auto MatchNUWAddsToSameValue = [&](const Value *A, const Value *B,
                                       const Value *&X,
                                       const APInt *&CA, const APInt *&CB) {
      if (match(A, m_NUWAdd(m_Value(X), m_APInt(CA))) &&
          match(B, m_NUWAdd(m_Specific(X), m_APInt(CB))))
        return true;

      // If X & C == 0 then (X | C) == X +_{nuw} C
      if (match(A, m_Or(m_Value(X), m_APInt(CA))) &&
          match(B, m_Or(m_Specific(X), m_APInt(CB)))) {
        unsigned BitWidth = CA->getBitWidth();
        APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
        computeKnownBits(X, KnownZero, KnownOne, DL, Depth + 1, AC, CxtI, DT);

        if ((KnownZero & *CA) == *CA && (KnownZero & *CB) == *CB)
          return true;
      }

      return false;
    };

    const Value *X;
    const APInt *CLHS, *CRHS;
    if (MatchNUWAddsToSameValue(LHS, RHS, X, CLHS, CRHS))
      return CLHS->ule(*CRHS);

    return false;
  }
  }
}

/// Return true if "icmp Pred BLHS BRHS" is true whenever "icmp Pred
/// ALHS ARHS" is true.  Otherwise, return None.
static Optional<bool>
isImpliedCondOperands(CmpInst::Predicate Pred, const Value *ALHS,
                      const Value *ARHS, const Value *BLHS,
                      const Value *BRHS, const DataLayout &DL,
                      unsigned Depth, AssumptionCache *AC,
                      const Instruction *CxtI, const DominatorTree *DT) {
  switch (Pred) {
  default:
    return None;

  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    if (isTruePredicate(CmpInst::ICMP_SLE, BLHS, ALHS, DL, Depth, AC, CxtI,
                        DT) &&
        isTruePredicate(CmpInst::ICMP_SLE, ARHS, BRHS, DL, Depth, AC, CxtI, DT))
      return true;
    return None;

  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
    if (isTruePredicate(CmpInst::ICMP_ULE, BLHS, ALHS, DL, Depth, AC, CxtI,
                        DT) &&
        isTruePredicate(CmpInst::ICMP_ULE, ARHS, BRHS, DL, Depth, AC, CxtI, DT))
      return true;
    return None;
  }
}

/// Return true if the operands of the two compares match.  IsSwappedOps is true
/// when the operands match, but are swapped.
static bool isMatchingOps(const Value *ALHS, const Value *ARHS,
                          const Value *BLHS, const Value *BRHS,
                          bool &IsSwappedOps) {

  bool IsMatchingOps = (ALHS == BLHS && ARHS == BRHS);
  IsSwappedOps = (ALHS == BRHS && ARHS == BLHS);
  return IsMatchingOps || IsSwappedOps;
}

/// Return true if "icmp1 APred ALHS ARHS" implies "icmp2 BPred BLHS BRHS" is
/// true.  Return false if "icmp1 APred ALHS ARHS" implies "icmp2 BPred BLHS
/// BRHS" is false.  Otherwise, return None if we can't infer anything.
static Optional<bool> isImpliedCondMatchingOperands(CmpInst::Predicate APred,
                                                    const Value *ALHS,
                                                    const Value *ARHS,
                                                    CmpInst::Predicate BPred,
                                                    const Value *BLHS,
                                                    const Value *BRHS,
                                                    bool IsSwappedOps) {
  // Canonicalize the operands so they're matching.
  if (IsSwappedOps) {
    std::swap(BLHS, BRHS);
    BPred = ICmpInst::getSwappedPredicate(BPred);
  }
  if (CmpInst::isImpliedTrueByMatchingCmp(APred, BPred))
    return true;
  if (CmpInst::isImpliedFalseByMatchingCmp(APred, BPred))
    return false;

  return None;
}

/// Return true if "icmp1 APred ALHS C1" implies "icmp2 BPred BLHS C2" is
/// true.  Return false if "icmp1 APred ALHS C1" implies "icmp2 BPred BLHS
/// C2" is false.  Otherwise, return None if we can't infer anything.
static Optional<bool>
isImpliedCondMatchingImmOperands(CmpInst::Predicate APred, const Value *ALHS,
                                 const ConstantInt *C1,
                                 CmpInst::Predicate BPred,
                                 const Value *BLHS, const ConstantInt *C2) {
  assert(ALHS == BLHS && "LHS operands must match.");
  ConstantRange DomCR =
      ConstantRange::makeExactICmpRegion(APred, C1->getValue());
  ConstantRange CR =
      ConstantRange::makeAllowedICmpRegion(BPred, C2->getValue());
  ConstantRange Intersection = DomCR.intersectWith(CR);
  ConstantRange Difference = DomCR.difference(CR);
  if (Intersection.isEmptySet())
    return false;
  if (Difference.isEmptySet())
    return true;
  return None;
}

Optional<bool> llvm::isImpliedCondition(const Value *LHS, const Value *RHS,
                                        const DataLayout &DL, bool InvertAPred,
                                        unsigned Depth, AssumptionCache *AC,
                                        const Instruction *CxtI,
                                        const DominatorTree *DT) {
  // A mismatch occurs when we compare a scalar cmp to a vector cmp, for example.
  if (LHS->getType() != RHS->getType())
    return None;

  Type *OpTy = LHS->getType();
  assert(OpTy->getScalarType()->isIntegerTy(1));

  // LHS ==> RHS by definition
  if (!InvertAPred && LHS == RHS)
    return true;

  if (OpTy->isVectorTy())
    // TODO: extending the code below to handle vectors
    return None;
  assert(OpTy->isIntegerTy(1) && "implied by above");

  ICmpInst::Predicate APred, BPred;
  Value *ALHS, *ARHS;
  Value *BLHS, *BRHS;

  if (!match(LHS, m_ICmp(APred, m_Value(ALHS), m_Value(ARHS))) ||
      !match(RHS, m_ICmp(BPred, m_Value(BLHS), m_Value(BRHS))))
    return None;

  if (InvertAPred)
    APred = CmpInst::getInversePredicate(APred);

  // Can we infer anything when the two compares have matching operands?
  bool IsSwappedOps;
  if (isMatchingOps(ALHS, ARHS, BLHS, BRHS, IsSwappedOps)) {
    if (Optional<bool> Implication = isImpliedCondMatchingOperands(
            APred, ALHS, ARHS, BPred, BLHS, BRHS, IsSwappedOps))
      return Implication;
    // No amount of additional analysis will infer the second condition, so
    // early exit.
    return None;
  }

  // Can we infer anything when the LHS operands match and the RHS operands are
  // constants (not necessarily matching)?
  if (ALHS == BLHS && isa<ConstantInt>(ARHS) && isa<ConstantInt>(BRHS)) {
    if (Optional<bool> Implication = isImpliedCondMatchingImmOperands(
            APred, ALHS, cast<ConstantInt>(ARHS), BPred, BLHS,
            cast<ConstantInt>(BRHS)))
      return Implication;
    // No amount of additional analysis will infer the second condition, so
    // early exit.
    return None;
  }

  if (APred == BPred)
    return isImpliedCondOperands(APred, ALHS, ARHS, BLHS, BRHS, DL, Depth, AC,
                                 CxtI, DT);

  return None;
}
