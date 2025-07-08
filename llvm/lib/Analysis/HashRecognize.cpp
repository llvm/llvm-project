//===- HashRecognize.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The HashRecognize analysis recognizes unoptimized polynomial hash functions
// with operations over a Galois field of characteristic 2, also called binary
// fields, or GF(2^n): this class of hash functions can be optimized using a
// lookup-table-driven implementation, or with target-specific instructions.
// Examples:
//
//  1. Cyclic redundancy check (CRC), which is a polynomial division in GF(2).
//  2. Rabin fingerprint, a component of the Rabin-Karp algorithm, which is a
//     rolling hash polynomial division in GF(2).
//  3. Rijndael MixColumns, a step in AES computation, which is a polynomial
//     multiplication in GF(2^3).
//  4. GHASH, the authentication mechanism in AES Galois/Counter Mode (GCM),
//     which is a polynomial evaluation in GF(2^128).
//
// All of them use an irreducible generating polynomial of degree m,
//
//    c_m * x^m + c_(m-1) * x^(m-1) + ... + c_0 * x^0
//
// where each coefficient c is can take values in GF(2^n), where 2^n is termed
// the order of the Galois field. For GF(2), each coefficient can take values
// either 0 or 1, and the polynomial is simply represented by m+1 bits,
// corresponding to the coefficients. The different variants of CRC are named by
// degree of generating polynomial used: so CRC-32 would use a polynomial of
// degree 32.
//
// The reason algorithms on GF(2^n) can be optimized with a lookup-table is the
// following: in such fields, polynomial addition and subtraction are identical
// and equivalent to XOR, polynomial multiplication is an AND, and polynomial
// division is identity: the XOR and AND operations in unoptimized
// implementations are performed bit-wise, and can be optimized to be performed
// chunk-wise, by interleaving copies of the generating polynomial, and storing
// the pre-computed values in a table.
//
// A generating polynomial of m bits always has the MSB set, so we usually
// omit it. An example of a 16-bit polynomial is the CRC-16-CCITT polynomial:
//
//   (x^16) + x^12 + x^5 + 1 = (1) 0001 0000 0010 0001 = 0x1021
//
// Transmissions are either in big-endian or little-endian form, and hash
// algorithms are written according to this. For example, IEEE 802 and RS-232
// specify little-endian transmission.
//
//===----------------------------------------------------------------------===//
//
// At the moment, we only recognize the CRC algorithm.
// Documentation on CRC32 from the kernel:
// https://www.kernel.org/doc/Documentation/crc32.txt
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/HashRecognize.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;
using namespace PatternMatch;
using namespace SCEVPatternMatch;

#define DEBUG_TYPE "hash-recognize"

// KnownBits for a PHI node. There are at most two PHI nodes, corresponding to
// the Simple Recurrence and Conditional Recurrence. The IndVar PHI is not
// relevant.
using KnownPhiMap = SmallDenseMap<const PHINode *, KnownBits, 2>;

// A pair of a PHI node along with its incoming value from within a loop.
using PhiStepPair = std::pair<const PHINode *, const Instruction *>;

/// A much simpler version of ValueTracking, in that it computes KnownBits of
/// values, except that it computes the evolution of KnownBits in a loop with a
/// given trip count, and predication is specialized for a significant-bit
/// check.
class ValueEvolution {
  const unsigned TripCount;
  const bool ByteOrderSwapped;
  APInt GenPoly;
  StringRef ErrStr;

  // Compute the KnownBits of a BinaryOperator.
  KnownBits computeBinOp(const BinaryOperator *I);

  // Compute the KnownBits of an Instruction.
  KnownBits computeInstr(const Instruction *I);

  // Compute the KnownBits of a Value.
  KnownBits compute(const Value *V);

public:
  // ValueEvolution is meant to be constructed with the TripCount of the loop,
  // and whether the polynomial algorithm is big-endian, for the significant-bit
  // check.
  ValueEvolution(unsigned TripCount, bool ByteOrderSwapped);

  // Given a list of PHI nodes along with their incoming value from within the
  // loop, computeEvolutions computes the KnownBits of each of the PHI nodes on
  // the final iteration. Returns true on success and false on error.
  bool computeEvolutions(ArrayRef<PhiStepPair> PhiEvolutions);

  // In case ValueEvolution encounters an error, this is meant to be used for a
  // precise error message.
  StringRef getError() const { return ErrStr; }

  // The computed KnownBits for each PHI node, which is populated after
  // computeEvolutions is called.
  KnownPhiMap KnownPhis;
};

ValueEvolution::ValueEvolution(unsigned TripCount, bool ByteOrderSwapped)
    : TripCount(TripCount), ByteOrderSwapped(ByteOrderSwapped) {}

KnownBits ValueEvolution::computeBinOp(const BinaryOperator *I) {
  KnownBits KnownL(compute(I->getOperand(0)));
  KnownBits KnownR(compute(I->getOperand(1)));

  switch (I->getOpcode()) {
  case Instruction::BinaryOps::And:
    return KnownL & KnownR;
  case Instruction::BinaryOps::Or:
    return KnownL | KnownR;
  case Instruction::BinaryOps::Xor:
    return KnownL ^ KnownR;
  case Instruction::BinaryOps::Shl: {
    auto *OBO = cast<OverflowingBinaryOperator>(I);
    return KnownBits::shl(KnownL, KnownR, OBO->hasNoUnsignedWrap(),
                          OBO->hasNoSignedWrap());
  }
  case Instruction::BinaryOps::LShr:
    return KnownBits::lshr(KnownL, KnownR);
  case Instruction::BinaryOps::AShr:
    return KnownBits::ashr(KnownL, KnownR);
  case Instruction::BinaryOps::Add: {
    auto *OBO = cast<OverflowingBinaryOperator>(I);
    return KnownBits::add(KnownL, KnownR, OBO->hasNoUnsignedWrap(),
                          OBO->hasNoSignedWrap());
  }
  case Instruction::BinaryOps::Sub: {
    auto *OBO = cast<OverflowingBinaryOperator>(I);
    return KnownBits::sub(KnownL, KnownR, OBO->hasNoUnsignedWrap(),
                          OBO->hasNoSignedWrap());
  }
  case Instruction::BinaryOps::Mul: {
    Value *Op0 = I->getOperand(0);
    Value *Op1 = I->getOperand(1);
    bool SelfMultiply = Op0 == Op1 && isGuaranteedNotToBeUndef(Op0);
    return KnownBits::mul(KnownL, KnownR, SelfMultiply);
  }
  case Instruction::BinaryOps::UDiv:
    return KnownBits::udiv(KnownL, KnownR);
  case Instruction::BinaryOps::SDiv:
    return KnownBits::sdiv(KnownL, KnownR);
  case Instruction::BinaryOps::URem:
    return KnownBits::urem(KnownL, KnownR);
  case Instruction::BinaryOps::SRem:
    return KnownBits::srem(KnownL, KnownR);
  default:
    ErrStr = "Unknown BinaryOperator";
    unsigned BitWidth = I->getType()->getScalarSizeInBits();
    return {BitWidth};
  }
}

KnownBits ValueEvolution::computeInstr(const Instruction *I) {
  unsigned BitWidth = I->getType()->getScalarSizeInBits();

  // We look up in the map that contains the KnownBits of the PHI from the
  // previous iteration.
  if (const PHINode *P = dyn_cast<PHINode>(I))
    return KnownPhis.lookup_or(P, BitWidth);

  // Compute the KnownBits for a Select(Cmp()), forcing it to take the branch
  // that is predicated on the (least|most)-significant-bit check.
  CmpPredicate Pred;
  Value *L, *R, *TV, *FV;
  if (match(I, m_Select(m_ICmp(Pred, m_Value(L), m_Value(R)), m_Value(TV),
                        m_Value(FV)))) {
    // We need to check LCR against [0, 2) in the little-endian case, because
    // the RCR check is insufficient: it is simply [0, 1).
    if (!ByteOrderSwapped) {
      KnownBits KnownL = compute(L);
      unsigned ICmpBW = KnownL.getBitWidth();
      auto LCR = ConstantRange::fromKnownBits(KnownL, false);
      auto CheckLCR = ConstantRange(APInt::getZero(ICmpBW), APInt(ICmpBW, 2));
      if (LCR != CheckLCR) {
        ErrStr = "Bad LHS of significant-bit-check";
        return {BitWidth};
      }
    }

    // Check that the predication is on (most|least) significant bit.
    KnownBits KnownR = compute(R);
    unsigned ICmpBW = KnownR.getBitWidth();
    auto RCR = ConstantRange::fromKnownBits(KnownR, false);
    auto AllowedR = ConstantRange::makeAllowedICmpRegion(Pred, RCR);
    ConstantRange CheckRCR(APInt::getZero(ICmpBW),
                           ByteOrderSwapped ? APInt::getSignedMinValue(ICmpBW)
                                            : APInt(ICmpBW, 1));
    if (AllowedR == CheckRCR)
      return compute(TV);
    if (AllowedR.inverse() == CheckRCR)
      return compute(FV);

    ErrStr = "Bad RHS of significant-bit-check";
    return {BitWidth};
  }

  if (auto *BO = dyn_cast<BinaryOperator>(I))
    return computeBinOp(BO);

  switch (I->getOpcode()) {
  case Instruction::CastOps::Trunc:
    return compute(I->getOperand(0)).trunc(BitWidth);
  case Instruction::CastOps::ZExt:
    return compute(I->getOperand(0)).zext(BitWidth);
  case Instruction::CastOps::SExt:
    return compute(I->getOperand(0)).sext(BitWidth);
  default:
    ErrStr = "Unknown Instruction";
    return {BitWidth};
  }
}

KnownBits ValueEvolution::compute(const Value *V) {
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return KnownBits::makeConstant(CI->getValue());

  if (auto *I = dyn_cast<Instruction>(V))
    return computeInstr(I);

  ErrStr = "Unknown Value";
  unsigned BitWidth = V->getType()->getScalarSizeInBits();
  return {BitWidth};
}

bool ValueEvolution::computeEvolutions(ArrayRef<PhiStepPair> PhiEvolutions) {
  for (unsigned I = 0; I < TripCount; ++I)
    for (auto [Phi, Step] : PhiEvolutions)
      KnownPhis.emplace_or_assign(Phi, computeInstr(Step));

  return ErrStr.empty();
}

/// A structure that can hold either a Simple Recurrence or a Conditional
/// Recurrence. Note that in the case of a Simple Recurrence, Step is an operand
/// of the BO, while in a Conditional Recurrence, it is a SelectInst.
struct RecurrenceInfo {
  const Loop &L;
  const PHINode *Phi = nullptr;
  BinaryOperator *BO = nullptr;
  Value *Start = nullptr;
  Value *Step = nullptr;
  std::optional<APInt> ExtraConst;

  RecurrenceInfo(const Loop &L) : L(L) {}
  operator bool() const { return BO; }

  void print(raw_ostream &OS, unsigned Indent = 0) const {
    OS.indent(Indent) << "Phi: ";
    Phi->print(OS);
    OS << "\n";
    OS.indent(Indent) << "BinaryOperator: ";
    BO->print(OS);
    OS << "\n";
    OS.indent(Indent) << "Start: ";
    Start->print(OS);
    OS << "\n";
    OS.indent(Indent) << "Step: ";
    Step->print(OS);
    OS << "\n";
    if (ExtraConst) {
      OS.indent(Indent) << "ExtraConst: ";
      ExtraConst->print(OS, false);
      OS << "\n";
    }
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif

  bool matchSimpleRecurrence(const PHINode *P);
  bool matchConditionalRecurrence(
      const PHINode *P,
      Instruction::BinaryOps BOWithConstOpToMatch = Instruction::BinaryOpsEnd);

private:
  BinaryOperator *digRecurrence(
      Instruction *V,
      Instruction::BinaryOps BOWithConstOpToMatch = Instruction::BinaryOpsEnd);
};

/// Wraps llvm::matchSimpleRecurrence. Match a simple first order recurrence
/// cycle of the form:
///
/// loop:
///    %rec = phi [%start, %entry], [%BO, %loop]
///     ...
///     %BO = binop %rec, %step
///
/// or
///
/// loop:
///    %rec = phi [%start, %entry], [%BO, %loop]
///    ...
///    %BO = binop %step, %rec
///
bool RecurrenceInfo::matchSimpleRecurrence(const PHINode *P) {
  Phi = P;
  return llvm::matchSimpleRecurrence(Phi, BO, Start, Step);
}

/// Digs for a recurrence starting with \p V hitting the PHI node in a use-def
/// chain. Used by matchConditionalRecurrence.
BinaryOperator *
RecurrenceInfo::digRecurrence(Instruction *V,
                              Instruction::BinaryOps BOWithConstOpToMatch) {
  SmallVector<Instruction *> Worklist;
  Worklist.push_back(V);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();

    // Don't add a PHI's operands to the Worklist.
    if (isa<PHINode>(I))
      continue;

    // Find a recurrence over a BinOp, by matching either of its operands
    // with with the PHINode.
    if (match(I, m_c_BinOp(m_Value(), m_Specific(Phi))))
      return cast<BinaryOperator>(I);

    // Bind to ExtraConst, if we match exactly one.
    if (I->getOpcode() == BOWithConstOpToMatch) {
      if (ExtraConst)
        return nullptr;
      const APInt *C = nullptr;
      if (match(I, m_c_BinOp(m_APInt(C), m_Value())))
        ExtraConst = *C;
    }

    // Continue along the use-def chain.
    for (Use &U : I->operands())
      if (auto *UI = dyn_cast<Instruction>(U))
        if (L.contains(UI))
          Worklist.push_back(UI);
  }
  return nullptr;
}

/// A Conditional Recurrence is a recurrence of the form:
///
/// loop:
///    %rec = phi [%start, %entry], [%step, %loop]
///    ...
///    %step = select _, %tv, %fv
///
/// where %tv and %fv ultimately end up using %rec via the same %BO instruction,
/// after digging through the use-def chain.
///
/// ExtraConst is relevant if \p BOWithConstOpToMatch is supplied: when digging
/// the use-def chain, a BinOp with opcode \p BOWithConstOpToMatch is matched,
/// and ExtraConst is a constant operand of that BinOp. This peculiarity exists,
/// because in a CRC algorithm, the \p BOWithConstOpToMatch is an XOR, and the
/// ExtraConst ends up being the generating polynomial.
bool RecurrenceInfo::matchConditionalRecurrence(
    const PHINode *P, Instruction::BinaryOps BOWithConstOpToMatch) {
  Phi = P;
  if (Phi->getNumIncomingValues() != 2)
    return false;

  for (unsigned Idx = 0; Idx != 2; ++Idx) {
    Value *FoundStep = Phi->getIncomingValue(Idx);
    Value *FoundStart = Phi->getIncomingValue(!Idx);

    Instruction *TV, *FV;
    if (!match(FoundStep,
               m_Select(m_Cmp(), m_Instruction(TV), m_Instruction(FV))))
      continue;

    // For a conditional recurrence, both the true and false values of the
    // select must ultimately end up in the same recurrent BinOp.
    BinaryOperator *FoundBO = digRecurrence(TV, BOWithConstOpToMatch);
    BinaryOperator *AltBO = digRecurrence(FV, BOWithConstOpToMatch);
    if (!FoundBO || FoundBO != AltBO)
      return false;

    if (BOWithConstOpToMatch != Instruction::BinaryOpsEnd && !ExtraConst) {
      LLVM_DEBUG(dbgs() << "HashRecognize: Unable to match single BinaryOp "
                           "with constant in conditional recurrence\n");
      return false;
    }

    BO = FoundBO;
    Start = FoundStart;
    Step = FoundStep;
    return true;
  }
  return false;
}

/// Iterates over all the phis in \p LoopLatch, and attempts to extract a
/// Conditional Recurrence and an optional Simple Recurrence.
static std::optional<std::pair<RecurrenceInfo, RecurrenceInfo>>
getRecurrences(BasicBlock *LoopLatch, const PHINode *IndVar, const Loop &L) {
  auto Phis = LoopLatch->phis();
  unsigned NumPhis = std::distance(Phis.begin(), Phis.end());
  if (NumPhis != 2 && NumPhis != 3)
    return {};

  RecurrenceInfo SimpleRecurrence(L);
  RecurrenceInfo ConditionalRecurrence(L);
  for (PHINode &P : Phis) {
    if (&P == IndVar)
      continue;
    if (!SimpleRecurrence)
      SimpleRecurrence.matchSimpleRecurrence(&P);
    if (!ConditionalRecurrence)
      ConditionalRecurrence.matchConditionalRecurrence(
          &P, Instruction::BinaryOps::Xor);
  }
  if (NumPhis == 3 && (!SimpleRecurrence || !ConditionalRecurrence))
    return {};
  return std::make_pair(SimpleRecurrence, ConditionalRecurrence);
}

PolynomialInfo::PolynomialInfo(unsigned TripCount, Value *LHS, const APInt &RHS,
                               Value *ComputedValue, bool ByteOrderSwapped,
                               Value *LHSAux)
    : TripCount(TripCount), LHS(LHS), RHS(RHS), ComputedValue(ComputedValue),
      ByteOrderSwapped(ByteOrderSwapped), LHSAux(LHSAux) {}

/// In the big-endian case, checks the bottom N bits against CheckFn, and that
/// the rest are unknown. In the little-endian case, checks the top N bits
/// against CheckFn, and that the rest are unknown. Callers usually call this
/// function with N = TripCount, and CheckFn checking that the remainder bits of
/// the CRC polynomial division are zero.
static bool checkExtractBits(const KnownBits &Known, unsigned N,
                             function_ref<bool(const KnownBits &)> CheckFn,
                             bool ByteOrderSwapped) {
  // Check that the entire thing is a constant.
  if (N == Known.getBitWidth())
    return CheckFn(Known.extractBits(N, 0));

  // Check that the {top, bottom} N bits are not unknown and that the {bottom,
  // top} N bits are known.
  unsigned BitPos = ByteOrderSwapped ? 0 : Known.getBitWidth() - N;
  unsigned SwappedBitPos = ByteOrderSwapped ? N : 0;
  return CheckFn(Known.extractBits(N, BitPos)) &&
         Known.extractBits(Known.getBitWidth() - N, SwappedBitPos).isUnknown();
}

/// Generate a lookup table of 256 entries by interleaving the generating
/// polynomial. The optimization technique of table-lookup for CRC is also
/// called the Sarwate algorithm.
CRCTable HashRecognize::genSarwateTable(const APInt &GenPoly,
                                        bool ByteOrderSwapped) {
  unsigned BW = GenPoly.getBitWidth();
  CRCTable Table;
  Table[0] = APInt::getZero(BW);

  if (ByteOrderSwapped) {
    APInt CRCInit = APInt::getSignedMinValue(BW);
    for (unsigned I = 1; I < 256; I <<= 1) {
      CRCInit = CRCInit.shl(1) ^
                (CRCInit.isSignBitSet() ? GenPoly : APInt::getZero(BW));
      for (unsigned J = 0; J < I; ++J)
        Table[I + J] = CRCInit ^ Table[J];
    }
    return Table;
  }

  APInt CRCInit(BW, 1);
  for (unsigned I = 128; I; I >>= 1) {
    CRCInit = CRCInit.lshr(1) ^ (CRCInit[0] ? GenPoly : APInt::getZero(BW));
    for (unsigned J = 0; J < 256; J += (I << 1))
      Table[I + J] = CRCInit ^ Table[J];
  }
  return Table;
}

/// Checks that \p P1 and \p P2 are used together in an XOR in the use-def chain
/// of \p SI's condition, ignoring any casts. The purpose of this function is to
/// ensure that LHSAux from the SimpleRecurrence is used correctly in the CRC
/// computation. We cannot check the correctness of casts at this point, and
/// rely on the KnownBits propagation to check correctness of the CRC
/// computation.
///
/// In other words, it checks for the following pattern:
///
/// loop:
///   %P1 = phi [_, %entry], [%P1.next, %loop]
///   %P2 = phi [_, %entry], [%P2.next, %loop]
///   ...
///   %xor = xor (CastOrSelf %P1), (CastOrSelf %P2)
///
/// where %xor is in the use-def chain of \p SI's condition.
static bool isConditionalOnXorOfPHIs(const SelectInst *SI, const PHINode *P1,
                                     const PHINode *P2, const Loop &L) {
  SmallVector<const Instruction *> Worklist;

  // matchConditionalRecurrence has already ensured that the SelectInst's
  // condition is an Instruction.
  Worklist.push_back(cast<Instruction>(SI->getCondition()));

  while (!Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();

    // Don't add a PHI's operands to the Worklist.
    if (isa<PHINode>(I))
      continue;

    // If we match an XOR of the two PHIs ignoring casts, we're done.
    if (match(I, m_c_Xor(m_CastOrSelf(m_Specific(P1)),
                         m_CastOrSelf(m_Specific(P2)))))
      return true;

    // Continue along the use-def chain.
    for (const Use &U : I->operands())
      if (auto *UI = dyn_cast<Instruction>(U))
        if (L.contains(UI))
          Worklist.push_back(UI);
  }
  return false;
}

// Recognizes a multiplication or division by the constant two, using SCEV. By
// doing this, we're immune to whether the IR expression is mul/udiv or
// equivalently shl/lshr. Return false when it is a UDiv, true when it is a Mul,
// and std::nullopt otherwise.
static std::optional<bool> isBigEndianBitShift(Value *V, ScalarEvolution &SE) {
  if (!V->getType()->isIntegerTy())
    return {};

  const SCEV *E = SE.getSCEV(V);
  if (match(E, m_scev_UDiv(m_SCEV(), m_scev_SpecificInt(2))))
    return false;
  if (match(E, m_scev_Mul(m_scev_SpecificInt(2), m_SCEV())))
    return true;
  return {};
}

/// The main entry point for analyzing a loop and recognizing the CRC algorithm.
/// Returns a PolynomialInfo on success, and either an ErrBits or a StringRef on
/// failure.
std::variant<PolynomialInfo, ErrBits, StringRef>
HashRecognize::recognizeCRC() const {
  if (!L.isInnermost())
    return "Loop is not innermost";
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *Exit = L.getExitBlock();
  const PHINode *IndVar = L.getCanonicalInductionVariable();
  if (!Latch || !Exit || !IndVar || L.getNumBlocks() != 1)
    return "Loop not in canonical form";
  unsigned TC = SE.getSmallConstantTripCount(&L);
  if (!TC || TC > 256 || TC % 8)
    return "Unable to find a small constant byte-multiple trip count";

  auto R = getRecurrences(Latch, IndVar, L);
  if (!R)
    return "Found stray PHI";
  auto [SimpleRecurrence, ConditionalRecurrence] = *R;
  if (!ConditionalRecurrence)
    return "Unable to find conditional recurrence";

  // Make sure that all recurrences are either all SCEVMul with two or SCEVDiv
  // with two, or in other words, that they're single bit-shifts.
  std::optional<bool> ByteOrderSwapped =
      isBigEndianBitShift(ConditionalRecurrence.BO, SE);
  if (!ByteOrderSwapped)
    return "Loop with non-unit bitshifts";
  if (SimpleRecurrence) {
    if (isBigEndianBitShift(SimpleRecurrence.BO, SE) != ByteOrderSwapped)
      return "Loop with non-unit bitshifts";

    // Ensure that the PHIs have exactly two uses:
    // the bit-shift, and the XOR (or a cast feeding into the XOR).
    if (!ConditionalRecurrence.Phi->hasNUses(2) ||
        !SimpleRecurrence.Phi->hasNUses(2))
      return "Recurrences have stray uses";

    // Check that the SelectInst ConditionalRecurrence.Step is conditional on
    // the XOR of SimpleRecurrence.Phi and ConditionalRecurrence.Phi.
    if (!isConditionalOnXorOfPHIs(cast<SelectInst>(ConditionalRecurrence.Step),
                                  SimpleRecurrence.Phi,
                                  ConditionalRecurrence.Phi, L))
      return "Recurrences not intertwined with XOR";
  }

  // Make sure that the TC doesn't exceed the bitwidth of LHSAux, or LHS.
  Value *LHS = ConditionalRecurrence.Start;
  Value *LHSAux = SimpleRecurrence ? SimpleRecurrence.Start : nullptr;
  if (TC > (LHSAux ? LHSAux->getType()->getIntegerBitWidth()
                   : LHS->getType()->getIntegerBitWidth()))
    return "Loop iterations exceed bitwidth of data";

  // Make sure that the computed value is used in the exit block: this should be
  // true even if it is only really used in an outer loop's exit block, since
  // the loop is in LCSSA form.
  auto *ComputedValue = cast<SelectInst>(ConditionalRecurrence.Step);
  if (none_of(ComputedValue->users(), [Exit](User *U) {
        auto *UI = dyn_cast<Instruction>(U);
        return UI && UI->getParent() == Exit;
      }))
    return "Unable to find use of computed value in loop exit block";

  assert(ConditionalRecurrence.ExtraConst &&
         "Expected ExtraConst in conditional recurrence");
  const APInt &GenPoly = *ConditionalRecurrence.ExtraConst;

  // PhiEvolutions are pairs of PHINodes along with their incoming value from
  // within the loop, which we term as their step. Note that in the case of a
  // Simple Recurrence, Step is an operand of the BO, while in a Conditional
  // Recurrence, it is a SelectInst.
  SmallVector<PhiStepPair, 2> PhiEvolutions;
  PhiEvolutions.emplace_back(ConditionalRecurrence.Phi, ComputedValue);
  if (SimpleRecurrence)
    PhiEvolutions.emplace_back(SimpleRecurrence.Phi, SimpleRecurrence.BO);

  ValueEvolution VE(TC, *ByteOrderSwapped);
  if (!VE.computeEvolutions(PhiEvolutions))
    return VE.getError();
  KnownBits ResultBits = VE.KnownPhis.at(ConditionalRecurrence.Phi);

  unsigned N = std::min(TC, ResultBits.getBitWidth());
  auto IsZero = [](const KnownBits &K) { return K.isZero(); };
  if (!checkExtractBits(ResultBits, N, IsZero, *ByteOrderSwapped))
    return ErrBits(ResultBits, TC, *ByteOrderSwapped);

  return PolynomialInfo(TC, LHS, GenPoly, ComputedValue, *ByteOrderSwapped,
                        LHSAux);
}

void CRCTable::print(raw_ostream &OS) const {
  for (unsigned I = 0; I < 256; I++) {
    (*this)[I].print(OS, false);
    OS << (I % 16 == 15 ? '\n' : ' ');
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void CRCTable::dump() const { print(dbgs()); }
#endif

void HashRecognize::print(raw_ostream &OS) const {
  if (!L.isInnermost())
    return;
  OS << "HashRecognize: Checking a loop in '"
     << L.getHeader()->getParent()->getName() << "' from " << L.getLocStr()
     << "\n";
  auto Ret = recognizeCRC();
  if (!std::holds_alternative<PolynomialInfo>(Ret)) {
    OS << "Did not find a hash algorithm\n";
    if (std::holds_alternative<StringRef>(Ret))
      OS << "Reason: " << std::get<StringRef>(Ret) << "\n";
    if (std::holds_alternative<ErrBits>(Ret)) {
      auto [Actual, Iter, ByteOrderSwapped] = std::get<ErrBits>(Ret);
      OS << "Reason: Expected " << (ByteOrderSwapped ? "bottom " : "top ")
         << Iter << " bits zero (";
      Actual.print(OS);
      OS << ")\n";
    }
    return;
  }

  auto Info = std::get<PolynomialInfo>(Ret);
  OS << "Found" << (Info.ByteOrderSwapped ? " big-endian " : " little-endian ")
     << "CRC-" << Info.RHS.getBitWidth() << " loop with trip count "
     << Info.TripCount << "\n";
  OS.indent(2) << "Initial CRC: ";
  Info.LHS->print(OS);
  OS << "\n";
  OS.indent(2) << "Generating polynomial: ";
  Info.RHS.print(OS, false);
  OS << "\n";
  OS.indent(2) << "Computed CRC: ";
  Info.ComputedValue->print(OS);
  OS << "\n";
  if (Info.LHSAux) {
    OS.indent(2) << "Auxiliary data: ";
    Info.LHSAux->print(OS);
    OS << "\n";
  }
  OS.indent(2) << "Computed CRC lookup table:\n";
  genSarwateTable(Info.RHS, Info.ByteOrderSwapped).print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void HashRecognize::dump() const { print(dbgs()); }
#endif

std::optional<PolynomialInfo> HashRecognize::getResult() const {
  auto Res = HashRecognize(L, SE).recognizeCRC();
  if (std::holds_alternative<PolynomialInfo>(Res))
    return std::get<PolynomialInfo>(Res);
  return std::nullopt;
}

HashRecognize::HashRecognize(const Loop &L, ScalarEvolution &SE)
    : L(L), SE(SE) {}

PreservedAnalyses HashRecognizePrinterPass::run(Loop &L,
                                                LoopAnalysisManager &AM,
                                                LoopStandardAnalysisResults &AR,
                                                LPMUpdater &) {
  HashRecognize(L, AR.SE).print(OS);
  return PreservedAnalyses::all();
}
