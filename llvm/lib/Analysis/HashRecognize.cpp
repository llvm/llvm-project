//===- HashRecognize.h ------------------------------------------*- C++ -*-===//
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
// implmentations are performed bit-wise, and can be optimized to be performed
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
#include "llvm/ADT/SmallSet.h"
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
  unsigned TripCount;
  bool ByteOrderSwapped;
  APInt GenPoly;
  StringRef ErrStr;
  unsigned AtIteration;

  KnownBits computeBinOp(const BinaryOperator *I, const KnownPhiMap &KnownPhis);
  KnownBits computeInstr(const Instruction *I, const KnownPhiMap &KnownPhis);
  KnownBits compute(const Value *V, const KnownPhiMap &KnownPhis);

public:
  ValueEvolution(unsigned TripCount, bool ByteOrderSwapped);

  // In case ValueEvolution encounters an error, these are meant to be used for
  // a precise error message.
  bool hasError() const;
  StringRef getError() const;

  // Given a list of PHI nodes along with their incoming value from within the
  // loop, and the trip-count of the loop, computeEvolutions
  // computes the KnownBits of each of the PHI nodes on the final iteration.
  std::optional<KnownPhiMap>
  computeEvolutions(ArrayRef<PhiStepPair> PhiEvolutions);
};

ValueEvolution::ValueEvolution(unsigned TripCount, bool ByteOrderSwapped)
    : TripCount(TripCount), ByteOrderSwapped(ByteOrderSwapped) {}

bool ValueEvolution::hasError() const { return !ErrStr.empty(); }
StringRef ValueEvolution::getError() const { return ErrStr; }

/// Compute the KnownBits of BinaryOperator \p I.
KnownBits ValueEvolution::computeBinOp(const BinaryOperator *I,
                                       const KnownPhiMap &KnownPhis) {
  unsigned BitWidth = I->getType()->getScalarSizeInBits();

  KnownBits KnownL(compute(I->getOperand(0), KnownPhis));
  KnownBits KnownR(compute(I->getOperand(1), KnownPhis));

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
    return {BitWidth};
  }
}

/// Compute the KnownBits of Instruction \p I.
KnownBits ValueEvolution::computeInstr(const Instruction *I,
                                       const KnownPhiMap &KnownPhis) {
  unsigned BitWidth = I->getType()->getScalarSizeInBits();

  // We look up in the map that contains the KnownBits of the PHI from the
  // previous iteration.
  if (const PHINode *P = dyn_cast<PHINode>(I))
    return KnownPhis.lookup_or(P, BitWidth);

  // Compute the KnownBits for a Select(Cmp()), forcing it to take the take the
  // branch that is predicated on the (least|most)-significant-bit check.
  CmpPredicate Pred;
  Value *L, *R, *TV, *FV;
  if (match(I, m_Select(m_ICmp(Pred, m_Value(L), m_Value(R)), m_Value(TV),
                        m_Value(FV)))) {
    KnownBits KnownL = compute(L, KnownPhis).zextOrTrunc(BitWidth);
    KnownBits KnownR = compute(R, KnownPhis).zextOrTrunc(BitWidth);
    KnownBits KnownTV = compute(TV, KnownPhis);
    KnownBits KnownFV = compute(FV, KnownPhis);
    auto LCR = ConstantRange::fromKnownBits(KnownL, false);
    auto RCR = ConstantRange::fromKnownBits(KnownR, false);

    // We need to check LCR against [0, 2) in the little-endian case, because
    // the RCR check is insufficient: it is simply [0, 1).
    auto CheckLCR = ConstantRange(APInt::getZero(BitWidth), APInt(BitWidth, 2));
    if (!ByteOrderSwapped && LCR != CheckLCR) {
      ErrStr = "Bad LHS of significant-bit-check";
      return {BitWidth};
    }

    // Check that the predication is on (most|least) significant bit.
    auto AllowedR = ConstantRange::makeAllowedICmpRegion(Pred, RCR);
    auto InverseR = ConstantRange::makeAllowedICmpRegion(
        CmpInst::getInversePredicate(Pred), RCR);
    ConstantRange LSBRange(APInt::getZero(BitWidth), APInt(BitWidth, 1));
    ConstantRange MSBRange(APInt::getZero(BitWidth),
                           APInt::getSignedMinValue(BitWidth));
    const ConstantRange &CheckRCR = ByteOrderSwapped ? MSBRange : LSBRange;
    if (AllowedR == CheckRCR)
      return KnownTV;
    if (AllowedR.inverse() == CheckRCR)
      return KnownFV;

    ErrStr = "Bad RHS of significant-bit-check";
    return {BitWidth};
  }

  if (auto *BO = dyn_cast<BinaryOperator>(I))
    return computeBinOp(BO, KnownPhis);

  switch (I->getOpcode()) {
  case Instruction::CastOps::Trunc:
    return compute(I->getOperand(0), KnownPhis).trunc(BitWidth);
  case Instruction::CastOps::ZExt:
    return compute(I->getOperand(0), KnownPhis).zext(BitWidth);
  case Instruction::CastOps::SExt:
    return compute(I->getOperand(0), KnownPhis).sext(BitWidth);
  default:
    ErrStr = "Unknown Instruction";
    return {BitWidth};
  }
}

/// Compute the KnownBits of Value \p V.
KnownBits ValueEvolution::compute(const Value *V,
                                  const KnownPhiMap &KnownPhis) {
  unsigned BitWidth = V->getType()->getScalarSizeInBits();

  const APInt *C;
  if (match(V, m_APInt(C)))
    return KnownBits::makeConstant(*C);

  if (auto *I = dyn_cast<Instruction>(V))
    return computeInstr(I, KnownPhis);

  ErrStr = "Unknown Value";
  return {BitWidth};
}

// Takes every PHI-step pair in PhiEvolutions, and computes KnownBits on the
// final iteration, using KnownBits from the previous iteration.
std::optional<KnownPhiMap>
ValueEvolution::computeEvolutions(ArrayRef<PhiStepPair> PhiEvolutions) {
  KnownPhiMap KnownPhis;
  for (unsigned I = 0; I < TripCount; ++I) {
    AtIteration = I;
    for (auto [Phi, Step] : PhiEvolutions) {
      KnownBits KnownAtIter = computeInstr(Step, KnownPhis);
      if (KnownAtIter.getBitWidth() < I + 1) {
        ErrStr = "Loop iterations exceed bitwidth of result";
        return std::nullopt;
      }
      KnownPhis.emplace_or_assign(Phi, KnownAtIter);
    }
  }
  return KnownPhis;
}

/// Digs for a recurrence starting with \p V hitting the PHI node \p P in a
/// use-def chain. Used by matchConditionalRecurrence.
static BinaryOperator *
digRecurrence(Instruction *V, const PHINode *P, const Loop &L,
              const APInt *&ExtraConst,
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
    if (match(I, m_c_BinOp(m_Value(), m_Specific(P))))
      return cast<BinaryOperator>(I);

    // Bind to ExtraConst, if we match exactly one.
    if (I->getOpcode() == BOWithConstOpToMatch) {
      if (ExtraConst)
        return nullptr;
      match(I, m_c_BinOp(m_APInt(ExtraConst), m_Value()));
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
///    %rec = [%start, %entry], [%step, %loop]
///    ...
///    %step = select _, %tv, %fv
///
/// where %tv and %fv ultimately end up using %rec via the same %BO instruction,
/// after digging through the use-def chain.
///
/// \p ExtraConst is relevant if \p BOWithConstOpToMatch is supplied: when
/// digging the use-def chain, a BinOp with opcode \p BOWithConstOpToMatch is
/// matched, and \p ExtraConst is a constant operand of that BinOp. This
/// peculiary exists, because in a CRC algorithm, the \p BOWithConstOpToMatch is
/// an XOR, and the \p ExtraConst ends up being the generating polynomial.
static bool matchConditionalRecurrence(
    const PHINode *P, BinaryOperator *&BO, Value *&Start, Value *&Step,
    const Loop &L, const APInt *&ExtraConst,
    Instruction::BinaryOps BOWithConstOpToMatch = Instruction::BinaryOpsEnd) {
  if (P->getNumIncomingValues() != 2)
    return false;

  for (unsigned Idx = 0; Idx != 2; ++Idx) {
    Value *FoundStep = P->getIncomingValue(Idx);
    Value *FoundStart = P->getIncomingValue(!Idx);

    Instruction *TV, *FV;
    if (!match(FoundStep,
               m_Select(m_Cmp(), m_Instruction(TV), m_Instruction(FV))))
      continue;

    // For a conditional recurrence, both the true and false values of the
    // select must ultimately end up in the same recurrent BinOp.
    ExtraConst = nullptr;
    BinaryOperator *FoundBO =
        digRecurrence(TV, P, L, ExtraConst, BOWithConstOpToMatch);
    BinaryOperator *AltBO =
        digRecurrence(FV, P, L, ExtraConst, BOWithConstOpToMatch);

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

/// A structure that can hold either a Simple Recurrence or a Conditional
/// Recurrence. Note that in the case of a Simple Recurrence, Step is an operand
/// of the BO, while in a Conditional Recurrence, it is a SelectInst.
struct RecurrenceInfo {
  PHINode *Phi;
  BinaryOperator *BO;
  Value *Start;
  Value *Step;
  std::optional<APInt> ExtraConst;

  RecurrenceInfo(PHINode *Phi, BinaryOperator *BO, Value *Start, Value *Step,
                 std::optional<APInt> ExtraConst = std::nullopt)
      : Phi(Phi), BO(BO), Start(Start), Step(Step), ExtraConst(ExtraConst) {}

  void print(raw_ostream &OS, unsigned Indent) const {
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
};

/// Iterates over all the phis in \p LoopLatch, and attempts to extract a Simple
/// Recurrence, and a Conditional Recurrence.
static std::pair<std::optional<RecurrenceInfo>, std::optional<RecurrenceInfo>>
getRecurrences(BasicBlock *LoopLatch, const PHINode *IndVar, const Loop &L) {
  std::optional<RecurrenceInfo> SimpleRecurrence, ConditionalRecurrence;
  for (PHINode &P : LoopLatch->phis()) {
    if (&P == IndVar)
      continue;
    if (!P.getType()->isIntegerTy()) {
      LLVM_DEBUG(dbgs() << "HashRecognize: Non-integral PHI found\n");
      return {};
    }

    BinaryOperator *BO;
    Value *Start, *Step;
    const APInt *GenPoly;
    if (!SimpleRecurrence && matchSimpleRecurrence(&P, BO, Start, Step)) {
      SimpleRecurrence = {&P, BO, Start, Step};
    } else if (!ConditionalRecurrence &&
               matchConditionalRecurrence(&P, BO, Start, Step, L, GenPoly,
                                          Instruction::BinaryOps::Xor)) {
      ConditionalRecurrence = {&P, BO, Start, Step, *GenPoly};
    } else {
      LLVM_DEBUG(dbgs() << "HashRecognize: Stray PHI found: " << P << "\n");
      return {};
    }
  }
  return {SimpleRecurrence, ConditionalRecurrence};
}

PolynomialInfo::PolynomialInfo(unsigned TripCount, const Value *LHS,
                               const APInt &RHS, const Value *ComputedValue,
                               bool ByteOrderSwapped, const Value *LHSAux)
    : TripCount(TripCount), LHS(LHS), RHS(RHS), ComputedValue(ComputedValue),
      ByteOrderSwapped(ByteOrderSwapped), LHSAux(LHSAux) {}

/// In big-endian case, checks that bottom N bits against CheckFn, and that the
/// rest are unknown. In little-endian case, checks that the top N bits against
/// CheckFn, and that the rest are unknown.
static bool checkExtractBits(const KnownBits &Known, unsigned N,
                             function_ref<bool(const KnownBits &)> CheckFn,
                             bool ByteOrderSwapped) {
  unsigned BitPos = ByteOrderSwapped ? 0 : Known.getBitWidth() - N;
  unsigned SwappedBitPos = ByteOrderSwapped ? N : 0;

  // Check that the entire thing is a constant.
  if (N == Known.getBitWidth())
    return CheckFn(Known.extractBits(N, 0));

  // Check that the {top, bottom} N bits are not unknown and that the {bottom,
  // top} N bits are known.
  return CheckFn(Known.extractBits(N, BitPos)) &&
         Known.extractBits(Known.getBitWidth() - N, SwappedBitPos).isUnknown();
}

/// Generate a lookup table of 256 entries by interleaving the generating
/// polynomial. The optimization technique of table-lookup for CRC is also
/// called the Sarwate algorithm.
CRCTable HashRecognize::genSarwateTable(const APInt &GenPoly,
                                        bool ByteOrderSwapped) const {
  unsigned BW = GenPoly.getBitWidth();
  unsigned MSB = 1 << (BW - 1);
  CRCTable Table;
  Table[0] = APInt::getZero(BW);

  if (ByteOrderSwapped) {
    APInt CRCInit(BW, 1);
    for (unsigned I = 1; I < 256; I <<= 1) {
      CRCInit = CRCInit.shl(1) ^
                ((CRCInit & MSB).isZero() ? APInt::getZero(BW) : GenPoly);
      for (unsigned J = 0; J < I; ++J)
        Table[I + J] = CRCInit ^ Table[J];
    }
    return Table;
  }

  APInt CRCInit(BW, 128);
  for (unsigned I = 128; I; I >>= 1) {
    CRCInit = CRCInit.lshr(1) ^
              ((CRCInit & 1).isZero() ? APInt::getZero(BW) : GenPoly);
    for (unsigned J = 0; J < 256; J += (I << 1))
      Table[I + J] = CRCInit ^ Table[J];
  }
  return Table;
}

/// Checks if \p Reference is reachable from \p Needle on the use-def chain, and
/// that there are no stray PHI nodes while digging the use-def chain. \p
/// BOToMatch is a CRC peculiarity: at least one of the Users of Needle needs to
/// match this OpCode, which is XOR for CRC.
static bool arePHIsIntertwined(
    const PHINode *Needle, const PHINode *Reference, const Loop &L,
    Instruction::BinaryOps BOToMatch = Instruction::BinaryOpsEnd) {
  // Initialize the worklist with Users of the Needle.
  SmallVector<const Instruction *> Worklist;
  for (const User *U : Needle->users()) {
    if (auto *UI = dyn_cast<Instruction>(U))
      if (L.contains(UI))
        Worklist.push_back(UI);
  }

  // BOToMatch is usually XOR for CRC.
  if (BOToMatch != Instruction::BinaryOpsEnd) {
    if (count_if(Worklist, [BOToMatch](const Instruction *I) {
          return I->getOpcode() == BOToMatch;
        }) != 1)
      return false;
  }

  while (!Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();

    // Since Needle is never pushed onto the Worklist, I must either be the
    // Reference PHI node (in which case we're done), or a stray PHI node (in
    // which case we abort).
    if (isa<PHINode>(I))
      return I == Reference;

    for (const Use &U : I->operands())
      if (auto *UI = dyn_cast<Instruction>(U))
        // Don't push Needle back onto the Worklist.
        if (UI != Needle && L.contains(UI))
          Worklist.push_back(UI);
  }
  return false;
}

// Recognizes a multiplication or division by the constant two, using SCEV. By
// doing this, we're immune to whether the IR expression is mul/udiv or
// equivalently shl/lshr. Return false when it is a UDiv, true when it is a Mul,
// and std::nullopt otherwise.
static std::optional<bool> isBigEndianBitShift(const SCEV *E) {
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
  unsigned TC = SE.getSmallConstantMaxTripCount(&L);
  if (!TC)
    return "Unable to find a small constant trip count";
  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *Exit = L.getExitBlock();
  const PHINode *IndVar = L.getCanonicalInductionVariable();
  if (!Exit || !Latch || !IndVar)
    return "Loop not in canonical form";

  auto [SimpleRecurrence, ConditionalRecurrence] =
      getRecurrences(Latch, IndVar, L);

  if (!ConditionalRecurrence)
    return "Unable to find conditional recurrence";

  // Make sure that all recurrences are either all SCEVMul with two or SCEVDiv
  // with two, or in other words, that they're single bit-shifts.
  SmallSet<std::optional<bool>, 2> EndianStatus;
  for (auto Info : {SimpleRecurrence, ConditionalRecurrence})
    if (Info)
      EndianStatus.insert(isBigEndianBitShift(SE.getSCEV(Info->BO)));

  if (EndianStatus.size() != 1 || !*EndianStatus.begin())
    return "Loop with non-unit bitshifts";

  bool ByteOrderSwapped = **EndianStatus.begin();

  if (SimpleRecurrence &&
      !arePHIsIntertwined(SimpleRecurrence->Phi, ConditionalRecurrence->Phi, L,
                          Instruction::BinaryOps::Xor))
    return "Simple recurrence doesn't use conditional recurrence with XOR";

  // Make sure that the computed value is used in the exit block: this should be
  // true even if it is only really used in an outer loop's exit block, since
  // the loop is in LCSSA form.
  auto *ComputedValue = cast<SelectInst>(ConditionalRecurrence->Step);
  if (none_of(ComputedValue->users(), [Exit](User *U) {
        auto *UI = dyn_cast<Instruction>(U);
        return UI && UI->getParent() == Exit;
      }))
    return "Unable to find use of computed value in loop exit block";

  assert(ConditionalRecurrence->ExtraConst &&
         "Expected ExtraConst in conditional recurrence");
  const APInt &GenPoly = *ConditionalRecurrence->ExtraConst;

  // PhiEvolutions are pairs of PHINodes along with their incoming value from
  // within the loop, which we term as their step.
  SmallVector<PhiStepPair, 2> PhiEvolutions;
  PhiEvolutions.emplace_back(ConditionalRecurrence->Phi, ComputedValue);
  if (SimpleRecurrence)
    PhiEvolutions.emplace_back(SimpleRecurrence->Phi, SimpleRecurrence->BO);

  const Value *LHSAux = SimpleRecurrence ? SimpleRecurrence->Start : nullptr;

  ValueEvolution VE(TC, ByteOrderSwapped);
  std::optional<KnownPhiMap> KnownPhis = VE.computeEvolutions(PhiEvolutions);

  if (VE.hasError())
    return VE.getError();

  KnownBits ResultBits = KnownPhis->at(ConditionalRecurrence->Phi);
  auto IsZero = [](const KnownBits &K) { return K.isZero(); };
  if (!checkExtractBits(ResultBits, TC, IsZero, ByteOrderSwapped))
    return ErrBits(ResultBits, TC, ByteOrderSwapped);

  return PolynomialInfo(TC, ConditionalRecurrence->Start, GenPoly,
                        ComputedValue, ByteOrderSwapped, LHSAux);
}

void CRCTable::print(raw_ostream &OS) const {
  for (unsigned I = 0; I < 256; I++) {
    (*this)[I].print(OS, false);
    OS << (I % 16 == 15 ? '\n' : ' ');
  }
}

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

HashRecognize::HashRecognize(const Loop &L, ScalarEvolution &SE)
    : L(L), SE(SE) {}

PreservedAnalyses HashRecognizePrinterPass::run(Loop &L,
                                                LoopAnalysisManager &AM,
                                                LoopStandardAnalysisResults &AR,
                                                LPMUpdater &) {
  AM.getResult<HashRecognizeAnalysis>(L, AR).print(OS);
  return PreservedAnalyses::all();
}

HashRecognize HashRecognizeAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                                         LoopStandardAnalysisResults &AR) {
  return {L, AR.SE};
}

AnalysisKey HashRecognizeAnalysis::Key;
