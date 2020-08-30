//===- EarlyCSE.cpp - Simple and fast CSE pass ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs a simple dominator tree walk that eliminates trivially
// redundant instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/GuardUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <deque>
#include <memory>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "early-cse"

STATISTIC(NumSimplify, "Number of instructions simplified or DCE'd");
STATISTIC(NumCSE,      "Number of instructions CSE'd");
STATISTIC(NumCSECVP,   "Number of compare instructions CVP'd");
STATISTIC(NumCSELoad,  "Number of load instructions CSE'd");
STATISTIC(NumCSECall,  "Number of call instructions CSE'd");
STATISTIC(NumDSE,      "Number of trivial dead stores removed");

DEBUG_COUNTER(CSECounter, "early-cse",
              "Controls which instructions are removed");

static cl::opt<unsigned> EarlyCSEMssaOptCap(
    "earlycse-mssa-optimization-cap", cl::init(500), cl::Hidden,
    cl::desc("Enable imprecision in EarlyCSE in pathological cases, in exchange "
             "for faster compile. Caps the MemorySSA clobbering calls."));

static cl::opt<bool> EarlyCSEDebugHash(
    "earlycse-debug-hash", cl::init(false), cl::Hidden,
    cl::desc("Perform extra assertion checking to verify that SimpleValue's hash "
             "function is well-behaved w.r.t. its isEqual predicate"));

//===----------------------------------------------------------------------===//
// SimpleValue
//===----------------------------------------------------------------------===//

namespace {

/// Struct representing the available values in the scoped hash table.
struct SimpleValue {
  Instruction *Inst;

  SimpleValue(Instruction *I) : Inst(I) {
    assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
  }

  bool isSentinel() const {
    return Inst == DenseMapInfo<Instruction *>::getEmptyKey() ||
           Inst == DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static bool canHandle(Instruction *Inst) {
    // This can only handle non-void readnone functions.
    if (CallInst *CI = dyn_cast<CallInst>(Inst))
      return CI->doesNotAccessMemory() && !CI->getType()->isVoidTy();
    return isa<CastInst>(Inst) || isa<UnaryOperator>(Inst) ||
           isa<BinaryOperator>(Inst) || isa<GetElementPtrInst>(Inst) ||
           isa<CmpInst>(Inst) || isa<SelectInst>(Inst) ||
           isa<ExtractElementInst>(Inst) || isa<InsertElementInst>(Inst) ||
           isa<ShuffleVectorInst>(Inst) || isa<ExtractValueInst>(Inst) ||
           isa<InsertValueInst>(Inst) || isa<FreezeInst>(Inst);
  }
};

} // end anonymous namespace

namespace llvm {

template <> struct DenseMapInfo<SimpleValue> {
  static inline SimpleValue getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }

  static inline SimpleValue getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static unsigned getHashValue(SimpleValue Val);
  static bool isEqual(SimpleValue LHS, SimpleValue RHS);
};

} // end namespace llvm

/// Match a 'select' including an optional 'not's of the condition.
static bool matchSelectWithOptionalNotCond(Value *V, Value *&Cond, Value *&A,
                                           Value *&B,
                                           SelectPatternFlavor &Flavor) {
  // Return false if V is not even a select.
  if (!match(V, m_Select(m_Value(Cond), m_Value(A), m_Value(B))))
    return false;

  // Look through a 'not' of the condition operand by swapping A/B.
  Value *CondNot;
  if (match(Cond, m_Not(m_Value(CondNot)))) {
    Cond = CondNot;
    std::swap(A, B);
  }

  // Match canonical forms of abs/nabs/min/max. We are not using ValueTracking's
  // more powerful matchSelectPattern() because it may rely on instruction flags
  // such as "nsw". That would be incompatible with the current hashing
  // mechanism that may remove flags to increase the likelihood of CSE.

  // These are the canonical forms of abs(X) and nabs(X) created by instcombine:
  // %N = sub i32 0, %X
  // %C = icmp slt i32 %X, 0
  // %ABS = select i1 %C, i32 %N, i32 %X
  //
  // %N = sub i32 0, %X
  // %C = icmp slt i32 %X, 0
  // %NABS = select i1 %C, i32 %X, i32 %N
  Flavor = SPF_UNKNOWN;
  CmpInst::Predicate Pred;
  if (match(Cond, m_ICmp(Pred, m_Specific(B), m_ZeroInt())) &&
      Pred == ICmpInst::ICMP_SLT && match(A, m_Neg(m_Specific(B)))) {
    // ABS: B < 0 ? -B : B
    Flavor = SPF_ABS;
    return true;
  }
  if (match(Cond, m_ICmp(Pred, m_Specific(A), m_ZeroInt())) &&
      Pred == ICmpInst::ICMP_SLT && match(B, m_Neg(m_Specific(A)))) {
    // NABS: A < 0 ? A : -A
    Flavor = SPF_NABS;
    return true;
  }

  if (!match(Cond, m_ICmp(Pred, m_Specific(A), m_Specific(B)))) {
    // Check for commuted variants of min/max by swapping predicate.
    // If we do not match the standard or commuted patterns, this is not a
    // recognized form of min/max, but it is still a select, so return true.
    if (!match(Cond, m_ICmp(Pred, m_Specific(B), m_Specific(A))))
      return true;
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  switch (Pred) {
  case CmpInst::ICMP_UGT: Flavor = SPF_UMAX; break;
  case CmpInst::ICMP_ULT: Flavor = SPF_UMIN; break;
  case CmpInst::ICMP_SGT: Flavor = SPF_SMAX; break;
  case CmpInst::ICMP_SLT: Flavor = SPF_SMIN; break;
  default: break;
  }

  return true;
}

static unsigned getHashValueImpl(SimpleValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash in all of the operands as pointers.
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst)) {
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);
    if (BinOp->isCommutative() && BinOp->getOperand(0) > BinOp->getOperand(1))
      std::swap(LHS, RHS);

    return hash_combine(BinOp->getOpcode(), LHS, RHS);
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(Inst)) {
    // Compares can be commuted by swapping the comparands and
    // updating the predicate.  Choose the form that has the
    // comparands in sorted order, or in the case of a tie, the
    // one with the lower predicate.
    Value *LHS = CI->getOperand(0);
    Value *RHS = CI->getOperand(1);
    CmpInst::Predicate Pred = CI->getPredicate();
    CmpInst::Predicate SwappedPred = CI->getSwappedPredicate();
    if (std::tie(LHS, Pred) > std::tie(RHS, SwappedPred)) {
      std::swap(LHS, RHS);
      Pred = SwappedPred;
    }
    return hash_combine(Inst->getOpcode(), Pred, LHS, RHS);
  }

  // Hash general selects to allow matching commuted true/false operands.
  SelectPatternFlavor SPF;
  Value *Cond, *A, *B;
  if (matchSelectWithOptionalNotCond(Inst, Cond, A, B, SPF)) {
    // Hash min/max/abs (cmp + select) to allow for commuted operands.
    // Min/max may also have non-canonical compare predicate (eg, the compare for
    // smin may use 'sgt' rather than 'slt'), and non-canonical operands in the
    // compare.
    // TODO: We should also detect FP min/max.
    if (SPF == SPF_SMIN || SPF == SPF_SMAX ||
        SPF == SPF_UMIN || SPF == SPF_UMAX) {
      if (A > B)
        std::swap(A, B);
      return hash_combine(Inst->getOpcode(), SPF, A, B);
    }
    if (SPF == SPF_ABS || SPF == SPF_NABS) {
      // ABS/NABS always puts the input in A and its negation in B.
      return hash_combine(Inst->getOpcode(), SPF, A, B);
    }

    // Hash general selects to allow matching commuted true/false operands.

    // If we do not have a compare as the condition, just hash in the condition.
    CmpInst::Predicate Pred;
    Value *X, *Y;
    if (!match(Cond, m_Cmp(Pred, m_Value(X), m_Value(Y))))
      return hash_combine(Inst->getOpcode(), Cond, A, B);

    // Similar to cmp normalization (above) - canonicalize the predicate value:
    // select (icmp Pred, X, Y), A, B --> select (icmp InvPred, X, Y), B, A
    if (CmpInst::getInversePredicate(Pred) < Pred) {
      Pred = CmpInst::getInversePredicate(Pred);
      std::swap(A, B);
    }
    return hash_combine(Inst->getOpcode(), Pred, X, Y, A, B);
  }

  if (CastInst *CI = dyn_cast<CastInst>(Inst))
    return hash_combine(CI->getOpcode(), CI->getType(), CI->getOperand(0));

  if (FreezeInst *FI = dyn_cast<FreezeInst>(Inst))
    return hash_combine(FI->getOpcode(), FI->getOperand(0));

  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(Inst))
    return hash_combine(EVI->getOpcode(), EVI->getOperand(0),
                        hash_combine_range(EVI->idx_begin(), EVI->idx_end()));

  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(Inst))
    return hash_combine(IVI->getOpcode(), IVI->getOperand(0),
                        IVI->getOperand(1),
                        hash_combine_range(IVI->idx_begin(), IVI->idx_end()));

  assert((isa<CallInst>(Inst) || isa<GetElementPtrInst>(Inst) ||
          isa<ExtractElementInst>(Inst) || isa<InsertElementInst>(Inst) ||
          isa<ShuffleVectorInst>(Inst) || isa<UnaryOperator>(Inst) ||
          isa<FreezeInst>(Inst)) &&
         "Invalid/unknown instruction");

  // Handle intrinsics with commutative operands.
  // TODO: Extend this to handle intrinsics with >2 operands where the 1st
  //       2 operands are commutative.
  auto *II = dyn_cast<IntrinsicInst>(Inst);
  if (II && II->isCommutative() && II->getNumArgOperands() == 2) {
    Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
    if (LHS > RHS)
      std::swap(LHS, RHS);
    return hash_combine(II->getOpcode(), LHS, RHS);
  }

  // Mix in the opcode.
  return hash_combine(
      Inst->getOpcode(),
      hash_combine_range(Inst->value_op_begin(), Inst->value_op_end()));
}

unsigned DenseMapInfo<SimpleValue>::getHashValue(SimpleValue Val) {
#ifndef NDEBUG
  // If -earlycse-debug-hash was specified, return a constant -- this
  // will force all hashing to collide, so we'll exhaustively search
  // the table for a match, and the assertion in isEqual will fire if
  // there's a bug causing equal keys to hash differently.
  if (EarlyCSEDebugHash)
    return 0;
#endif
  return getHashValueImpl(Val);
}

static bool isEqualImpl(SimpleValue LHS, SimpleValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;

  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;

  if (LHSI->getOpcode() != RHSI->getOpcode())
    return false;
  if (LHSI->isIdenticalToWhenDefined(RHSI))
    return true;

  // If we're not strictly identical, we still might be a commutable instruction
  if (BinaryOperator *LHSBinOp = dyn_cast<BinaryOperator>(LHSI)) {
    if (!LHSBinOp->isCommutative())
      return false;

    assert(isa<BinaryOperator>(RHSI) &&
           "same opcode, but different instruction type?");
    BinaryOperator *RHSBinOp = cast<BinaryOperator>(RHSI);

    // Commuted equality
    return LHSBinOp->getOperand(0) == RHSBinOp->getOperand(1) &&
           LHSBinOp->getOperand(1) == RHSBinOp->getOperand(0);
  }
  if (CmpInst *LHSCmp = dyn_cast<CmpInst>(LHSI)) {
    assert(isa<CmpInst>(RHSI) &&
           "same opcode, but different instruction type?");
    CmpInst *RHSCmp = cast<CmpInst>(RHSI);
    // Commuted equality
    return LHSCmp->getOperand(0) == RHSCmp->getOperand(1) &&
           LHSCmp->getOperand(1) == RHSCmp->getOperand(0) &&
           LHSCmp->getSwappedPredicate() == RHSCmp->getPredicate();
  }

  // TODO: Extend this for >2 args by matching the trailing N-2 args.
  auto *LII = dyn_cast<IntrinsicInst>(LHSI);
  auto *RII = dyn_cast<IntrinsicInst>(RHSI);
  if (LII && RII && LII->getIntrinsicID() == RII->getIntrinsicID() &&
      LII->isCommutative() && LII->getNumArgOperands() == 2) {
    return LII->getArgOperand(0) == RII->getArgOperand(1) &&
           LII->getArgOperand(1) == RII->getArgOperand(0);
  }

  // Min/max/abs can occur with commuted operands, non-canonical predicates,
  // and/or non-canonical operands.
  // Selects can be non-trivially equivalent via inverted conditions and swaps.
  SelectPatternFlavor LSPF, RSPF;
  Value *CondL, *CondR, *LHSA, *RHSA, *LHSB, *RHSB;
  if (matchSelectWithOptionalNotCond(LHSI, CondL, LHSA, LHSB, LSPF) &&
      matchSelectWithOptionalNotCond(RHSI, CondR, RHSA, RHSB, RSPF)) {
    if (LSPF == RSPF) {
      // TODO: We should also detect FP min/max.
      if (LSPF == SPF_SMIN || LSPF == SPF_SMAX ||
          LSPF == SPF_UMIN || LSPF == SPF_UMAX)
        return ((LHSA == RHSA && LHSB == RHSB) ||
                (LHSA == RHSB && LHSB == RHSA));

      if (LSPF == SPF_ABS || LSPF == SPF_NABS) {
        // Abs results are placed in a defined order by matchSelectPattern.
        return LHSA == RHSA && LHSB == RHSB;
      }

      // select Cond, A, B <--> select not(Cond), B, A
      if (CondL == CondR && LHSA == RHSA && LHSB == RHSB)
        return true;
    }

    // If the true/false operands are swapped and the conditions are compares
    // with inverted predicates, the selects are equal:
    // select (icmp Pred, X, Y), A, B <--> select (icmp InvPred, X, Y), B, A
    //
    // This also handles patterns with a double-negation in the sense of not +
    // inverse, because we looked through a 'not' in the matching function and
    // swapped A/B:
    // select (cmp Pred, X, Y), A, B <--> select (not (cmp InvPred, X, Y)), B, A
    //
    // This intentionally does NOT handle patterns with a double-negation in
    // the sense of not + not, because doing so could result in values
    // comparing
    // as equal that hash differently in the min/max/abs cases like:
    // select (cmp slt, X, Y), X, Y <--> select (not (not (cmp slt, X, Y))), X, Y
    //   ^ hashes as min                  ^ would not hash as min
    // In the context of the EarlyCSE pass, however, such cases never reach
    // this code, as we simplify the double-negation before hashing the second
    // select (and so still succeed at CSEing them).
    if (LHSA == RHSB && LHSB == RHSA) {
      CmpInst::Predicate PredL, PredR;
      Value *X, *Y;
      if (match(CondL, m_Cmp(PredL, m_Value(X), m_Value(Y))) &&
          match(CondR, m_Cmp(PredR, m_Specific(X), m_Specific(Y))) &&
          CmpInst::getInversePredicate(PredL) == PredR)
        return true;
    }
  }

  return false;
}

bool DenseMapInfo<SimpleValue>::isEqual(SimpleValue LHS, SimpleValue RHS) {
  // These comparisons are nontrivial, so assert that equality implies
  // hash equality (DenseMap demands this as an invariant).
  bool Result = isEqualImpl(LHS, RHS);
  assert(!Result || (LHS.isSentinel() && LHS.Inst == RHS.Inst) ||
         getHashValueImpl(LHS) == getHashValueImpl(RHS));
  return Result;
}

//===----------------------------------------------------------------------===//
// CallValue
//===----------------------------------------------------------------------===//

namespace {

/// Struct representing the available call values in the scoped hash
/// table.
struct CallValue {
  Instruction *Inst;

  CallValue(Instruction *I) : Inst(I) {
    assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
  }

  bool isSentinel() const {
    return Inst == DenseMapInfo<Instruction *>::getEmptyKey() ||
           Inst == DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static bool canHandle(Instruction *Inst) {
    // Don't value number anything that returns void.
    if (Inst->getType()->isVoidTy())
      return false;

    CallInst *CI = dyn_cast<CallInst>(Inst);
    if (!CI || !CI->onlyReadsMemory())
      return false;
    return true;
  }
};

} // end anonymous namespace

namespace llvm {

template <> struct DenseMapInfo<CallValue> {
  static inline CallValue getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }

  static inline CallValue getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static unsigned getHashValue(CallValue Val);
  static bool isEqual(CallValue LHS, CallValue RHS);
};

} // end namespace llvm

unsigned DenseMapInfo<CallValue>::getHashValue(CallValue Val) {
  Instruction *Inst = Val.Inst;

  // gc.relocate is 'special' call: its second and third operands are
  // not real values, but indices into statepoint's argument list.
  // Get values they point to.
  if (const GCRelocateInst *GCR = dyn_cast<GCRelocateInst>(Inst))
    return hash_combine(GCR->getOpcode(), GCR->getOperand(0),
                        GCR->getBasePtr(), GCR->getDerivedPtr());

  // Hash all of the operands as pointers and mix in the opcode.
  return hash_combine(
      Inst->getOpcode(),
      hash_combine_range(Inst->value_op_begin(), Inst->value_op_end()));
}

bool DenseMapInfo<CallValue>::isEqual(CallValue LHS, CallValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;
  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;

  // See comment above in `getHashValue()`.
  if (const GCRelocateInst *GCR1 = dyn_cast<GCRelocateInst>(LHSI))
    if (const GCRelocateInst *GCR2 = dyn_cast<GCRelocateInst>(RHSI))
      return GCR1->getOperand(0) == GCR2->getOperand(0) &&
             GCR1->getBasePtr() == GCR2->getBasePtr() &&
             GCR1->getDerivedPtr() == GCR2->getDerivedPtr();

  return LHSI->isIdenticalTo(RHSI);
}

//===----------------------------------------------------------------------===//
// EarlyCSE implementation
//===----------------------------------------------------------------------===//

namespace {

/// A simple and fast domtree-based CSE pass.
///
/// This pass does a simple depth-first walk over the dominator tree,
/// eliminating trivially redundant instructions and using instsimplify to
/// canonicalize things as it goes. It is intended to be fast and catch obvious
/// cases so that instcombine and other passes are more effective. It is
/// expected that a later pass of GVN will catch the interesting/hard cases.
class EarlyCSE {
public:
  const TargetLibraryInfo &TLI;
  const TargetTransformInfo &TTI;
  DominatorTree &DT;
  AssumptionCache &AC;
  const SimplifyQuery SQ;
  MemorySSA *MSSA;
  std::unique_ptr<MemorySSAUpdater> MSSAUpdater;

  using AllocatorTy =
      RecyclingAllocator<BumpPtrAllocator,
                         ScopedHashTableVal<SimpleValue, Value *>>;
  using ScopedHTType =
      ScopedHashTable<SimpleValue, Value *, DenseMapInfo<SimpleValue>,
                      AllocatorTy>;

  /// A scoped hash table of the current values of all of our simple
  /// scalar expressions.
  ///
  /// As we walk down the domtree, we look to see if instructions are in this:
  /// if so, we replace them with what we find, otherwise we insert them so
  /// that dominated values can succeed in their lookup.
  ScopedHTType AvailableValues;

  /// A scoped hash table of the current values of previously encountered
  /// memory locations.
  ///
  /// This allows us to get efficient access to dominating loads or stores when
  /// we have a fully redundant load.  In addition to the most recent load, we
  /// keep track of a generation count of the read, which is compared against
  /// the current generation count.  The current generation count is incremented
  /// after every possibly writing memory operation, which ensures that we only
  /// CSE loads with other loads that have no intervening store.  Ordering
  /// events (such as fences or atomic instructions) increment the generation
  /// count as well; essentially, we model these as writes to all possible
  /// locations.  Note that atomic and/or volatile loads and stores can be
  /// present the table; it is the responsibility of the consumer to inspect
  /// the atomicity/volatility if needed.
  struct LoadValue {
    Instruction *DefInst = nullptr;
    unsigned Generation = 0;
    int MatchingId = -1;
    bool IsAtomic = false;

    LoadValue() = default;
    LoadValue(Instruction *Inst, unsigned Generation, unsigned MatchingId,
              bool IsAtomic)
        : DefInst(Inst), Generation(Generation), MatchingId(MatchingId),
          IsAtomic(IsAtomic) {}
  };

  using LoadMapAllocator =
      RecyclingAllocator<BumpPtrAllocator,
                         ScopedHashTableVal<Value *, LoadValue>>;
  using LoadHTType =
      ScopedHashTable<Value *, LoadValue, DenseMapInfo<Value *>,
                      LoadMapAllocator>;

  LoadHTType AvailableLoads;

  // A scoped hash table mapping memory locations (represented as typed
  // addresses) to generation numbers at which that memory location became
  // (henceforth indefinitely) invariant.
  using InvariantMapAllocator =
      RecyclingAllocator<BumpPtrAllocator,
                         ScopedHashTableVal<MemoryLocation, unsigned>>;
  using InvariantHTType =
      ScopedHashTable<MemoryLocation, unsigned, DenseMapInfo<MemoryLocation>,
                      InvariantMapAllocator>;
  InvariantHTType AvailableInvariants;

  /// A scoped hash table of the current values of read-only call
  /// values.
  ///
  /// It uses the same generation count as loads.
  using CallHTType =
      ScopedHashTable<CallValue, std::pair<Instruction *, unsigned>>;
  CallHTType AvailableCalls;

  /// This is the current generation of the memory value.
  unsigned CurrentGeneration = 0;

  /// Set up the EarlyCSE runner for a particular function.
  EarlyCSE(const DataLayout &DL, const TargetLibraryInfo &TLI,
           const TargetTransformInfo &TTI, DominatorTree &DT,
           AssumptionCache &AC, MemorySSA *MSSA)
      : TLI(TLI), TTI(TTI), DT(DT), AC(AC), SQ(DL, &TLI, &DT, &AC), MSSA(MSSA),
        MSSAUpdater(std::make_unique<MemorySSAUpdater>(MSSA)) {}

  bool run();

private:
  unsigned ClobberCounter = 0;
  // Almost a POD, but needs to call the constructors for the scoped hash
  // tables so that a new scope gets pushed on. These are RAII so that the
  // scope gets popped when the NodeScope is destroyed.
  class NodeScope {
  public:
    NodeScope(ScopedHTType &AvailableValues, LoadHTType &AvailableLoads,
              InvariantHTType &AvailableInvariants, CallHTType &AvailableCalls)
      : Scope(AvailableValues), LoadScope(AvailableLoads),
        InvariantScope(AvailableInvariants), CallScope(AvailableCalls) {}
    NodeScope(const NodeScope &) = delete;
    NodeScope &operator=(const NodeScope &) = delete;

  private:
    ScopedHTType::ScopeTy Scope;
    LoadHTType::ScopeTy LoadScope;
    InvariantHTType::ScopeTy InvariantScope;
    CallHTType::ScopeTy CallScope;
  };

  // Contains all the needed information to create a stack for doing a depth
  // first traversal of the tree. This includes scopes for values, loads, and
  // calls as well as the generation. There is a child iterator so that the
  // children do not need to be store separately.
  class StackNode {
  public:
    StackNode(ScopedHTType &AvailableValues, LoadHTType &AvailableLoads,
              InvariantHTType &AvailableInvariants, CallHTType &AvailableCalls,
              unsigned cg, DomTreeNode *n, DomTreeNode::const_iterator child,
              DomTreeNode::const_iterator end)
        : CurrentGeneration(cg), ChildGeneration(cg), Node(n), ChildIter(child),
          EndIter(end),
          Scopes(AvailableValues, AvailableLoads, AvailableInvariants,
                 AvailableCalls)
          {}
    StackNode(const StackNode &) = delete;
    StackNode &operator=(const StackNode &) = delete;

    // Accessors.
    unsigned currentGeneration() { return CurrentGeneration; }
    unsigned childGeneration() { return ChildGeneration; }
    void childGeneration(unsigned generation) { ChildGeneration = generation; }
    DomTreeNode *node() { return Node; }
    DomTreeNode::const_iterator childIter() { return ChildIter; }

    DomTreeNode *nextChild() {
      DomTreeNode *child = *ChildIter;
      ++ChildIter;
      return child;
    }

    DomTreeNode::const_iterator end() { return EndIter; }
    bool isProcessed() { return Processed; }
    void process() { Processed = true; }

  private:
    unsigned CurrentGeneration;
    unsigned ChildGeneration;
    DomTreeNode *Node;
    DomTreeNode::const_iterator ChildIter;
    DomTreeNode::const_iterator EndIter;
    NodeScope Scopes;
    bool Processed = false;
  };

  /// Wrapper class to handle memory instructions, including loads,
  /// stores and intrinsic loads and stores defined by the target.
  class ParseMemoryInst {
  public:
    ParseMemoryInst(Instruction *Inst, const TargetTransformInfo &TTI)
      : Inst(Inst) {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst))
        if (TTI.getTgtMemIntrinsic(II, Info))
          IsTargetMemInst = true;
    }

    bool isLoad() const {
      if (IsTargetMemInst) return Info.ReadMem;
      return isa<LoadInst>(Inst);
    }

    bool isStore() const {
      if (IsTargetMemInst) return Info.WriteMem;
      return isa<StoreInst>(Inst);
    }

    bool isAtomic() const {
      if (IsTargetMemInst)
        return Info.Ordering != AtomicOrdering::NotAtomic;
      return Inst->isAtomic();
    }

    bool isUnordered() const {
      if (IsTargetMemInst)
        return Info.isUnordered();

      if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        return LI->isUnordered();
      } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        return SI->isUnordered();
      }
      // Conservative answer
      return !Inst->isAtomic();
    }

    bool isVolatile() const {
      if (IsTargetMemInst)
        return Info.IsVolatile;

      if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        return LI->isVolatile();
      } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        return SI->isVolatile();
      }
      // Conservative answer
      return true;
    }

    bool isInvariantLoad() const {
      if (auto *LI = dyn_cast<LoadInst>(Inst))
        return LI->hasMetadata(LLVMContext::MD_invariant_load);
      return false;
    }

    bool isMatchingMemLoc(const ParseMemoryInst &Inst) const {
      return (getPointerOperand() == Inst.getPointerOperand() &&
              getMatchingId() == Inst.getMatchingId());
    }

    bool isValid() const { return getPointerOperand() != nullptr; }

    // For regular (non-intrinsic) loads/stores, this is set to -1. For
    // intrinsic loads/stores, the id is retrieved from the corresponding
    // field in the MemIntrinsicInfo structure.  That field contains
    // non-negative values only.
    int getMatchingId() const {
      if (IsTargetMemInst) return Info.MatchingId;
      return -1;
    }

    Value *getPointerOperand() const {
      if (IsTargetMemInst) return Info.PtrVal;
      return getLoadStorePointerOperand(Inst);
    }

    bool mayReadFromMemory() const {
      if (IsTargetMemInst) return Info.ReadMem;
      return Inst->mayReadFromMemory();
    }

    bool mayWriteToMemory() const {
      if (IsTargetMemInst) return Info.WriteMem;
      return Inst->mayWriteToMemory();
    }

  private:
    bool IsTargetMemInst = false;
    MemIntrinsicInfo Info;
    Instruction *Inst;
  };

  bool processNode(DomTreeNode *Node);

  bool handleBranchCondition(Instruction *CondInst, const BranchInst *BI,
                             const BasicBlock *BB, const BasicBlock *Pred);

  Value *getOrCreateResult(Value *Inst, Type *ExpectedType) const {
    if (auto *LI = dyn_cast<LoadInst>(Inst))
      return LI;
    if (auto *SI = dyn_cast<StoreInst>(Inst))
      return SI->getValueOperand();
    assert(isa<IntrinsicInst>(Inst) && "Instruction not supported");
    return TTI.getOrCreateResultFromMemIntrinsic(cast<IntrinsicInst>(Inst),
                                                 ExpectedType);
  }

  /// Return true if the instruction is known to only operate on memory
  /// provably invariant in the given "generation".
  bool isOperatingOnInvariantMemAt(Instruction *I, unsigned GenAt);

  bool isSameMemGeneration(unsigned EarlierGeneration, unsigned LaterGeneration,
                           Instruction *EarlierInst, Instruction *LaterInst);

  void removeMSSA(Instruction &Inst) {
    if (!MSSA)
      return;
    if (VerifyMemorySSA)
      MSSA->verifyMemorySSA();
    // Removing a store here can leave MemorySSA in an unoptimized state by
    // creating MemoryPhis that have identical arguments and by creating
    // MemoryUses whose defining access is not an actual clobber. The phi case
    // is handled by MemorySSA when passing OptimizePhis = true to
    // removeMemoryAccess.  The non-optimized MemoryUse case is lazily updated
    // by MemorySSA's getClobberingMemoryAccess.
    MSSAUpdater->removeMemoryAccess(&Inst, true);
  }
};

} // end anonymous namespace

/// Determine if the memory referenced by LaterInst is from the same heap
/// version as EarlierInst.
/// This is currently called in two scenarios:
///
///   load p
///   ...
///   load p
///
/// and
///
///   x = load p
///   ...
///   store x, p
///
/// in both cases we want to verify that there are no possible writes to the
/// memory referenced by p between the earlier and later instruction.
bool EarlyCSE::isSameMemGeneration(unsigned EarlierGeneration,
                                   unsigned LaterGeneration,
                                   Instruction *EarlierInst,
                                   Instruction *LaterInst) {
  // Check the simple memory generation tracking first.
  if (EarlierGeneration == LaterGeneration)
    return true;

  if (!MSSA)
    return false;

  // If MemorySSA has determined that one of EarlierInst or LaterInst does not
  // read/write memory, then we can safely return true here.
  // FIXME: We could be more aggressive when checking doesNotAccessMemory(),
  // onlyReadsMemory(), mayReadFromMemory(), and mayWriteToMemory() in this pass
  // by also checking the MemorySSA MemoryAccess on the instruction.  Initial
  // experiments suggest this isn't worthwhile, at least for C/C++ code compiled
  // with the default optimization pipeline.
  auto *EarlierMA = MSSA->getMemoryAccess(EarlierInst);
  if (!EarlierMA)
    return true;
  auto *LaterMA = MSSA->getMemoryAccess(LaterInst);
  if (!LaterMA)
    return true;

  // Since we know LaterDef dominates LaterInst and EarlierInst dominates
  // LaterInst, if LaterDef dominates EarlierInst then it can't occur between
  // EarlierInst and LaterInst and neither can any other write that potentially
  // clobbers LaterInst.
  MemoryAccess *LaterDef;
  if (ClobberCounter < EarlyCSEMssaOptCap) {
    LaterDef = MSSA->getWalker()->getClobberingMemoryAccess(LaterInst);
    ClobberCounter++;
  } else
    LaterDef = LaterMA->getDefiningAccess();

  return MSSA->dominates(LaterDef, EarlierMA);
}

bool EarlyCSE::isOperatingOnInvariantMemAt(Instruction *I, unsigned GenAt) {
  // A location loaded from with an invariant_load is assumed to *never* change
  // within the visible scope of the compilation.
  if (auto *LI = dyn_cast<LoadInst>(I))
    if (LI->hasMetadata(LLVMContext::MD_invariant_load))
      return true;

  auto MemLocOpt = MemoryLocation::getOrNone(I);
  if (!MemLocOpt)
    // "target" intrinsic forms of loads aren't currently known to
    // MemoryLocation::get.  TODO
    return false;
  MemoryLocation MemLoc = *MemLocOpt;
  if (!AvailableInvariants.count(MemLoc))
    return false;

  // Is the generation at which this became invariant older than the
  // current one?
  return AvailableInvariants.lookup(MemLoc) <= GenAt;
}

bool EarlyCSE::handleBranchCondition(Instruction *CondInst,
                                     const BranchInst *BI, const BasicBlock *BB,
                                     const BasicBlock *Pred) {
  assert(BI->isConditional() && "Should be a conditional branch!");
  assert(BI->getCondition() == CondInst && "Wrong condition?");
  assert(BI->getSuccessor(0) == BB || BI->getSuccessor(1) == BB);
  auto *TorF = (BI->getSuccessor(0) == BB)
                   ? ConstantInt::getTrue(BB->getContext())
                   : ConstantInt::getFalse(BB->getContext());
  auto MatchBinOp = [](Instruction *I, unsigned Opcode) {
    if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(I))
      return BOp->getOpcode() == Opcode;
    return false;
  };
  // If the condition is AND operation, we can propagate its operands into the
  // true branch. If it is OR operation, we can propagate them into the false
  // branch.
  unsigned PropagateOpcode =
      (BI->getSuccessor(0) == BB) ? Instruction::And : Instruction::Or;

  bool MadeChanges = false;
  SmallVector<Instruction *, 4> WorkList;
  SmallPtrSet<Instruction *, 4> Visited;
  WorkList.push_back(CondInst);
  while (!WorkList.empty()) {
    Instruction *Curr = WorkList.pop_back_val();

    AvailableValues.insert(Curr, TorF);
    LLVM_DEBUG(dbgs() << "EarlyCSE CVP: Add conditional value for '"
                      << Curr->getName() << "' as " << *TorF << " in "
                      << BB->getName() << "\n");
    if (!DebugCounter::shouldExecute(CSECounter)) {
      LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
    } else {
      // Replace all dominated uses with the known value.
      if (unsigned Count = replaceDominatedUsesWith(Curr, TorF, DT,
                                                    BasicBlockEdge(Pred, BB))) {
        NumCSECVP += Count;
        MadeChanges = true;
      }
    }

    if (MatchBinOp(Curr, PropagateOpcode))
      for (auto &Op : cast<BinaryOperator>(Curr)->operands())
        if (Instruction *OPI = dyn_cast<Instruction>(Op))
          if (SimpleValue::canHandle(OPI) && Visited.insert(OPI).second)
            WorkList.push_back(OPI);
  }

  return MadeChanges;
}

bool EarlyCSE::processNode(DomTreeNode *Node) {
  bool Changed = false;
  BasicBlock *BB = Node->getBlock();

  // If this block has a single predecessor, then the predecessor is the parent
  // of the domtree node and all of the live out memory values are still current
  // in this block.  If this block has multiple predecessors, then they could
  // have invalidated the live-out memory values of our parent value.  For now,
  // just be conservative and invalidate memory if this block has multiple
  // predecessors.
  if (!BB->getSinglePredecessor())
    ++CurrentGeneration;

  // If this node has a single predecessor which ends in a conditional branch,
  // we can infer the value of the branch condition given that we took this
  // path.  We need the single predecessor to ensure there's not another path
  // which reaches this block where the condition might hold a different
  // value.  Since we're adding this to the scoped hash table (like any other
  // def), it will have been popped if we encounter a future merge block.
  if (BasicBlock *Pred = BB->getSinglePredecessor()) {
    auto *BI = dyn_cast<BranchInst>(Pred->getTerminator());
    if (BI && BI->isConditional()) {
      auto *CondInst = dyn_cast<Instruction>(BI->getCondition());
      if (CondInst && SimpleValue::canHandle(CondInst))
        Changed |= handleBranchCondition(CondInst, BI, BB, Pred);
    }
  }

  /// LastStore - Keep track of the last non-volatile store that we saw... for
  /// as long as there in no instruction that reads memory.  If we see a store
  /// to the same location, we delete the dead store.  This zaps trivial dead
  /// stores which can occur in bitfield code among other things.
  Instruction *LastStore = nullptr;

  // See if any instructions in the block can be eliminated.  If so, do it.  If
  // not, add them to AvailableValues.
  for (Instruction &Inst : make_early_inc_range(BB->getInstList())) {
    // Dead instructions should just be removed.
    if (isInstructionTriviallyDead(&Inst, &TLI)) {
      LLVM_DEBUG(dbgs() << "EarlyCSE DCE: " << Inst << '\n');
      if (!DebugCounter::shouldExecute(CSECounter)) {
        LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
        continue;
      }

      salvageKnowledge(&Inst, &AC);
      salvageDebugInfo(Inst);
      removeMSSA(Inst);
      Inst.eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }

    // Skip assume intrinsics, they don't really have side effects (although
    // they're marked as such to ensure preservation of control dependencies),
    // and this pass will not bother with its removal. However, we should mark
    // its condition as true for all dominated blocks.
    if (match(&Inst, m_Intrinsic<Intrinsic::assume>())) {
      auto *CondI =
          dyn_cast<Instruction>(cast<CallInst>(Inst).getArgOperand(0));
      if (CondI && SimpleValue::canHandle(CondI)) {
        LLVM_DEBUG(dbgs() << "EarlyCSE considering assumption: " << Inst
                          << '\n');
        AvailableValues.insert(CondI, ConstantInt::getTrue(BB->getContext()));
      } else
        LLVM_DEBUG(dbgs() << "EarlyCSE skipping assumption: " << Inst << '\n');
      continue;
    }

    // Skip sideeffect intrinsics, for the same reason as assume intrinsics.
    if (match(&Inst, m_Intrinsic<Intrinsic::sideeffect>())) {
      LLVM_DEBUG(dbgs() << "EarlyCSE skipping sideeffect: " << Inst << '\n');
      continue;
    }

    // We can skip all invariant.start intrinsics since they only read memory,
    // and we can forward values across it. For invariant starts without
    // invariant ends, we can use the fact that the invariantness never ends to
    // start a scope in the current generaton which is true for all future
    // generations.  Also, we dont need to consume the last store since the
    // semantics of invariant.start allow us to perform   DSE of the last
    // store, if there was a store following invariant.start. Consider:
    //
    // store 30, i8* p
    // invariant.start(p)
    // store 40, i8* p
    // We can DSE the store to 30, since the store 40 to invariant location p
    // causes undefined behaviour.
    if (match(&Inst, m_Intrinsic<Intrinsic::invariant_start>())) {
      // If there are any uses, the scope might end.
      if (!Inst.use_empty())
        continue;
      MemoryLocation MemLoc =
          MemoryLocation::getForArgument(&cast<CallInst>(Inst), 1, TLI);
      // Don't start a scope if we already have a better one pushed
      if (!AvailableInvariants.count(MemLoc))
        AvailableInvariants.insert(MemLoc, CurrentGeneration);
      continue;
    }

    if (isGuard(&Inst)) {
      if (auto *CondI =
              dyn_cast<Instruction>(cast<CallInst>(Inst).getArgOperand(0))) {
        if (SimpleValue::canHandle(CondI)) {
          // Do we already know the actual value of this condition?
          if (auto *KnownCond = AvailableValues.lookup(CondI)) {
            // Is the condition known to be true?
            if (isa<ConstantInt>(KnownCond) &&
                cast<ConstantInt>(KnownCond)->isOne()) {
              LLVM_DEBUG(dbgs()
                         << "EarlyCSE removing guard: " << Inst << '\n');
              salvageKnowledge(&Inst, &AC);
              removeMSSA(Inst);
              Inst.eraseFromParent();
              Changed = true;
              continue;
            } else
              // Use the known value if it wasn't true.
              cast<CallInst>(Inst).setArgOperand(0, KnownCond);
          }
          // The condition we're on guarding here is true for all dominated
          // locations.
          AvailableValues.insert(CondI, ConstantInt::getTrue(BB->getContext()));
        }
      }

      // Guard intrinsics read all memory, but don't write any memory.
      // Accordingly, don't update the generation but consume the last store (to
      // avoid an incorrect DSE).
      LastStore = nullptr;
      continue;
    }

    // If the instruction can be simplified (e.g. X+0 = X) then replace it with
    // its simpler value.
    if (Value *V = SimplifyInstruction(&Inst, SQ)) {
      LLVM_DEBUG(dbgs() << "EarlyCSE Simplify: " << Inst << "  to: " << *V
                        << '\n');
      if (!DebugCounter::shouldExecute(CSECounter)) {
        LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
      } else {
        bool Killed = false;
        if (!Inst.use_empty()) {
          Inst.replaceAllUsesWith(V);
          Changed = true;
        }
        if (isInstructionTriviallyDead(&Inst, &TLI)) {
          salvageKnowledge(&Inst, &AC);
          removeMSSA(Inst);
          Inst.eraseFromParent();
          Changed = true;
          Killed = true;
        }
        if (Changed)
          ++NumSimplify;
        if (Killed)
          continue;
      }
    }

    // If this is a simple instruction that we can value number, process it.
    if (SimpleValue::canHandle(&Inst)) {
      // See if the instruction has an available value.  If so, use it.
      if (Value *V = AvailableValues.lookup(&Inst)) {
        LLVM_DEBUG(dbgs() << "EarlyCSE CSE: " << Inst << "  to: " << *V
                          << '\n');
        if (!DebugCounter::shouldExecute(CSECounter)) {
          LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
          continue;
        }
        if (auto *I = dyn_cast<Instruction>(V))
          I->andIRFlags(&Inst);
        Inst.replaceAllUsesWith(V);
        salvageKnowledge(&Inst, &AC);
        removeMSSA(Inst);
        Inst.eraseFromParent();
        Changed = true;
        ++NumCSE;
        continue;
      }

      // Otherwise, just remember that this value is available.
      AvailableValues.insert(&Inst, &Inst);
      continue;
    }

    ParseMemoryInst MemInst(&Inst, TTI);
    // If this is a non-volatile load, process it.
    if (MemInst.isValid() && MemInst.isLoad()) {
      // (conservatively) we can't peak past the ordering implied by this
      // operation, but we can add this load to our set of available values
      if (MemInst.isVolatile() || !MemInst.isUnordered()) {
        LastStore = nullptr;
        ++CurrentGeneration;
      }

      if (MemInst.isInvariantLoad()) {
        // If we pass an invariant load, we know that memory location is
        // indefinitely constant from the moment of first dereferenceability.
        // We conservatively treat the invariant_load as that moment.  If we
        // pass a invariant load after already establishing a scope, don't
        // restart it since we want to preserve the earliest point seen.
        auto MemLoc = MemoryLocation::get(&Inst);
        if (!AvailableInvariants.count(MemLoc))
          AvailableInvariants.insert(MemLoc, CurrentGeneration);
      }

      // If we have an available version of this load, and if it is the right
      // generation or the load is known to be from an invariant location,
      // replace this instruction.
      //
      // If either the dominating load or the current load are invariant, then
      // we can assume the current load loads the same value as the dominating
      // load.
      LoadValue InVal = AvailableLoads.lookup(MemInst.getPointerOperand());
      if (InVal.DefInst != nullptr &&
          InVal.MatchingId == MemInst.getMatchingId() &&
          // We don't yet handle removing loads with ordering of any kind.
          !MemInst.isVolatile() && MemInst.isUnordered() &&
          // We can't replace an atomic load with one which isn't also atomic.
          InVal.IsAtomic >= MemInst.isAtomic() &&
          (isOperatingOnInvariantMemAt(&Inst, InVal.Generation) ||
           isSameMemGeneration(InVal.Generation, CurrentGeneration,
                               InVal.DefInst, &Inst))) {
        Value *Op = getOrCreateResult(InVal.DefInst, Inst.getType());
        if (Op != nullptr) {
          LLVM_DEBUG(dbgs() << "EarlyCSE CSE LOAD: " << Inst
                            << "  to: " << *InVal.DefInst << '\n');
          if (!DebugCounter::shouldExecute(CSECounter)) {
            LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
            continue;
          }
          if (!Inst.use_empty())
            Inst.replaceAllUsesWith(Op);
          salvageKnowledge(&Inst, &AC);
          removeMSSA(Inst);
          Inst.eraseFromParent();
          Changed = true;
          ++NumCSELoad;
          continue;
        }
      }

      // Otherwise, remember that we have this instruction.
      AvailableLoads.insert(MemInst.getPointerOperand(),
                            LoadValue(&Inst, CurrentGeneration,
                                      MemInst.getMatchingId(),
                                      MemInst.isAtomic()));
      LastStore = nullptr;
      continue;
    }

    // If this instruction may read from memory or throw (and potentially read
    // from memory in the exception handler), forget LastStore.  Load/store
    // intrinsics will indicate both a read and a write to memory.  The target
    // may override this (e.g. so that a store intrinsic does not read from
    // memory, and thus will be treated the same as a regular store for
    // commoning purposes).
    if ((Inst.mayReadFromMemory() || Inst.mayThrow()) &&
        !(MemInst.isValid() && !MemInst.mayReadFromMemory()))
      LastStore = nullptr;

    // If this is a read-only call, process it.
    if (CallValue::canHandle(&Inst)) {
      // If we have an available version of this call, and if it is the right
      // generation, replace this instruction.
      std::pair<Instruction *, unsigned> InVal = AvailableCalls.lookup(&Inst);
      if (InVal.first != nullptr &&
          isSameMemGeneration(InVal.second, CurrentGeneration, InVal.first,
                              &Inst)) {
        LLVM_DEBUG(dbgs() << "EarlyCSE CSE CALL: " << Inst
                          << "  to: " << *InVal.first << '\n');
        if (!DebugCounter::shouldExecute(CSECounter)) {
          LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
          continue;
        }
        if (!Inst.use_empty())
          Inst.replaceAllUsesWith(InVal.first);
        salvageKnowledge(&Inst, &AC);
        removeMSSA(Inst);
        Inst.eraseFromParent();
        Changed = true;
        ++NumCSECall;
        continue;
      }

      // Otherwise, remember that we have this instruction.
      AvailableCalls.insert(&Inst, std::make_pair(&Inst, CurrentGeneration));
      continue;
    }

    // A release fence requires that all stores complete before it, but does
    // not prevent the reordering of following loads 'before' the fence.  As a
    // result, we don't need to consider it as writing to memory and don't need
    // to advance the generation.  We do need to prevent DSE across the fence,
    // but that's handled above.
    if (auto *FI = dyn_cast<FenceInst>(&Inst))
      if (FI->getOrdering() == AtomicOrdering::Release) {
        assert(Inst.mayReadFromMemory() && "relied on to prevent DSE above");
        continue;
      }

    // write back DSE - If we write back the same value we just loaded from
    // the same location and haven't passed any intervening writes or ordering
    // operations, we can remove the write.  The primary benefit is in allowing
    // the available load table to remain valid and value forward past where
    // the store originally was.
    if (MemInst.isValid() && MemInst.isStore()) {
      LoadValue InVal = AvailableLoads.lookup(MemInst.getPointerOperand());
      if (InVal.DefInst &&
          InVal.DefInst == getOrCreateResult(&Inst, InVal.DefInst->getType()) &&
          InVal.MatchingId == MemInst.getMatchingId() &&
          // We don't yet handle removing stores with ordering of any kind.
          !MemInst.isVolatile() && MemInst.isUnordered() &&
          (isOperatingOnInvariantMemAt(&Inst, InVal.Generation) ||
           isSameMemGeneration(InVal.Generation, CurrentGeneration,
                               InVal.DefInst, &Inst))) {
        // It is okay to have a LastStore to a different pointer here if MemorySSA
        // tells us that the load and store are from the same memory generation.
        // In that case, LastStore should keep its present value since we're
        // removing the current store.
        assert((!LastStore ||
                ParseMemoryInst(LastStore, TTI).getPointerOperand() ==
                    MemInst.getPointerOperand() ||
                MSSA) &&
               "can't have an intervening store if not using MemorySSA!");
        LLVM_DEBUG(dbgs() << "EarlyCSE DSE (writeback): " << Inst << '\n');
        if (!DebugCounter::shouldExecute(CSECounter)) {
          LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
          continue;
        }
        salvageKnowledge(&Inst, &AC);
        removeMSSA(Inst);
        Inst.eraseFromParent();
        Changed = true;
        ++NumDSE;
        // We can avoid incrementing the generation count since we were able
        // to eliminate this store.
        continue;
      }
    }

    // Okay, this isn't something we can CSE at all.  Check to see if it is
    // something that could modify memory.  If so, our available memory values
    // cannot be used so bump the generation count.
    if (Inst.mayWriteToMemory()) {
      ++CurrentGeneration;

      if (MemInst.isValid() && MemInst.isStore()) {
        // We do a trivial form of DSE if there are two stores to the same
        // location with no intervening loads.  Delete the earlier store.
        // At the moment, we don't remove ordered stores, but do remove
        // unordered atomic stores.  There's no special requirement (for
        // unordered atomics) about removing atomic stores only in favor of
        // other atomic stores since we were going to execute the non-atomic
        // one anyway and the atomic one might never have become visible.
        if (LastStore) {
          ParseMemoryInst LastStoreMemInst(LastStore, TTI);
          assert(LastStoreMemInst.isUnordered() &&
                 !LastStoreMemInst.isVolatile() &&
                 "Violated invariant");
          if (LastStoreMemInst.isMatchingMemLoc(MemInst)) {
            LLVM_DEBUG(dbgs() << "EarlyCSE DEAD STORE: " << *LastStore
                              << "  due to: " << Inst << '\n');
            if (!DebugCounter::shouldExecute(CSECounter)) {
              LLVM_DEBUG(dbgs() << "Skipping due to debug counter\n");
            } else {
              salvageKnowledge(&Inst, &AC);
              removeMSSA(*LastStore);
              LastStore->eraseFromParent();
              Changed = true;
              ++NumDSE;
              LastStore = nullptr;
            }
          }
          // fallthrough - we can exploit information about this store
        }

        // Okay, we just invalidated anything we knew about loaded values.  Try
        // to salvage *something* by remembering that the stored value is a live
        // version of the pointer.  It is safe to forward from volatile stores
        // to non-volatile loads, so we don't have to check for volatility of
        // the store.
        AvailableLoads.insert(MemInst.getPointerOperand(),
                              LoadValue(&Inst, CurrentGeneration,
                                        MemInst.getMatchingId(),
                                        MemInst.isAtomic()));

        // Remember that this was the last unordered store we saw for DSE. We
        // don't yet handle DSE on ordered or volatile stores since we don't
        // have a good way to model the ordering requirement for following
        // passes  once the store is removed.  We could insert a fence, but
        // since fences are slightly stronger than stores in their ordering,
        // it's not clear this is a profitable transform. Another option would
        // be to merge the ordering with that of the post dominating store.
        if (MemInst.isUnordered() && !MemInst.isVolatile())
          LastStore = &Inst;
        else
          LastStore = nullptr;
      }
    }
  }

  return Changed;
}

bool EarlyCSE::run() {
  // Note, deque is being used here because there is significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. For more information see the mailing list
  // discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<StackNode *> nodesToProcess;

  bool Changed = false;

  // Process the root node.
  nodesToProcess.push_back(new StackNode(
      AvailableValues, AvailableLoads, AvailableInvariants, AvailableCalls,
      CurrentGeneration, DT.getRootNode(),
      DT.getRootNode()->begin(), DT.getRootNode()->end()));

  assert(!CurrentGeneration && "Create a new EarlyCSE instance to rerun it.");

  // Process the stack.
  while (!nodesToProcess.empty()) {
    // Grab the first item off the stack. Set the current generation, remove
    // the node from the stack, and process it.
    StackNode *NodeToProcess = nodesToProcess.back();

    // Initialize class members.
    CurrentGeneration = NodeToProcess->currentGeneration();

    // Check if the node needs to be processed.
    if (!NodeToProcess->isProcessed()) {
      // Process the node.
      Changed |= processNode(NodeToProcess->node());
      NodeToProcess->childGeneration(CurrentGeneration);
      NodeToProcess->process();
    } else if (NodeToProcess->childIter() != NodeToProcess->end()) {
      // Push the next child onto the stack.
      DomTreeNode *child = NodeToProcess->nextChild();
      nodesToProcess.push_back(
          new StackNode(AvailableValues, AvailableLoads, AvailableInvariants,
                        AvailableCalls, NodeToProcess->childGeneration(),
                        child, child->begin(), child->end()));
    } else {
      // It has been processed, and there are no more children to process,
      // so delete it and pop it off the stack.
      delete NodeToProcess;
      nodesToProcess.pop_back();
    }
  } // while (!nodes...)

  return Changed;
}

PreservedAnalyses EarlyCSEPass::run(Function &F,
                                    FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto *MSSA =
      UseMemorySSA ? &AM.getResult<MemorySSAAnalysis>(F).getMSSA() : nullptr;

  EarlyCSE CSE(F.getParent()->getDataLayout(), TLI, TTI, DT, AC, MSSA);

  if (!CSE.run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GlobalsAA>();
  if (UseMemorySSA)
    PA.preserve<MemorySSAAnalysis>();
  return PA;
}

namespace {

/// A simple and fast domtree-based CSE pass.
///
/// This pass does a simple depth-first walk over the dominator tree,
/// eliminating trivially redundant instructions and using instsimplify to
/// canonicalize things as it goes. It is intended to be fast and catch obvious
/// cases so that instcombine and other passes are more effective. It is
/// expected that a later pass of GVN will catch the interesting/hard cases.
template<bool UseMemorySSA>
class EarlyCSELegacyCommonPass : public FunctionPass {
public:
  static char ID;

  EarlyCSELegacyCommonPass() : FunctionPass(ID) {
    if (UseMemorySSA)
      initializeEarlyCSEMemSSALegacyPassPass(*PassRegistry::getPassRegistry());
    else
      initializeEarlyCSELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *MSSA =
        UseMemorySSA ? &getAnalysis<MemorySSAWrapperPass>().getMSSA() : nullptr;

    EarlyCSE CSE(F.getParent()->getDataLayout(), TLI, TTI, DT, AC, MSSA);

    return CSE.run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    if (UseMemorySSA) {
      AU.addRequired<MemorySSAWrapperPass>();
      AU.addPreserved<MemorySSAWrapperPass>();
    }
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

using EarlyCSELegacyPass = EarlyCSELegacyCommonPass</*UseMemorySSA=*/false>;

template<>
char EarlyCSELegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(EarlyCSELegacyPass, "early-cse", "Early CSE", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(EarlyCSELegacyPass, "early-cse", "Early CSE", false, false)

using EarlyCSEMemSSALegacyPass =
    EarlyCSELegacyCommonPass</*UseMemorySSA=*/true>;

template<>
char EarlyCSEMemSSALegacyPass::ID = 0;

FunctionPass *llvm::createEarlyCSEPass(bool UseMemorySSA) {
  if (UseMemorySSA)
    return new EarlyCSEMemSSALegacyPass();
  else
    return new EarlyCSELegacyPass();
}

INITIALIZE_PASS_BEGIN(EarlyCSEMemSSALegacyPass, "early-cse-memssa",
                      "Early CSE w/ MemorySSA", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_END(EarlyCSEMemSSALegacyPass, "early-cse-memssa",
                    "Early CSE w/ MemorySSA", false, false)
