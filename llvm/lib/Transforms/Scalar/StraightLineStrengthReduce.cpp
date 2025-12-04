//===- StraightLineStrengthReduce.cpp - -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements straight-line strength reduction (SLSR). Unlike loop
// strength reduction, this algorithm is designed to reduce arithmetic
// redundancy in straight-line code instead of loops. It has proven to be
// effective in simplifying arithmetic statements derived from an unrolled loop.
// It can also simplify the logic of SeparateConstOffsetFromGEP.
//
// There are many optimizations we can perform in the domain of SLSR.
// We look for strength reduction candidates in the following forms:
//
// Form Add: B + i * S
// Form Mul: (B + i) * S
// Form GEP: &B[i * S]
//
// where S is an integer variable, and i is a constant integer. If we found two
// candidates S1 and S2 in the same form and S1 dominates S2, we may rewrite S2
// in a simpler way with respect to S1 (index delta). For example,
//
// S1: X = B + i * S
// S2: Y = B + i' * S   => X + (i' - i) * S
//
// S1: X = (B + i) * S
// S2: Y = (B + i') * S => X + (i' - i) * S
//
// S1: X = &B[i * S]
// S2: Y = &B[i' * S]   => &X[(i' - i) * S]
//
// Note: (i' - i) * S is folded to the extent possible.
//
// For Add and GEP forms, we can also rewrite a candidate in a simpler way
// with respect to other dominating candidates if their B or S are different
// but other parts are the same. For example,
//
// Base Delta:
// S1: X = B  + i * S
// S2: Y = B' + i * S   => X + (B' - B)
//
// S1: X = &B [i * S]
// S2: Y = &B'[i * S]   => X + (B' - B)
//
// Stride Delta:
// S1: X = B + i * S
// S2: Y = B + i * S'   => X + i * (S' - S)
//
// S1: X = &B[i * S]
// S2: Y = &B[i * S']   => X + i * (S' - S)
//
// PS: Stride delta rewrite on Mul form is usually non-profitable, and Base
// delta rewrite sometimes is profitable, so we do not support them on Mul.
//
// This rewriting is in general a good idea. The code patterns we focus on
// usually come from loop unrolling, so the delta is likely the same
// across iterations and can be reused. When that happens, the optimized form
// takes only one add starting from the second iteration.
//
// When such rewriting is possible, we call S1 a "basis" of S2. When S2 has
// multiple bases, we choose to rewrite S2 with respect to its "immediate"
// basis, the basis that is the closest ancestor in the dominator tree.
//
// TODO:
//
// - Floating point arithmetics when fast math is enabled.

#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <list>
#include <queue>
#include <vector>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "slsr"

static const unsigned UnknownAddressSpace =
    std::numeric_limits<unsigned>::max();

DEBUG_COUNTER(StraightLineStrengthReduceCounter, "slsr-counter",
              "Controls whether rewriteCandidate is executed.");

namespace {

class StraightLineStrengthReduceLegacyPass : public FunctionPass {
  const DataLayout *DL = nullptr;

public:
  static char ID;

  StraightLineStrengthReduceLegacyPass() : FunctionPass(ID) {
    initializeStraightLineStrengthReduceLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    // We do not modify the shape of the CFG.
    AU.setPreservesCFG();
  }

  bool doInitialization(Module &M) override {
    DL = &M.getDataLayout();
    return false;
  }

  bool runOnFunction(Function &F) override;
};

class StraightLineStrengthReduce {
public:
  StraightLineStrengthReduce(const DataLayout *DL, DominatorTree *DT,
                             ScalarEvolution *SE, TargetTransformInfo *TTI)
      : DL(DL), DT(DT), SE(SE), TTI(TTI) {}

  // SLSR candidate. Such a candidate must be in one of the forms described in
  // the header comments.
  struct Candidate {
    enum Kind {
      Invalid, // reserved for the default constructor
      Add,     // B + i * S
      Mul,     // (B + i) * S
      GEP,     // &B[..][i * S][..]
    };

    enum DKind {
      InvalidDelta, // reserved for the default constructor
      IndexDelta,   // Delta is a constant from Index
      BaseDelta,    // Delta is a constant or variable from Base
      StrideDelta,  // Delta is a constant or variable from Stride
    };

    Candidate() = default;
    Candidate(Kind CT, const SCEV *B, ConstantInt *Idx, Value *S,
              Instruction *I, const SCEV *StrideSCEV)
        : CandidateKind(CT), Base(B), Index(Idx), Stride(S), Ins(I),
          StrideSCEV(StrideSCEV) {}

    Kind CandidateKind = Invalid;

    const SCEV *Base = nullptr;
    // TODO: Swap Index and Stride's name.
    // Note that Index and Stride of a GEP candidate do not necessarily have the
    // same integer type. In that case, during rewriting, Stride will be
    // sign-extended or truncated to Index's type.
    ConstantInt *Index = nullptr;

    Value *Stride = nullptr;

    // The instruction this candidate corresponds to. It helps us to rewrite a
    // candidate with respect to its immediate basis. Note that one instruction
    // can correspond to multiple candidates depending on how you associate the
    // expression. For instance,
    //
    // (a + 1) * (b + 2)
    //
    // can be treated as
    //
    // <Base: a, Index: 1, Stride: b + 2>
    //
    // or
    //
    // <Base: b, Index: 2, Stride: a + 1>
    Instruction *Ins = nullptr;

    // Points to the immediate basis of this candidate, or nullptr if we cannot
    // find any basis for this candidate.
    Candidate *Basis = nullptr;

    DKind DeltaKind = InvalidDelta;

    // Store SCEV of Stride to compute delta from different strides
    const SCEV *StrideSCEV = nullptr;

    // Points to (Y - X) that will be used to rewrite this candidate.
    Value *Delta = nullptr;

    /// Cost model: Evaluate the computational efficiency of the candidate.
    ///
    /// Efficiency levels (higher is better):
    ///   ZeroInst (5) - [Variable] or [Const]
    ///   OneInstOneVar (4) - [Variable + Const] or [Variable * Const]
    ///   OneInstTwoVar (3) - [Variable + Variable] or [Variable * Variable]
    ///   TwoInstOneVar (2) - [Const + Const * Variable]
    ///   TwoInstTwoVar (1) - [Variable + Const * Variable]
    enum EfficiencyLevel : unsigned {
      Unknown = 0,
      TwoInstTwoVar = 1,
      TwoInstOneVar = 2,
      OneInstTwoVar = 3,
      OneInstOneVar = 4,
      ZeroInst = 5
    };

    static EfficiencyLevel
    getComputationEfficiency(Kind CandidateKind, const ConstantInt *Index,
                             const Value *Stride, const SCEV *Base = nullptr) {
      bool IsConstantBase = false;
      bool IsZeroBase = false;
      // When evaluating the efficiency of a rewrite, if the Base's SCEV is
      // not available, conservatively assume the base is not constant.
      if (auto *ConstBase = dyn_cast_or_null<SCEVConstant>(Base)) {
        IsConstantBase = true;
        IsZeroBase = ConstBase->getValue()->isZero();
      }

      bool IsConstantStride = isa<ConstantInt>(Stride);
      bool IsZeroStride =
          IsConstantStride && cast<ConstantInt>(Stride)->isZero();
      // All constants
      if (IsConstantBase && IsConstantStride)
        return ZeroInst;

      // (Base + Index) * Stride
      if (CandidateKind == Mul) {
        if (IsZeroStride)
          return ZeroInst;
        if (Index->isZero())
          return (IsConstantStride || IsConstantBase) ? OneInstOneVar
                                                      : OneInstTwoVar;

        if (IsConstantBase)
          return IsZeroBase && (Index->isOne() || Index->isMinusOne())
                     ? ZeroInst
                     : OneInstOneVar;

        if (IsConstantStride) {
          auto *CI = cast<ConstantInt>(Stride);
          return (CI->isOne() || CI->isMinusOne()) ? OneInstOneVar
                                                   : TwoInstOneVar;
        }
        return TwoInstTwoVar;
      }

      // Base + Index * Stride
      assert(CandidateKind == Add || CandidateKind == GEP);
      if (Index->isZero() || IsZeroStride)
        return ZeroInst;

      bool IsSimpleIndex = Index->isOne() || Index->isMinusOne();

      if (IsConstantBase)
        return IsZeroBase ? (IsSimpleIndex ? ZeroInst : OneInstOneVar)
                          : (IsSimpleIndex ? OneInstOneVar : TwoInstOneVar);

      if (IsConstantStride)
        return IsZeroStride ? ZeroInst : OneInstOneVar;

      if (IsSimpleIndex)
        return OneInstTwoVar;

      return TwoInstTwoVar;
    }

    // Evaluate if the given delta is profitable to rewrite this candidate.
    bool isProfitableRewrite(const Value *Delta, const DKind DeltaKind) const {
      // This function cannot accurately evaluate the profit of whole expression
      // with context. A candidate (B + I * S) cannot express whether this
      // instruction needs to compute on its own (I * S), which may be shared
      // with other candidates or may need instructions to compute.
      // If the rewritten form has the same strength, still rewrite to
      // (X + Delta) since it may expose more CSE opportunities on Delta, as
      // unrolled loops usually have identical Delta for each unrolled body.
      //
      // Note, this function should only be used on Index Delta rewrite.
      // Base and Stride delta need context info to evaluate the register
      // pressure impact from variable delta.
      return getComputationEfficiency(CandidateKind, Index, Stride, Base) <=
             getRewriteEfficiency(Delta, DeltaKind);
    }

    // Evaluate the rewrite efficiency of this candidate with its Basis
    EfficiencyLevel getRewriteEfficiency() const {
      return Basis ? getRewriteEfficiency(Delta, DeltaKind) : Unknown;
    }

    // Evaluate the rewrite efficiency of this candidate with a given delta
    EfficiencyLevel getRewriteEfficiency(const Value *Delta,
                                         const DKind DeltaKind) const {
      switch (DeltaKind) {
      case BaseDelta: // [X + Delta]
        return getComputationEfficiency(
            CandidateKind,
            ConstantInt::get(cast<IntegerType>(Delta->getType()), 1), Delta);
      case StrideDelta: // [X + Index * Delta]
        return getComputationEfficiency(CandidateKind, Index, Delta);
      case IndexDelta: // [X + Delta * Stride]
        return getComputationEfficiency(CandidateKind, cast<ConstantInt>(Delta),
                                        Stride);
      default:
        return Unknown;
      }
    }

    bool isHighEfficiency() const {
      return getComputationEfficiency(CandidateKind, Index, Stride, Base) >=
             OneInstOneVar;
    }

    // Verify that this candidate has valid delta components relative to the
    // basis
    bool hasValidDelta(const Candidate &Basis) const {
      switch (DeltaKind) {
      case IndexDelta:
        // Index differs, Base and Stride must match
        return Base == Basis.Base && StrideSCEV == Basis.StrideSCEV;
      case StrideDelta:
        // Stride differs, Base and Index must match
        return Base == Basis.Base && Index == Basis.Index;
      case BaseDelta:
        // Base differs, Stride and Index must match
        return StrideSCEV == Basis.StrideSCEV && Index == Basis.Index;
      default:
        return false;
      }
    }
  };

  bool runOnFunction(Function &F);

private:
  // Fetch straight-line basis for rewriting C, update C.Basis to point to it,
  // and store the delta between C and its Basis in C.Delta.
  void setBasisAndDeltaFor(Candidate &C);
  // Returns whether the candidate can be folded into an addressing mode.
  bool isFoldable(const Candidate &C, TargetTransformInfo *TTI);

  // Checks whether I is in a candidate form. If so, adds all the matching forms
  // to Candidates, and tries to find the immediate basis for each of them.
  void allocateCandidatesAndFindBasis(Instruction *I);

  // Allocate candidates and find bases for Add instructions.
  void allocateCandidatesAndFindBasisForAdd(Instruction *I);

  // Given I = LHS + RHS, factors RHS into i * S and makes (LHS + i * S) a
  // candidate.
  void allocateCandidatesAndFindBasisForAdd(Value *LHS, Value *RHS,
                                            Instruction *I);
  // Allocate candidates and find bases for Mul instructions.
  void allocateCandidatesAndFindBasisForMul(Instruction *I);

  // Splits LHS into Base + Index and, if succeeds, calls
  // allocateCandidatesAndFindBasis.
  void allocateCandidatesAndFindBasisForMul(Value *LHS, Value *RHS,
                                            Instruction *I);

  // Allocate candidates and find bases for GetElementPtr instructions.
  void allocateCandidatesAndFindBasisForGEP(GetElementPtrInst *GEP);

  // Adds the given form <CT, B, Idx, S> to Candidates, and finds its immediate
  // basis.
  void allocateCandidatesAndFindBasis(Candidate::Kind CT, const SCEV *B,
                                      ConstantInt *Idx, Value *S,
                                      Instruction *I);

  // Rewrites candidate C with respect to Basis.
  void rewriteCandidate(const Candidate &C);

  // Emit code that computes the "bump" from Basis to C.
  static Value *emitBump(const Candidate &Basis, const Candidate &C,
                         IRBuilder<> &Builder, const DataLayout *DL);

  const DataLayout *DL = nullptr;
  DominatorTree *DT = nullptr;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI = nullptr;
  std::list<Candidate> Candidates;

  // Map from SCEV to instructions that represent the value,
  // instructions are sorted in depth-first order.
  DenseMap<const SCEV *, SmallSetVector<Instruction *, 2>> SCEVToInsts;

  // Record the dependency between instructions. If C.Basis == B, we would have
  // {B.Ins -> {C.Ins, ...}}.
  MapVector<Instruction *, std::vector<Instruction *>> DependencyGraph;

  // Map between each instruction and its possible candidates.
  DenseMap<Instruction *, SmallVector<Candidate *, 3>> RewriteCandidates;

  // All instructions that have candidates sort in topological order based on
  // dependency graph, from roots to leaves.
  std::vector<Instruction *> SortedCandidateInsts;

  // Record all instructions that are already rewritten and will be removed
  // later.
  std::vector<Instruction *> DeadInstructions;

  // Classify candidates against Delta kind
  class CandidateDictTy {
  public:
    using CandsTy = SmallVector<Candidate *, 8>;
    using BBToCandsTy = DenseMap<const BasicBlock *, CandsTy>;

  private:
    // Index delta Basis must have the same (Base, StrideSCEV, Inst.Type)
    using IndexDeltaKeyTy = std::tuple<const SCEV *, const SCEV *, Type *>;
    DenseMap<IndexDeltaKeyTy, BBToCandsTy> IndexDeltaCandidates;

    // Base delta Basis must have the same (StrideSCEV, Index, Inst.Type)
    using BaseDeltaKeyTy = std::tuple<const SCEV *, ConstantInt *, Type *>;
    DenseMap<BaseDeltaKeyTy, BBToCandsTy> BaseDeltaCandidates;

    // Stride delta Basis must have the same (Base, Index, Inst.Type)
    using StrideDeltaKeyTy = std::tuple<const SCEV *, ConstantInt *, Type *>;
    DenseMap<StrideDeltaKeyTy, BBToCandsTy> StrideDeltaCandidates;

  public:
    // TODO: Disable index delta on GEP after we completely move
    // from typed GEP to PtrAdd.
    const BBToCandsTy *getCandidatesWithDeltaKind(const Candidate &C,
                                                  Candidate::DKind K) const {
      assert(K != Candidate::InvalidDelta);
      if (K == Candidate::IndexDelta) {
        IndexDeltaKeyTy IndexDeltaKey(C.Base, C.StrideSCEV, C.Ins->getType());
        auto It = IndexDeltaCandidates.find(IndexDeltaKey);
        if (It != IndexDeltaCandidates.end())
          return &It->second;
      } else if (K == Candidate::BaseDelta) {
        BaseDeltaKeyTy BaseDeltaKey(C.StrideSCEV, C.Index, C.Ins->getType());
        auto It = BaseDeltaCandidates.find(BaseDeltaKey);
        if (It != BaseDeltaCandidates.end())
          return &It->second;
      } else {
        assert(K == Candidate::StrideDelta);
        StrideDeltaKeyTy StrideDeltaKey(C.Base, C.Index, C.Ins->getType());
        auto It = StrideDeltaCandidates.find(StrideDeltaKey);
        if (It != StrideDeltaCandidates.end())
          return &It->second;
      }
      return nullptr;
    }

    // Pointers to C must remain valid until CandidateDict is cleared.
    void add(Candidate &C) {
      Type *ValueType = C.Ins->getType();
      BasicBlock *BB = C.Ins->getParent();
      IndexDeltaKeyTy IndexDeltaKey(C.Base, C.StrideSCEV, ValueType);
      BaseDeltaKeyTy BaseDeltaKey(C.StrideSCEV, C.Index, ValueType);
      StrideDeltaKeyTy StrideDeltaKey(C.Base, C.Index, ValueType);
      IndexDeltaCandidates[IndexDeltaKey][BB].push_back(&C);
      BaseDeltaCandidates[BaseDeltaKey][BB].push_back(&C);
      StrideDeltaCandidates[StrideDeltaKey][BB].push_back(&C);
    }
    // Remove all mappings from set
    void clear() {
      IndexDeltaCandidates.clear();
      BaseDeltaCandidates.clear();
      StrideDeltaCandidates.clear();
    }
  } CandidateDict;

  const SCEV *getAndRecordSCEV(Value *V) {
    auto *S = SE->getSCEV(V);
    if (isa<Instruction>(V) && !(isa<SCEVCouldNotCompute>(S) ||
                                 isa<SCEVUnknown>(S) || isa<SCEVConstant>(S)))
      SCEVToInsts[S].insert(cast<Instruction>(V));

    return S;
  }

  // Get the nearest instruction before CI that represents the value of S,
  // return nullptr if no instruction is associated with S or S is not a
  // reusable expression.
  Value *getNearestValueOfSCEV(const SCEV *S, const Instruction *CI) const {
    if (isa<SCEVCouldNotCompute>(S))
      return nullptr;

    if (auto *SU = dyn_cast<SCEVUnknown>(S))
      return SU->getValue();
    if (auto *SC = dyn_cast<SCEVConstant>(S))
      return SC->getValue();

    auto It = SCEVToInsts.find(S);
    if (It == SCEVToInsts.end())
      return nullptr;

    // Instructions are sorted in depth-first order, so search for the nearest
    // instruction by walking the list in reverse order.
    for (Instruction *I : reverse(It->second))
      if (DT->dominates(I, CI))
        return I;

    return nullptr;
  }

  struct DeltaInfo {
    Candidate *Cand;
    Candidate::DKind DeltaKind;
    Value *Delta;

    DeltaInfo()
        : Cand(nullptr), DeltaKind(Candidate::InvalidDelta), Delta(nullptr) {}
    DeltaInfo(Candidate *Cand, Candidate::DKind DeltaKind, Value *Delta)
        : Cand(Cand), DeltaKind(DeltaKind), Delta(Delta) {}
    operator bool() const { return Cand != nullptr; }
  };

  friend raw_ostream &operator<<(raw_ostream &OS, const DeltaInfo &DI);

  DeltaInfo compressPath(Candidate &C, Candidate *Basis) const;

  Candidate *pickRewriteCandidate(Instruction *I) const;
  void sortCandidateInstructions();
  static Constant *getIndexDelta(Candidate &C, Candidate &Basis);
  static bool isSimilar(Candidate &C, Candidate &Basis, Candidate::DKind K);

  // Add Basis -> C in DependencyGraph and propagate
  // C.Stride and C.Delta's dependency to C
  void addDependency(Candidate &C, Candidate *Basis) {
    if (Basis)
      DependencyGraph[Basis->Ins].emplace_back(C.Ins);

    // If any candidate of Inst has a basis, then Inst will be rewritten,
    // C must be rewritten after rewriting Inst, so we need to propagate
    // the dependency to C
    auto PropagateDependency = [&](Instruction *Inst) {
      if (auto CandsIt = RewriteCandidates.find(Inst);
          CandsIt != RewriteCandidates.end() &&
          llvm::any_of(CandsIt->second,
                       [](Candidate *Cand) { return Cand->Basis; }))
        DependencyGraph[Inst].emplace_back(C.Ins);
    };

    // If C has a variable delta and the delta is a candidate,
    // propagate its dependency to C
    if (auto *DeltaInst = dyn_cast_or_null<Instruction>(C.Delta))
      PropagateDependency(DeltaInst);

    // If the stride is a candidate, propagate its dependency to C
    if (auto *StrideInst = dyn_cast<Instruction>(C.Stride))
      PropagateDependency(StrideInst);
  };
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const StraightLineStrengthReduce::Candidate &C) {
  OS << "Ins: " << *C.Ins << "\n  Base: " << *C.Base
     << "\n  Index: " << *C.Index << "\n  Stride: " << *C.Stride
     << "\n  StrideSCEV: " << *C.StrideSCEV;
  if (C.Basis)
    OS << "\n  Delta: " << *C.Delta << "\n  Basis: \n  [ " << *C.Basis << " ]";
  return OS;
}

[[maybe_unused]] LLVM_DUMP_METHOD inline raw_ostream &
operator<<(raw_ostream &OS, const StraightLineStrengthReduce::DeltaInfo &DI) {
  OS << "Cand: " << *DI.Cand << "\n";
  OS << "Delta Kind: ";
  switch (DI.DeltaKind) {
  case StraightLineStrengthReduce::Candidate::IndexDelta:
    OS << "Index";
    break;
  case StraightLineStrengthReduce::Candidate::BaseDelta:
    OS << "Base";
    break;
  case StraightLineStrengthReduce::Candidate::StrideDelta:
    OS << "Stride";
    break;
  default:
    break;
  }
  OS << "\nDelta: " << *DI.Delta;
  return OS;
}

} // end anonymous namespace

char StraightLineStrengthReduceLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(StraightLineStrengthReduceLegacyPass, "slsr",
                      "Straight line strength reduction", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(StraightLineStrengthReduceLegacyPass, "slsr",
                    "Straight line strength reduction", false, false)

FunctionPass *llvm::createStraightLineStrengthReducePass() {
  return new StraightLineStrengthReduceLegacyPass();
}

// A helper function that unifies the bitwidth of A and B.
static void unifyBitWidth(APInt &A, APInt &B) {
  if (A.getBitWidth() < B.getBitWidth())
    A = A.sext(B.getBitWidth());
  else if (A.getBitWidth() > B.getBitWidth())
    B = B.sext(A.getBitWidth());
}

Constant *StraightLineStrengthReduce::getIndexDelta(Candidate &C,
                                                    Candidate &Basis) {
  APInt Idx = C.Index->getValue(), BasisIdx = Basis.Index->getValue();
  unifyBitWidth(Idx, BasisIdx);
  APInt IndexDelta = Idx - BasisIdx;
  IntegerType *DeltaType =
      IntegerType::get(C.Ins->getContext(), IndexDelta.getBitWidth());
  return ConstantInt::get(DeltaType, IndexDelta);
}

bool StraightLineStrengthReduce::isSimilar(Candidate &C, Candidate &Basis,
                                           Candidate::DKind K) {
  bool SameType = false;
  switch (K) {
  case Candidate::StrideDelta:
    SameType = C.StrideSCEV->getType() == Basis.StrideSCEV->getType();
    break;
  case Candidate::BaseDelta:
    SameType = C.Base->getType() == Basis.Base->getType();
    break;
  case Candidate::IndexDelta:
    SameType = true;
    break;
  default:;
  }
  return SameType && Basis.Ins != C.Ins &&
         Basis.CandidateKind == C.CandidateKind;
}

void StraightLineStrengthReduce::setBasisAndDeltaFor(Candidate &C) {
  auto SearchFrom = [this, &C](const CandidateDictTy::BBToCandsTy &BBToCands,
                               auto IsTarget) -> bool {
    // Search dominating candidates by walking the immediate-dominator chain
    // from the candidate's defining block upward. Visiting blocks in this
    // order ensures we prefer the closest dominating basis.
    const BasicBlock *BB = C.Ins->getParent();
    while (BB) {
      auto It = BBToCands.find(BB);
      if (It != BBToCands.end())
        for (Candidate *Basis : reverse(It->second))
          if (IsTarget(Basis))
            return true;

      const DomTreeNode *Node = DT->getNode(BB);
      if (!Node)
        break;
      Node = Node->getIDom();
      BB = Node ? Node->getBlock() : nullptr;
    }
    return false;
  };

  // Priority:
  // Constant Delta from Index > Constant Delta from Base >
  // Constant Delta from Stride > Variable Delta from Base or Stride
  // TODO: Change the priority to align with the cost model.

  // First, look for a constant index-diff basis
  if (const auto *IndexDeltaCandidates =
          CandidateDict.getCandidatesWithDeltaKind(C, Candidate::IndexDelta)) {
    bool FoundConstDelta =
        SearchFrom(*IndexDeltaCandidates, [&](Candidate *Basis) {
          if (isSimilar(C, *Basis, Candidate::IndexDelta)) {
            assert(DT->dominates(Basis->Ins, C.Ins));
            auto *Delta = getIndexDelta(C, *Basis);
            if (!C.isProfitableRewrite(Delta, Candidate::IndexDelta))
              return false;
            C.Basis = Basis;
            C.DeltaKind = Candidate::IndexDelta;
            C.Delta = Delta;
            LLVM_DEBUG(dbgs() << "Found delta from Index " << *C.Delta << "\n");
            return true;
          }
          return false;
        });
    if (FoundConstDelta)
      return;
  }

  // No constant-index-diff basis found. look for the best possible base-diff
  // or stride-diff basis
  // Base/Stride diffs not supported for form (B + i) * S
  if (C.CandidateKind == Candidate::Mul)
    return;

  auto For = [this, &C](Candidate::DKind K) {
    // return true if find a Basis with constant delta and stop searching,
    // return false if did not find a Basis or the delta is not a constant
    // and continue searching for a Basis with constant delta
    return [K, this, &C](Candidate *Basis) -> bool {
      if (!isSimilar(C, *Basis, K))
        return false;

      assert(DT->dominates(Basis->Ins, C.Ins));
      const SCEV *BasisPart =
          (K == Candidate::BaseDelta) ? Basis->Base : Basis->StrideSCEV;
      const SCEV *CandPart =
          (K == Candidate::BaseDelta) ? C.Base : C.StrideSCEV;
      const SCEV *Diff = SE->getMinusSCEV(CandPart, BasisPart);
      Value *AvailableVal = getNearestValueOfSCEV(Diff, C.Ins);
      if (!AvailableVal)
        return false;

      // Record delta if none has been found yet, or the new delta is
      // a constant that is better than the existing delta.
      if (!C.Delta || isa<ConstantInt>(AvailableVal)) {
        C.Delta = AvailableVal;
        C.Basis = Basis;
        C.DeltaKind = K;
      }
      return isa<ConstantInt>(C.Delta);
    };
  };

  if (const auto *BaseDeltaCandidates =
          CandidateDict.getCandidatesWithDeltaKind(C, Candidate::BaseDelta)) {
    if (SearchFrom(*BaseDeltaCandidates, For(Candidate::BaseDelta))) {
      LLVM_DEBUG(dbgs() << "Found delta from Base: " << *C.Delta << "\n");
      return;
    }
  }

  if (const auto *StrideDeltaCandidates =
          CandidateDict.getCandidatesWithDeltaKind(C, Candidate::StrideDelta)) {
    if (SearchFrom(*StrideDeltaCandidates, For(Candidate::StrideDelta))) {
      LLVM_DEBUG(dbgs() << "Found delta from Stride: " << *C.Delta << "\n");
      return;
    }
  }

  // If we did not find a constant delta, we might have found a variable delta
  if (C.Delta) {
    LLVM_DEBUG({
      dbgs() << "Found delta from ";
      if (C.DeltaKind == Candidate::BaseDelta)
        dbgs() << "Base: ";
      else
        dbgs() << "Stride: ";
      dbgs() << *C.Delta << "\n";
    });
    assert(C.DeltaKind != Candidate::InvalidDelta && C.Basis);
  }
}

// Compress the path from `Basis` to the deepest Basis in the Basis chain
// to avoid non-profitable data dependency and improve ILP.
// X = A + 1
// Y = X + 1
// Z = Y + 1
// ->
// X = A + 1
// Y = A + 2
// Z = A + 3
// Return the delta info for C aginst the new Basis
auto StraightLineStrengthReduce::compressPath(Candidate &C,
                                              Candidate *Basis) const
    -> DeltaInfo {
  if (!Basis || !Basis->Basis || C.CandidateKind == Candidate::Mul)
    return {};
  Candidate *Root = Basis;
  Value *NewDelta = nullptr;
  auto NewKind = Candidate::InvalidDelta;

  while (Root->Basis) {
    Candidate *NextRoot = Root->Basis;
    if (C.Base == NextRoot->Base && C.StrideSCEV == NextRoot->StrideSCEV &&
        isSimilar(C, *NextRoot, Candidate::IndexDelta)) {
      ConstantInt *CI = cast<ConstantInt>(getIndexDelta(C, *NextRoot));
      if (CI->isZero() || CI->isOne() || isa<SCEVConstant>(C.StrideSCEV)) {
        Root = NextRoot;
        NewKind = Candidate::IndexDelta;
        NewDelta = CI;
        continue;
      }
    }

    const SCEV *CandPart = nullptr;
    const SCEV *BasisPart = nullptr;
    auto CurrKind = Candidate::InvalidDelta;
    if (C.Base == NextRoot->Base && C.Index == NextRoot->Index) {
      CandPart = C.StrideSCEV;
      BasisPart = NextRoot->StrideSCEV;
      CurrKind = Candidate::StrideDelta;
    } else if (C.StrideSCEV == NextRoot->StrideSCEV &&
               C.Index == NextRoot->Index) {
      CandPart = C.Base;
      BasisPart = NextRoot->Base;
      CurrKind = Candidate::BaseDelta;
    } else
      break;

    assert(CandPart && BasisPart);
    if (!isSimilar(C, *NextRoot, CurrKind))
      break;

    if (auto DeltaVal =
            dyn_cast<SCEVConstant>(SE->getMinusSCEV(CandPart, BasisPart))) {
      Root = NextRoot;
      NewDelta = DeltaVal->getValue();
      NewKind = CurrKind;
    } else
      break;
  }

  if (Root != Basis) {
    assert(NewKind != Candidate::InvalidDelta && NewDelta);
    LLVM_DEBUG(dbgs() << "Found new Basis with " << *NewDelta
                      << " from path compression.\n");
    return {Root, NewKind, NewDelta};
  }

  return {};
}

// Topologically sort candidate instructions based on their relationship in
// dependency graph.
void StraightLineStrengthReduce::sortCandidateInstructions() {
  SortedCandidateInsts.clear();
  // An instruction may have multiple candidates that get different Basis
  // instructions, and each candidate can get dependencies from Basis and
  // Stride when Stride will also be rewritten by SLSR. Hence, an instruction
  // may have multiple dependencies. Use InDegree to ensure all dependencies
  // processed before processing itself.
  DenseMap<Instruction *, int> InDegree;
  for (auto &KV : DependencyGraph) {
    InDegree.try_emplace(KV.first, 0);

    for (auto *Child : KV.second) {
      InDegree[Child]++;
    }
  }
  std::queue<Instruction *> WorkList;
  DenseSet<Instruction *> Visited;

  for (auto &KV : DependencyGraph)
    if (InDegree[KV.first] == 0)
      WorkList.push(KV.first);

  while (!WorkList.empty()) {
    Instruction *I = WorkList.front();
    WorkList.pop();
    if (!Visited.insert(I).second)
      continue;

    SortedCandidateInsts.push_back(I);

    for (auto *Next : DependencyGraph[I]) {
      auto &Degree = InDegree[Next];
      if (--Degree == 0)
        WorkList.push(Next);
    }
  }

  assert(SortedCandidateInsts.size() == DependencyGraph.size() &&
         "Dependency graph should not have cycles");
}

auto StraightLineStrengthReduce::pickRewriteCandidate(Instruction *I) const
    -> Candidate * {
  // Return the candidate of instruction I that has the highest profit.
  auto It = RewriteCandidates.find(I);
  if (It == RewriteCandidates.end())
    return nullptr;

  Candidate *BestC = nullptr;
  auto BestEfficiency = Candidate::Unknown;
  for (Candidate *C : reverse(It->second))
    if (C->Basis) {
      auto Efficiency = C->getRewriteEfficiency();
      if (Efficiency > BestEfficiency) {
        BestEfficiency = Efficiency;
        BestC = C;
      }
    }

  return BestC;
}

static bool isGEPFoldable(GetElementPtrInst *GEP,
                          const TargetTransformInfo *TTI) {
  SmallVector<const Value *, 4> Indices(GEP->indices());
  return TTI->getGEPCost(GEP->getSourceElementType(), GEP->getPointerOperand(),
                         Indices) == TargetTransformInfo::TCC_Free;
}

// Returns whether (Base + Index * Stride) can be folded to an addressing mode.
static bool isAddFoldable(const SCEV *Base, ConstantInt *Index, Value *Stride,
                          TargetTransformInfo *TTI) {
  // Index->getSExtValue() may crash if Index is wider than 64-bit.
  return Index->getBitWidth() <= 64 &&
         TTI->isLegalAddressingMode(Base->getType(), nullptr, 0, true,
                                    Index->getSExtValue(), UnknownAddressSpace);
}

bool StraightLineStrengthReduce::isFoldable(const Candidate &C,
                                            TargetTransformInfo *TTI) {
  if (C.CandidateKind == Candidate::Add)
    return isAddFoldable(C.Base, C.Index, C.Stride, TTI);
  if (C.CandidateKind == Candidate::GEP)
    return isGEPFoldable(cast<GetElementPtrInst>(C.Ins), TTI);
  return false;
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasis(
    Candidate::Kind CT, const SCEV *B, ConstantInt *Idx, Value *S,
    Instruction *I) {
  // Record the SCEV of S that we may use it as a variable delta.
  // Ensure that we rewrite C with a existing IR that reproduces delta value.

  Candidate C(CT, B, Idx, S, I, getAndRecordSCEV(S));
  // If we can fold I into an addressing mode, computing I is likely free or
  // takes only one instruction. So, we don't need to analyze or rewrite it.
  //
  // Currently, this algorithm can at best optimize complex computations into
  // a `variable +/* constant` form. However, some targets have stricter
  // constraints on the their addressing mode.
  // For example, a `variable + constant` can only be folded to an addressing
  // mode if the constant falls within a certain range.
  // So, we also check if the instruction is already high efficient enough
  // for the strength reduction algorithm.
  if (!isFoldable(C, TTI) && !C.isHighEfficiency()) {
    setBasisAndDeltaFor(C);

    // Compress unnecessary rewrite to improve ILP
    if (auto Res = compressPath(C, C.Basis)) {
      C.Basis = Res.Cand;
      C.DeltaKind = Res.DeltaKind;
      C.Delta = Res.Delta;
    }
  }
  // Regardless of whether we find a basis for C, we need to push C to the
  // candidate list so that it can be the basis of other candidates.
  LLVM_DEBUG(dbgs() << "Allocated Candidate: " << C << "\n");
  Candidates.push_back(C);
  RewriteCandidates[C.Ins].push_back(&Candidates.back());
  CandidateDict.add(Candidates.back());
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasis(
    Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
    allocateCandidatesAndFindBasisForAdd(I);
    break;
  case Instruction::Mul:
    allocateCandidatesAndFindBasisForMul(I);
    break;
  case Instruction::GetElementPtr:
    allocateCandidatesAndFindBasisForGEP(cast<GetElementPtrInst>(I));
    break;
  }
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasisForAdd(
    Instruction *I) {
  // Try matching B + i * S.
  if (!isa<IntegerType>(I->getType()))
    return;

  assert(I->getNumOperands() == 2 && "isn't I an add?");
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  allocateCandidatesAndFindBasisForAdd(LHS, RHS, I);
  if (LHS != RHS)
    allocateCandidatesAndFindBasisForAdd(RHS, LHS, I);
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasisForAdd(
    Value *LHS, Value *RHS, Instruction *I) {
  Value *S = nullptr;
  ConstantInt *Idx = nullptr;
  if (match(RHS, m_Mul(m_Value(S), m_ConstantInt(Idx)))) {
    // I = LHS + RHS = LHS + Idx * S
    allocateCandidatesAndFindBasis(Candidate::Add, SE->getSCEV(LHS), Idx, S, I);
  } else if (match(RHS, m_Shl(m_Value(S), m_ConstantInt(Idx)))) {
    // I = LHS + RHS = LHS + (S << Idx) = LHS + S * (1 << Idx)
    APInt One(Idx->getBitWidth(), 1);
    Idx = ConstantInt::get(Idx->getContext(), One << Idx->getValue());
    allocateCandidatesAndFindBasis(Candidate::Add, SE->getSCEV(LHS), Idx, S, I);
  } else {
    // At least, I = LHS + 1 * RHS
    ConstantInt *One = ConstantInt::get(cast<IntegerType>(I->getType()), 1);
    allocateCandidatesAndFindBasis(Candidate::Add, SE->getSCEV(LHS), One, RHS,
                                   I);
  }
}

// Returns true if A matches B + C where C is constant.
static bool matchesAdd(Value *A, Value *&B, ConstantInt *&C) {
  return match(A, m_c_Add(m_Value(B), m_ConstantInt(C)));
}

// Returns true if A matches B | C where C is constant.
static bool matchesOr(Value *A, Value *&B, ConstantInt *&C) {
  return match(A, m_c_Or(m_Value(B), m_ConstantInt(C)));
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasisForMul(
    Value *LHS, Value *RHS, Instruction *I) {
  Value *B = nullptr;
  ConstantInt *Idx = nullptr;
  if (matchesAdd(LHS, B, Idx)) {
    // If LHS is in the form of "Base + Index", then I is in the form of
    // "(Base + Index) * RHS".
    allocateCandidatesAndFindBasis(Candidate::Mul, SE->getSCEV(B), Idx, RHS, I);
  } else if (matchesOr(LHS, B, Idx) && haveNoCommonBitsSet(B, Idx, *DL)) {
    // If LHS is in the form of "Base | Index" and Base and Index have no common
    // bits set, then
    //   Base | Index = Base + Index
    // and I is thus in the form of "(Base + Index) * RHS".
    allocateCandidatesAndFindBasis(Candidate::Mul, SE->getSCEV(B), Idx, RHS, I);
  } else {
    // Otherwise, at least try the form (LHS + 0) * RHS.
    ConstantInt *Zero = ConstantInt::get(cast<IntegerType>(I->getType()), 0);
    allocateCandidatesAndFindBasis(Candidate::Mul, SE->getSCEV(LHS), Zero, RHS,
                                   I);
  }
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasisForMul(
    Instruction *I) {
  // Try matching (B + i) * S.
  // TODO: we could extend SLSR to float and vector types.
  if (!isa<IntegerType>(I->getType()))
    return;

  assert(I->getNumOperands() == 2 && "isn't I a mul?");
  Value *LHS = I->getOperand(0), *RHS = I->getOperand(1);
  allocateCandidatesAndFindBasisForMul(LHS, RHS, I);
  if (LHS != RHS) {
    // Symmetrically, try to split RHS to Base + Index.
    allocateCandidatesAndFindBasisForMul(RHS, LHS, I);
  }
}

void StraightLineStrengthReduce::allocateCandidatesAndFindBasisForGEP(
    GetElementPtrInst *GEP) {
  // TODO: handle vector GEPs
  if (GEP->getType()->isVectorTy())
    return;

  SmallVector<const SCEV *, 4> IndexExprs;
  for (Use &Idx : GEP->indices())
    IndexExprs.push_back(SE->getSCEV(Idx));

  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned I = 1, E = GEP->getNumOperands(); I != E; ++I, ++GTI) {
    if (GTI.isStruct())
      continue;

    const SCEV *OrigIndexExpr = IndexExprs[I - 1];
    IndexExprs[I - 1] = SE->getZero(OrigIndexExpr->getType());

    // The base of this candidate is GEP's base plus the offsets of all
    // indices except this current one.
    const SCEV *BaseExpr = SE->getGEPExpr(cast<GEPOperator>(GEP), IndexExprs);
    Value *ArrayIdx = GEP->getOperand(I);
    uint64_t ElementSize = GTI.getSequentialElementStride(*DL);
    IntegerType *PtrIdxTy = cast<IntegerType>(DL->getIndexType(GEP->getType()));
    ConstantInt *ElementSizeIdx = ConstantInt::get(PtrIdxTy, ElementSize, true);
    if (ArrayIdx->getType()->getIntegerBitWidth() <=
        DL->getIndexSizeInBits(GEP->getAddressSpace())) {
      // Skip factoring if ArrayIdx is wider than the index size, because
      // ArrayIdx is implicitly truncated to the index size.
      allocateCandidatesAndFindBasis(Candidate::GEP, BaseExpr, ElementSizeIdx,
                                     ArrayIdx, GEP);
    }
    // When ArrayIdx is the sext of a value, we try to factor that value as
    // well.  Handling this case is important because array indices are
    // typically sign-extended to the pointer index size.
    Value *TruncatedArrayIdx = nullptr;
    if (match(ArrayIdx, m_SExt(m_Value(TruncatedArrayIdx))) &&
        TruncatedArrayIdx->getType()->getIntegerBitWidth() <=
            DL->getIndexSizeInBits(GEP->getAddressSpace())) {
      // Skip factoring if TruncatedArrayIdx is wider than the pointer size,
      // because TruncatedArrayIdx is implicitly truncated to the pointer size.
      allocateCandidatesAndFindBasis(Candidate::GEP, BaseExpr, ElementSizeIdx,
                                     TruncatedArrayIdx, GEP);
    }

    IndexExprs[I - 1] = OrigIndexExpr;
  }
}

Value *StraightLineStrengthReduce::emitBump(const Candidate &Basis,
                                            const Candidate &C,
                                            IRBuilder<> &Builder,
                                            const DataLayout *DL) {
  auto CreateMul = [&](Value *LHS, Value *RHS) {
    if (ConstantInt *CR = dyn_cast<ConstantInt>(RHS)) {
      const APInt &ConstRHS = CR->getValue();
      IntegerType *DeltaType =
          IntegerType::get(C.Ins->getContext(), ConstRHS.getBitWidth());
      if (ConstRHS.isPowerOf2()) {
        ConstantInt *Exponent =
            ConstantInt::get(DeltaType, ConstRHS.logBase2());
        return Builder.CreateShl(LHS, Exponent);
      }
      if (ConstRHS.isNegatedPowerOf2()) {
        ConstantInt *Exponent =
            ConstantInt::get(DeltaType, (-ConstRHS).logBase2());
        return Builder.CreateNeg(Builder.CreateShl(LHS, Exponent));
      }
    }

    return Builder.CreateMul(LHS, RHS);
  };

  Value *Delta = C.Delta;
  // If Delta is 0, C is a fully redundant of C.Basis,
  // just replace C.Ins with Basis.Ins
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Delta);
      CI && CI->getValue().isZero())
    return nullptr;

  if (C.DeltaKind == Candidate::IndexDelta) {
    APInt IndexDelta = cast<ConstantInt>(C.Delta)->getValue();
    // IndexDelta
    // X = B + i * S
    // Y = B + i` * S
    //   = B + (i + IndexDelta) * S
    //   = B + i * S + IndexDelta * S
    //   = X + IndexDelta * S
    // Bump = (i' - i) * S

    // Common case 1: if (i' - i) is 1, Bump = S.
    if (IndexDelta == 1)
      return C.Stride;
    // Common case 2: if (i' - i) is -1, Bump = -S.
    if (IndexDelta.isAllOnes())
      return Builder.CreateNeg(C.Stride);

    IntegerType *DeltaType =
        IntegerType::get(Basis.Ins->getContext(), IndexDelta.getBitWidth());
    Value *ExtendedStride = Builder.CreateSExtOrTrunc(C.Stride, DeltaType);

    return CreateMul(ExtendedStride, C.Delta);
  }

  assert(C.DeltaKind == Candidate::StrideDelta ||
         C.DeltaKind == Candidate::BaseDelta);
  assert(C.CandidateKind != Candidate::Mul);
  // StrideDelta
  // X = B + i * S
  // Y = B + i * S'
  //   = B + i * (S + StrideDelta)
  //   = B + i * S + i * StrideDelta
  //   = X + i * StrideDelta
  // Bump = i * (S' - S)
  //
  // BaseDelta
  // X = B  + i * S
  // Y = B' + i * S
  //   = (B + BaseDelta) + i * S
  //   = X + BaseDelta
  // Bump = (B' - B).
  Value *Bump = C.Delta;
  if (C.DeltaKind == Candidate::StrideDelta) {
    // If this value is consumed by a GEP, promote StrideDelta before doing
    // StrideDelta * Index to ensure the same semantics as the original GEP.
    if (C.CandidateKind == Candidate::GEP) {
      auto *GEP = cast<GetElementPtrInst>(C.Ins);
      Type *NewScalarIndexTy =
          DL->getIndexType(GEP->getPointerOperandType()->getScalarType());
      Bump = Builder.CreateSExtOrTrunc(Bump, NewScalarIndexTy);
    }
    if (!C.Index->isOne()) {
      Value *ExtendedIndex =
          Builder.CreateSExtOrTrunc(C.Index, Bump->getType());
      Bump = CreateMul(Bump, ExtendedIndex);
    }
  }
  return Bump;
}

void StraightLineStrengthReduce::rewriteCandidate(const Candidate &C) {
  if (!DebugCounter::shouldExecute(StraightLineStrengthReduceCounter))
    return;

  const Candidate &Basis = *C.Basis;
  assert(C.Delta && C.CandidateKind == Basis.CandidateKind &&
         C.hasValidDelta(Basis));

  IRBuilder<> Builder(C.Ins);
  Value *Bump = emitBump(Basis, C, Builder, DL);
  Value *Reduced = nullptr; // equivalent to but weaker than C.Ins
  // If delta is 0, C is a fully redundant of Basis, and Bump is nullptr,
  // just replace C.Ins with Basis.Ins
  if (!Bump)
    Reduced = Basis.Ins;
  else {
    switch (C.CandidateKind) {
    case Candidate::Add:
    case Candidate::Mul: {
      // C = Basis + Bump
      Value *NegBump;
      if (match(Bump, m_Neg(m_Value(NegBump)))) {
        // If Bump is a neg instruction, emit C = Basis - (-Bump).
        Reduced = Builder.CreateSub(Basis.Ins, NegBump);
        // We only use the negative argument of Bump, and Bump itself may be
        // trivially dead.
        RecursivelyDeleteTriviallyDeadInstructions(Bump);
      } else {
        // It's tempting to preserve nsw on Bump and/or Reduced. However, it's
        // usually unsound, e.g.,
        //
        // X = (-2 +nsw 1) *nsw INT_MAX
        // Y = (-2 +nsw 3) *nsw INT_MAX
        //   =>
        // Y = X + 2 * INT_MAX
        //
        // Neither + and * in the resultant expression are nsw.
        Reduced = Builder.CreateAdd(Basis.Ins, Bump);
      }
      break;
    }
    case Candidate::GEP: {
      bool InBounds = cast<GetElementPtrInst>(C.Ins)->isInBounds();
      // C = (char *)Basis + Bump
      Reduced = Builder.CreatePtrAdd(Basis.Ins, Bump, "", InBounds);
      break;
    }
    default:
      llvm_unreachable("C.CandidateKind is invalid");
    };
    Reduced->takeName(C.Ins);
  }
  C.Ins->replaceAllUsesWith(Reduced);
  DeadInstructions.push_back(C.Ins);
}

bool StraightLineStrengthReduceLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  return StraightLineStrengthReduce(DL, DT, SE, TTI).runOnFunction(F);
}

bool StraightLineStrengthReduce::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "SLSR on Function: " << F.getName() << "\n");
  // Traverse the dominator tree in the depth-first order. This order makes sure
  // all bases of a candidate are in Candidates when we process it.
  for (const auto Node : depth_first(DT))
    for (auto &I : *(Node->getBlock()))
      allocateCandidatesAndFindBasis(&I);

  // Build the dependency graph and sort candidate instructions from dependency
  // roots to leaves
  for (auto &C : Candidates) {
    DependencyGraph.try_emplace(C.Ins);
    addDependency(C, C.Basis);
  }
  sortCandidateInstructions();

  // Rewrite candidates in the topological order that rewrites a Candidate
  // always before rewriting its Basis
  for (Instruction *I : reverse(SortedCandidateInsts))
    if (Candidate *C = pickRewriteCandidate(I))
      rewriteCandidate(*C);

  for (auto *DeadIns : DeadInstructions)
    // A dead instruction may be another dead instruction's op,
    // don't delete an instruction twice
    if (DeadIns->getParent())
      RecursivelyDeleteTriviallyDeadInstructions(DeadIns);

  bool Ret = !DeadInstructions.empty();
  DeadInstructions.clear();
  DependencyGraph.clear();
  RewriteCandidates.clear();
  SortedCandidateInsts.clear();
  // First clear all references to candidates in the list
  CandidateDict.clear();
  // Then destroy the list
  Candidates.clear();
  return Ret;
}

PreservedAnalyses
StraightLineStrengthReducePass::run(Function &F, FunctionAnalysisManager &AM) {
  const DataLayout *DL = &F.getDataLayout();
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TTI = &AM.getResult<TargetIRAnalysis>(F);

  if (!StraightLineStrengthReduce(DL, DT, SE, TTI).runOnFunction(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<TargetIRAnalysis>();
  return PA;
}
