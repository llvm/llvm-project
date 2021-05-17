//===- LoopCacheAnalysis.cpp - Loop Cache Analysis -------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the implementation for the loop cache analysis.
/// The implementation is largely based on the following paper:
///
///       Compiler Optimizations for Improving Data Locality
///       By: Steve Carr, Katherine S. McKinley, Chau-Wen Tseng
///       http://www.cs.utexas.edu/users/mckinley/papers/asplos-1994.pdf
///
/// The general approach taken to estimate the number of cache lines used by the
/// memory references in an inner loop is:
///    1. Partition memory references that exhibit temporal or spacial reuse
///       into reference groups.
///    2. For each loop L in the a loop nest LN:
///       a. Compute the cost of the reference group
///       b. Compute the loop cost by summing up the reference groups costs
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopCacheAnalysis.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "loop-cache-cost"

static cl::opt<unsigned> DefaultTripCount(
    "default-trip-count", cl::init(100), cl::Hidden,
    cl::desc("Use this to specify the default trip count of a loop"));

// In this analysis two array references are considered to exhibit temporal
// reuse if they access either the same memory location, or a memory location
// with distance smaller than a configurable threshold.
static cl::opt<unsigned> TemporalReuseThreshold(
    "temporal-reuse-threshold", cl::init(2), cl::Hidden,
    cl::desc("Use this to specify the max. distance between array elements "
             "accessed in a loop so that the elements are classified to have "
             "temporal reuse"));

/// Retrieve the innermost loop in the given loop nest \p Loops. It returns a
/// nullptr if any loops in the loop vector supplied has more than one sibling.
/// The loop vector is expected to contain loops collected in breadth-first
/// order.
static Loop *getInnerMostLoop(const LoopVectorTy &Loops) {
  assert(!Loops.empty() && "Expecting a non-empy loop vector");

  Loop *LastLoop = Loops.back();
  Loop *ParentLoop = LastLoop->getParentLoop();

  if (ParentLoop == nullptr) {
    assert(Loops.size() == 1 && "Expecting a single loop");
    return LastLoop;
  }

  return (llvm::is_sorted(Loops,
                          [](const Loop *L1, const Loop *L2) {
                            return L1->getLoopDepth() < L2->getLoopDepth();
                          }))
             ? LastLoop
             : nullptr;
}

static bool isOneDimensionalArray(const SCEV &AccessFn, const SCEV &ElemSize,
                                  const Loop &L, ScalarEvolution &SE) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(&AccessFn);
  if (!AR || !AR->isAffine())
    return false;

  assert(AR->getLoop() && "AR should have a loop");

  // Check that start and increment are not add recurrences.
  const SCEV *Start = AR->getStart();
  const SCEV *Step = AR->getStepRecurrence(SE);
  if (isa<SCEVAddRecExpr>(Start) || isa<SCEVAddRecExpr>(Step))
    return false;

  // Check that start and increment are both invariant in the loop.
  if (!SE.isLoopInvariant(Start, &L) || !SE.isLoopInvariant(Step, &L))
    return false;

  const SCEV *StepRec = AR->getStepRecurrence(SE);
  if (StepRec && SE.isKnownNegative(StepRec))
    StepRec = SE.getNegativeSCEV(StepRec);

  return StepRec == &ElemSize;
}

/// Compute the trip count for the given loop \p L. Return the SCEV expression
/// for the trip count or nullptr if it cannot be computed.
static const SCEV *computeTripCount(const Loop &L, ScalarEvolution &SE) {
  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(&L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount) ||
      !isa<SCEVConstant>(BackedgeTakenCount))
    return nullptr;

  return SE.getAddExpr(BackedgeTakenCount,
                       SE.getOne(BackedgeTakenCount->getType()));
}

//===----------------------------------------------------------------------===//
// IndexedReference implementation
//
raw_ostream &llvm::operator<<(raw_ostream &OS, const IndexedReference &R) {
  if (!R.IsValid) {
    OS << R.StoreOrLoadInst;
    OS << ", IsValid=false.";
    return OS;
  }

  OS << *R.BasePointer;
  for (const SCEV *Subscript : R.Subscripts)
    OS << "[" << *Subscript << "]";

  OS << ", Sizes: ";
  for (const SCEV *Size : R.Sizes)
    OS << "[" << *Size << "]";

  return OS;
}

IndexedReference::IndexedReference(Instruction &StoreOrLoadInst,
                                   const LoopInfo &LI, ScalarEvolution &SE)
    : StoreOrLoadInst(StoreOrLoadInst), SE(SE) {
  assert((isa<StoreInst>(StoreOrLoadInst) || isa<LoadInst>(StoreOrLoadInst)) &&
         "Expecting a load or store instruction");

  IsValid = delinearize(LI);
  if (IsValid)
    LLVM_DEBUG(dbgs().indent(2) << "Succesfully delinearized: " << *this
                                << "\n");
}

Optional<bool> IndexedReference::hasSpacialReuse(const IndexedReference &Other,
                                                 unsigned CLS,
                                                 AAResults &AA) const {
  assert(IsValid && "Expecting a valid reference");

  if (BasePointer != Other.getBasePointer() && !isAliased(Other, AA)) {
    LLVM_DEBUG(dbgs().indent(2)
               << "No spacial reuse: different base pointers\n");
    return false;
  }

  unsigned NumSubscripts = getNumSubscripts();
  if (NumSubscripts != Other.getNumSubscripts()) {
    LLVM_DEBUG(dbgs().indent(2)
               << "No spacial reuse: different number of subscripts\n");
    return false;
  }

  // all subscripts must be equal, except the leftmost one (the last one).
  for (auto SubNum : seq<unsigned>(0, NumSubscripts - 1)) {
    if (getSubscript(SubNum) != Other.getSubscript(SubNum)) {
      LLVM_DEBUG(dbgs().indent(2) << "No spacial reuse, different subscripts: "
                                  << "\n\t" << *getSubscript(SubNum) << "\n\t"
                                  << *Other.getSubscript(SubNum) << "\n");
      return false;
    }
  }

  // the difference between the last subscripts must be less than the cache line
  // size.
  const SCEV *LastSubscript = getLastSubscript();
  const SCEV *OtherLastSubscript = Other.getLastSubscript();
  const SCEVConstant *Diff = dyn_cast<SCEVConstant>(
      SE.getMinusSCEV(LastSubscript, OtherLastSubscript));

  if (Diff == nullptr) {
    LLVM_DEBUG(dbgs().indent(2)
               << "No spacial reuse, difference between subscript:\n\t"
               << *LastSubscript << "\n\t" << OtherLastSubscript
               << "\nis not constant.\n");
    return None;
  }

  bool InSameCacheLine = (Diff->getValue()->getSExtValue() < CLS);

  LLVM_DEBUG({
    if (InSameCacheLine)
      dbgs().indent(2) << "Found spacial reuse.\n";
    else
      dbgs().indent(2) << "No spacial reuse.\n";
  });

  return InSameCacheLine;
}

Optional<bool> IndexedReference::hasTemporalReuse(const IndexedReference &Other,
                                                  unsigned MaxDistance,
                                                  const Loop &L,
                                                  DependenceInfo &DI,
                                                  AAResults &AA) const {
  assert(IsValid && "Expecting a valid reference");

  if (BasePointer != Other.getBasePointer() && !isAliased(Other, AA)) {
    LLVM_DEBUG(dbgs().indent(2)
               << "No temporal reuse: different base pointer\n");
    return false;
  }

  std::unique_ptr<Dependence> D =
      DI.depends(&StoreOrLoadInst, &Other.StoreOrLoadInst, true);

  if (D == nullptr) {
    LLVM_DEBUG(dbgs().indent(2) << "No temporal reuse: no dependence\n");
    return false;
  }

  if (D->isLoopIndependent()) {
    LLVM_DEBUG(dbgs().indent(2) << "Found temporal reuse\n");
    return true;
  }

  // Check the dependence distance at every loop level. There is temporal reuse
  // if the distance at the given loop's depth is small (|d| <= MaxDistance) and
  // it is zero at every other loop level.
  int LoopDepth = L.getLoopDepth();
  int Levels = D->getLevels();
  for (int Level = 1; Level <= Levels; ++Level) {
    const SCEV *Distance = D->getDistance(Level);
    const SCEVConstant *SCEVConst = dyn_cast_or_null<SCEVConstant>(Distance);

    if (SCEVConst == nullptr) {
      LLVM_DEBUG(dbgs().indent(2) << "No temporal reuse: distance unknown\n");
      return None;
    }

    const ConstantInt &CI = *SCEVConst->getValue();
    if (Level != LoopDepth && !CI.isZero()) {
      LLVM_DEBUG(dbgs().indent(2)
                 << "No temporal reuse: distance is not zero at depth=" << Level
                 << "\n");
      return false;
    } else if (Level == LoopDepth && CI.getSExtValue() > MaxDistance) {
      LLVM_DEBUG(
          dbgs().indent(2)
          << "No temporal reuse: distance is greater than MaxDistance at depth="
          << Level << "\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs().indent(2) << "Found temporal reuse\n");
  return true;
}

CacheCostTy IndexedReference::computeRefCost(const Loop &L,
                                             unsigned CLS) const {
  assert(IsValid && "Expecting a valid reference");
  LLVM_DEBUG({
    dbgs().indent(2) << "Computing cache cost for:\n";
    dbgs().indent(4) << *this << "\n";
  });

  // If the indexed reference is loop invariant the cost is one.
  if (isLoopInvariant(L)) {
    LLVM_DEBUG(dbgs().indent(4) << "Reference is loop invariant: RefCost=1\n");
    return 1;
  }

  const SCEV *TripCount = computeTripCount(L, SE);
  if (!TripCount) {
    LLVM_DEBUG(dbgs() << "Trip count of loop " << L.getName()
                      << " could not be computed, using DefaultTripCount\n");
    const SCEV *ElemSize = Sizes.back();
    TripCount = SE.getConstant(ElemSize->getType(), DefaultTripCount);
  }
  LLVM_DEBUG(dbgs() << "TripCount=" << *TripCount << "\n");

  // If the indexed reference is 'consecutive' the cost is
  // (TripCount*Stride)/CLS, otherwise the cost is TripCount.
  const SCEV *RefCost = TripCount;

  if (isConsecutive(L, CLS)) {
    const SCEV *Coeff = getLastCoefficient();
    const SCEV *ElemSize = Sizes.back();
    const SCEV *Stride = SE.getMulExpr(Coeff, ElemSize);
    const SCEV *CacheLineSize = SE.getConstant(Stride->getType(), CLS);
    Type *WiderType = SE.getWiderType(Stride->getType(), TripCount->getType());
    if (SE.isKnownNegative(Stride))
      Stride = SE.getNegativeSCEV(Stride);
    Stride = SE.getNoopOrAnyExtend(Stride, WiderType);
    TripCount = SE.getNoopOrAnyExtend(TripCount, WiderType);
    const SCEV *Numerator = SE.getMulExpr(Stride, TripCount);
    RefCost = SE.getUDivExpr(Numerator, CacheLineSize);

    LLVM_DEBUG(dbgs().indent(4)
               << "Access is consecutive: RefCost=(TripCount*Stride)/CLS="
               << *RefCost << "\n");
  } else
    LLVM_DEBUG(dbgs().indent(4)
               << "Access is not consecutive: RefCost=TripCount=" << *RefCost
               << "\n");

  // Attempt to fold RefCost into a constant.
  if (auto ConstantCost = dyn_cast<SCEVConstant>(RefCost))
    return ConstantCost->getValue()->getSExtValue();

  LLVM_DEBUG(dbgs().indent(4)
             << "RefCost is not a constant! Setting to RefCost=InvalidCost "
                "(invalid value).\n");

  return CacheCost::InvalidCost;
}

bool IndexedReference::delinearize(const LoopInfo &LI) {
  assert(Subscripts.empty() && "Subscripts should be empty");
  assert(Sizes.empty() && "Sizes should be empty");
  assert(!IsValid && "Should be called once from the constructor");
  LLVM_DEBUG(dbgs() << "Delinearizing: " << StoreOrLoadInst << "\n");

  const SCEV *ElemSize = SE.getElementSize(&StoreOrLoadInst);
  const BasicBlock *BB = StoreOrLoadInst.getParent();

  if (Loop *L = LI.getLoopFor(BB)) {
    const SCEV *AccessFn =
        SE.getSCEVAtScope(getPointerOperand(&StoreOrLoadInst), L);

    BasePointer = dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFn));
    if (BasePointer == nullptr) {
      LLVM_DEBUG(
          dbgs().indent(2)
          << "ERROR: failed to delinearize, can't identify base pointer\n");
      return false;
    }

    AccessFn = SE.getMinusSCEV(AccessFn, BasePointer);

    LLVM_DEBUG(dbgs().indent(2) << "In Loop '" << L->getName()
                                << "', AccessFn: " << *AccessFn << "\n");

    SE.delinearize(AccessFn, Subscripts, Sizes,
                   SE.getElementSize(&StoreOrLoadInst));

    if (Subscripts.empty() || Sizes.empty() ||
        Subscripts.size() != Sizes.size()) {
      // Attempt to determine whether we have a single dimensional array access.
      // before giving up.
      if (!isOneDimensionalArray(*AccessFn, *ElemSize, *L, SE)) {
        LLVM_DEBUG(dbgs().indent(2)
                   << "ERROR: failed to delinearize reference\n");
        Subscripts.clear();
        Sizes.clear();
        return false;
      }

      // The array may be accessed in reverse, for example:
      //   for (i = N; i > 0; i--)
      //     A[i] = 0;
      // In this case, reconstruct the access function using the absolute value
      // of the step recurrence.
      const SCEVAddRecExpr *AccessFnAR = dyn_cast<SCEVAddRecExpr>(AccessFn);
      const SCEV *StepRec = AccessFnAR ? AccessFnAR->getStepRecurrence(SE) : nullptr;

      if (StepRec && SE.isKnownNegative(StepRec))
        AccessFn = SE.getAddRecExpr(AccessFnAR->getStart(),
                                    SE.getNegativeSCEV(StepRec),
                                    AccessFnAR->getLoop(),
                                    AccessFnAR->getNoWrapFlags());
      const SCEV *Div = SE.getUDivExactExpr(AccessFn, ElemSize);
      Subscripts.push_back(Div);
      Sizes.push_back(ElemSize);
    }

    return all_of(Subscripts, [&](const SCEV *Subscript) {
      return isSimpleAddRecurrence(*Subscript, *L);
    });
  }

  return false;
}

bool IndexedReference::isLoopInvariant(const Loop &L) const {
  Value *Addr = getPointerOperand(&StoreOrLoadInst);
  assert(Addr != nullptr && "Expecting either a load or a store instruction");
  assert(SE.isSCEVable(Addr->getType()) && "Addr should be SCEVable");

  if (SE.isLoopInvariant(SE.getSCEV(Addr), &L))
    return true;

  // The indexed reference is loop invariant if none of the coefficients use
  // the loop induction variable.
  bool allCoeffForLoopAreZero = all_of(Subscripts, [&](const SCEV *Subscript) {
    return isCoeffForLoopZeroOrInvariant(*Subscript, L);
  });

  return allCoeffForLoopAreZero;
}

bool IndexedReference::isConsecutive(const Loop &L, unsigned CLS) const {
  // The indexed reference is 'consecutive' if the only coefficient that uses
  // the loop induction variable is the last one...
  const SCEV *LastSubscript = Subscripts.back();
  for (const SCEV *Subscript : Subscripts) {
    if (Subscript == LastSubscript)
      continue;
    if (!isCoeffForLoopZeroOrInvariant(*Subscript, L))
      return false;
  }

  // ...and the access stride is less than the cache line size.
  const SCEV *Coeff = getLastCoefficient();
  const SCEV *ElemSize = Sizes.back();
  const SCEV *Stride = SE.getMulExpr(Coeff, ElemSize);
  const SCEV *CacheLineSize = SE.getConstant(Stride->getType(), CLS);

  Stride = SE.isKnownNegative(Stride) ? SE.getNegativeSCEV(Stride) : Stride;
  return SE.isKnownPredicate(ICmpInst::ICMP_ULT, Stride, CacheLineSize);
}

const SCEV *IndexedReference::getLastCoefficient() const {
  const SCEV *LastSubscript = getLastSubscript();
  assert(isa<SCEVAddRecExpr>(LastSubscript) &&
         "Expecting a SCEV add recurrence expression");
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LastSubscript);
  return AR->getStepRecurrence(SE);
}

bool IndexedReference::isCoeffForLoopZeroOrInvariant(const SCEV &Subscript,
                                                     const Loop &L) const {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(&Subscript);
  return (AR != nullptr) ? AR->getLoop() != &L
                         : SE.isLoopInvariant(&Subscript, &L);
}

bool IndexedReference::isSimpleAddRecurrence(const SCEV &Subscript,
                                             const Loop &L) const {
  if (!isa<SCEVAddRecExpr>(Subscript))
    return false;

  const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(&Subscript);
  assert(AR->getLoop() && "AR should have a loop");

  if (!AR->isAffine())
    return false;

  const SCEV *Start = AR->getStart();
  const SCEV *Step = AR->getStepRecurrence(SE);

  if (!SE.isLoopInvariant(Start, &L) || !SE.isLoopInvariant(Step, &L))
    return false;

  return true;
}

bool IndexedReference::isAliased(const IndexedReference &Other,
                                 AAResults &AA) const {
  const auto &Loc1 = MemoryLocation::get(&StoreOrLoadInst);
  const auto &Loc2 = MemoryLocation::get(&Other.StoreOrLoadInst);
  return AA.isMustAlias(Loc1, Loc2);
}

//===----------------------------------------------------------------------===//
// CacheCost implementation
//
raw_ostream &llvm::operator<<(raw_ostream &OS, const CacheCost &CC) {
  for (const auto &LC : CC.LoopCosts) {
    const Loop *L = LC.first;
    OS << "Loop '" << L->getName() << "' has cost = " << LC.second << "\n";
  }
  return OS;
}

CacheCost::CacheCost(const LoopVectorTy &Loops, const LoopInfo &LI,
                     ScalarEvolution &SE, TargetTransformInfo &TTI,
                     AAResults &AA, DependenceInfo &DI,
                     Optional<unsigned> TRT)
    : Loops(Loops), TripCounts(), LoopCosts(),
      TRT((TRT == None) ? Optional<unsigned>(TemporalReuseThreshold) : TRT),
      LI(LI), SE(SE), TTI(TTI), AA(AA), DI(DI) {
  assert(!Loops.empty() && "Expecting a non-empty loop vector.");

  for (const Loop *L : Loops) {
    unsigned TripCount = SE.getSmallConstantTripCount(L);
    TripCount = (TripCount == 0) ? DefaultTripCount : TripCount;
    TripCounts.push_back({L, TripCount});
  }

  calculateCacheFootprint();
}

std::unique_ptr<CacheCost>
CacheCost::getCacheCost(Loop &Root, LoopStandardAnalysisResults &AR,
                        DependenceInfo &DI, Optional<unsigned> TRT) {
  if (!Root.isOutermost()) {
    LLVM_DEBUG(dbgs() << "Expecting the outermost loop in a loop nest\n");
    return nullptr;
  }

  LoopVectorTy Loops;
  append_range(Loops, breadth_first(&Root));

  if (!getInnerMostLoop(Loops)) {
    LLVM_DEBUG(dbgs() << "Cannot compute cache cost of loop nest with more "
                         "than one innermost loop\n");
    return nullptr;
  }

  return std::make_unique<CacheCost>(Loops, AR.LI, AR.SE, AR.TTI, AR.AA, DI, TRT);
}

void CacheCost::calculateCacheFootprint() {
  LLVM_DEBUG(dbgs() << "POPULATING REFERENCE GROUPS\n");
  ReferenceGroupsTy RefGroups;
  if (!populateReferenceGroups(RefGroups))
    return;

  LLVM_DEBUG(dbgs() << "COMPUTING LOOP CACHE COSTS\n");
  for (const Loop *L : Loops) {
    assert((std::find_if(LoopCosts.begin(), LoopCosts.end(),
                         [L](const LoopCacheCostTy &LCC) {
                           return LCC.first == L;
                         }) == LoopCosts.end()) &&
           "Should not add duplicate element");
    CacheCostTy LoopCost = computeLoopCacheCost(*L, RefGroups);
    LoopCosts.push_back(std::make_pair(L, LoopCost));
  }

  sortLoopCosts();
  RefGroups.clear();
}

bool CacheCost::populateReferenceGroups(ReferenceGroupsTy &RefGroups) const {
  assert(RefGroups.empty() && "Reference groups should be empty");

  unsigned CLS = TTI.getCacheLineSize();
  Loop *InnerMostLoop = getInnerMostLoop(Loops);
  assert(InnerMostLoop != nullptr && "Expecting a valid innermost loop");

  for (BasicBlock *BB : InnerMostLoop->getBlocks()) {
    for (Instruction &I : *BB) {
      if (!isa<StoreInst>(I) && !isa<LoadInst>(I))
        continue;

      std::unique_ptr<IndexedReference> R(new IndexedReference(I, LI, SE));
      if (!R->isValid())
        continue;

      bool Added = false;
      for (ReferenceGroupTy &RefGroup : RefGroups) {
        const IndexedReference &Representative = *RefGroup.front().get();
        LLVM_DEBUG({
          dbgs() << "References:\n";
          dbgs().indent(2) << *R << "\n";
          dbgs().indent(2) << Representative << "\n";
        });


       // FIXME: Both positive and negative access functions will be placed
       // into the same reference group, resulting in a bi-directional array
       // access such as:
       //   for (i = N; i > 0; i--)
       //     A[i] = A[N - i];
       // having the same cost calculation as a single dimention access pattern
       //   for (i = 0; i < N; i++)
       //     A[i] = A[i];
       // when in actuality, depending on the array size, the first example
       // should have a cost closer to 2x the second due to the two cache
       // access per iteration from opposite ends of the array
        Optional<bool> HasTemporalReuse =
            R->hasTemporalReuse(Representative, *TRT, *InnerMostLoop, DI, AA);
        Optional<bool> HasSpacialReuse =
            R->hasSpacialReuse(Representative, CLS, AA);

        if ((HasTemporalReuse.hasValue() && *HasTemporalReuse) ||
            (HasSpacialReuse.hasValue() && *HasSpacialReuse)) {
          RefGroup.push_back(std::move(R));
          Added = true;
          break;
        }
      }

      if (!Added) {
        ReferenceGroupTy RG;
        RG.push_back(std::move(R));
        RefGroups.push_back(std::move(RG));
      }
    }
  }

  if (RefGroups.empty())
    return false;

  LLVM_DEBUG({
    dbgs() << "\nIDENTIFIED REFERENCE GROUPS:\n";
    int n = 1;
    for (const ReferenceGroupTy &RG : RefGroups) {
      dbgs().indent(2) << "RefGroup " << n << ":\n";
      for (const auto &IR : RG)
        dbgs().indent(4) << *IR << "\n";
      n++;
    }
    dbgs() << "\n";
  });

  return true;
}

CacheCostTy
CacheCost::computeLoopCacheCost(const Loop &L,
                                const ReferenceGroupsTy &RefGroups) const {
  if (!L.isLoopSimplifyForm())
    return InvalidCost;

  LLVM_DEBUG(dbgs() << "Considering loop '" << L.getName()
                    << "' as innermost loop.\n");

  // Compute the product of the trip counts of each other loop in the nest.
  CacheCostTy TripCountsProduct = 1;
  for (const auto &TC : TripCounts) {
    if (TC.first == &L)
      continue;
    TripCountsProduct *= TC.second;
  }

  CacheCostTy LoopCost = 0;
  for (const ReferenceGroupTy &RG : RefGroups) {
    CacheCostTy RefGroupCost = computeRefGroupCacheCost(RG, L);
    LoopCost += RefGroupCost * TripCountsProduct;
  }

  LLVM_DEBUG(dbgs().indent(2) << "Loop '" << L.getName()
                              << "' has cost=" << LoopCost << "\n");

  return LoopCost;
}

CacheCostTy CacheCost::computeRefGroupCacheCost(const ReferenceGroupTy &RG,
                                                const Loop &L) const {
  assert(!RG.empty() && "Reference group should have at least one member.");

  const IndexedReference *Representative = RG.front().get();
  return Representative->computeRefCost(L, TTI.getCacheLineSize());
}

//===----------------------------------------------------------------------===//
// LoopCachePrinterPass implementation
//
PreservedAnalyses LoopCachePrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                                            LoopStandardAnalysisResults &AR,
                                            LPMUpdater &U) {
  Function *F = L.getHeader()->getParent();
  DependenceInfo DI(F, &AR.AA, &AR.SE, &AR.LI);

  if (auto CC = CacheCost::getCacheCost(L, AR, DI))
    OS << *CC;

  return PreservedAnalyses::all();
}
