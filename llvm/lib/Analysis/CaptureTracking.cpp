//===--- CaptureTracking.cpp - Determine whether a pointer is captured ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
// A pointer value is captured if the function makes a copy of any part of the
// pointer that outlives the call.  Not being captured means, more or less, that
// the pointer is only dereferenced and not stored in a global.  Returning part
// of the pointer as the function return value may or may not count as capturing
// the pointer, depending on the context.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "capture-tracking"

STATISTIC(NumCaptured,          "Number of pointers maybe captured");
STATISTIC(NumNotCaptured,       "Number of pointers not captured");
STATISTIC(NumCapturedBefore,    "Number of pointers maybe captured before");
STATISTIC(NumNotCapturedBefore, "Number of pointers not captured before");

/// The default value for MaxUsesToExplore argument. It's relatively small to
/// keep the cost of analysis reasonable for clients like BasicAliasAnalysis,
/// where the results can't be cached.
/// TODO: we should probably introduce a caching CaptureTracking analysis and
/// use it where possible. The caching version can use much higher limit or
/// don't have this cap at all.
static cl::opt<unsigned>
    DefaultMaxUsesToExplore("capture-tracking-max-uses-to-explore", cl::Hidden,
                            cl::desc("Maximal number of uses to explore."),
                            cl::init(100));

unsigned llvm::getDefaultMaxUsesToExploreForCaptureTracking() {
  return DefaultMaxUsesToExplore;
}

CaptureTracker::~CaptureTracker() = default;

bool CaptureTracker::shouldExplore(const Use *U) { return true; }

bool CaptureTracker::isDereferenceableOrNull(Value *O, const DataLayout &DL) {
  // We want comparisons to null pointers to not be considered capturing,
  // but need to guard against cases like gep(p, -ptrtoint(p2)) == null,
  // which are equivalent to p == p2 and would capture the pointer.
  //
  // A dereferenceable pointer is a case where this is known to be safe,
  // because the pointer resulting from such a construction would not be
  // dereferenceable.
  //
  // It is not sufficient to check for inbounds GEP here, because GEP with
  // zero offset is always inbounds.
  bool CanBeNull, CanBeFreed;
  return O->getPointerDereferenceableBytes(DL, CanBeNull, CanBeFreed);
}

namespace {
struct SimpleCaptureTracker : public CaptureTracker {
  explicit SimpleCaptureTracker(bool ReturnCaptures)
      : ReturnCaptures(ReturnCaptures) {}

  void tooManyUses() override {
    LLVM_DEBUG(dbgs() << "Captured due to too many uses\n");
    Captured = true;
  }

  std::optional<CaptureComponents> captured(const Use *U,
                                            CaptureInfo CI) override {
    // TODO(captures): Use CaptureInfo.
    if (isa<ReturnInst>(U->getUser()) && !ReturnCaptures)
      return continueIgnoringReturn();

    LLVM_DEBUG(dbgs() << "Captured by: " << *U->getUser() << "\n");

    Captured = true;
    return stop();
  }

  bool ReturnCaptures;

  bool Captured = false;
};

/// Only find pointer captures which happen before the given instruction. Uses
/// the dominator tree to determine whether one instruction is before another.
/// Only support the case where the Value is defined in the same basic block
/// as the given instruction and the use.
struct CapturesBefore : public CaptureTracker {

  CapturesBefore(bool ReturnCaptures, const Instruction *I,
                 const DominatorTree *DT, bool IncludeI, const LoopInfo *LI)
      : BeforeHere(I), DT(DT), ReturnCaptures(ReturnCaptures),
        IncludeI(IncludeI), LI(LI) {}

  void tooManyUses() override { Captured = true; }

  bool isSafeToPrune(Instruction *I) {
    if (BeforeHere == I)
      return !IncludeI;

    // We explore this usage only if the usage can reach "BeforeHere".
    // If use is not reachable from entry, there is no need to explore.
    if (!DT->isReachableFromEntry(I->getParent()))
      return true;

    // Check whether there is a path from I to BeforeHere.
    return !isPotentiallyReachable(I, BeforeHere, nullptr, DT, LI);
  }

  std::optional<CaptureComponents> captured(const Use *U,
                                            CaptureInfo CI) override {
    // TODO(captures): Use CaptureInfo.
    Instruction *I = cast<Instruction>(U->getUser());
    if (isa<ReturnInst>(I) && !ReturnCaptures)
      return continueIgnoringReturn();

    // Check isSafeToPrune() here rather than in shouldExplore() to avoid
    // an expensive reachability query for every instruction we look at.
    // Instead we only do one for actual capturing candidates.
    if (isSafeToPrune(I))
      // If the use is not reachable, the instruction result isn't either.
      return continueIgnoringReturn();

    Captured = true;
    return stop();
  }

  const Instruction *BeforeHere;
  const DominatorTree *DT;

  bool ReturnCaptures;
  bool IncludeI;

  bool Captured = false;

  const LoopInfo *LI;
};

/// Find the 'earliest' instruction before which the pointer is known not to
/// be captured. Here an instruction A is considered earlier than instruction
/// B, if A dominates B. If 2 escapes do not dominate each other, the
/// terminator of the common dominator is chosen. If not all uses cannot be
/// analyzed, the earliest escape is set to the first instruction in the
/// function entry block.
// NOTE: Users have to make sure instructions compared against the earliest
// escape are not in a cycle.
struct EarliestCaptures : public CaptureTracker {

  EarliestCaptures(bool ReturnCaptures, Function &F, const DominatorTree &DT)
      : DT(DT), ReturnCaptures(ReturnCaptures), F(F) {}

  void tooManyUses() override {
    Captured = true;
    EarliestCapture = &*F.getEntryBlock().begin();
  }

  std::optional<CaptureComponents> captured(const Use *U,
                                            CaptureInfo CI) override {
    // TODO(captures): Use CaptureInfo.
    Instruction *I = cast<Instruction>(U->getUser());
    if (isa<ReturnInst>(I) && !ReturnCaptures)
      return continueIgnoringReturn();

    if (!EarliestCapture)
      EarliestCapture = I;
    else
      EarliestCapture = DT.findNearestCommonDominator(EarliestCapture, I);
    Captured = true;

    // Continue analysis, as we need to see all potential captures. However,
    // we do not need to follow the instruction result, as this use will
    // dominate any captures made through the instruction result..
    return continueIgnoringReturn();
  }

  Instruction *EarliestCapture = nullptr;

  const DominatorTree &DT;

  bool ReturnCaptures;

  bool Captured = false;

  Function &F;
};
} // namespace

/// PointerMayBeCaptured - Return true if this pointer value may be captured
/// by the enclosing function (which is required to exist).  This routine can
/// be expensive, so consider caching the results.  The boolean ReturnCaptures
/// specifies whether returning the value (or part of it) from the function
/// counts as capturing it or not.  The boolean StoreCaptures specified whether
/// storing the value (or part of it) into memory anywhere automatically
/// counts as capturing it or not.
bool llvm::PointerMayBeCaptured(const Value *V, bool ReturnCaptures,
                                bool StoreCaptures, unsigned MaxUsesToExplore) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  // TODO: If StoreCaptures is not true, we could do Fancy analysis
  // to determine whether this store is not actually an escape point.
  // In that case, BasicAliasAnalysis should be updated as well to
  // take advantage of this.
  (void)StoreCaptures;

  LLVM_DEBUG(dbgs() << "Captured?: " << *V << " = ");

  SimpleCaptureTracker SCT(ReturnCaptures);
  PointerMayBeCaptured(V, &SCT, MaxUsesToExplore);
  if (SCT.Captured)
    ++NumCaptured;
  else {
    ++NumNotCaptured;
    LLVM_DEBUG(dbgs() << "not captured\n");
  }
  return SCT.Captured;
}

/// PointerMayBeCapturedBefore - Return true if this pointer value may be
/// captured by the enclosing function (which is required to exist). If a
/// DominatorTree is provided, only captures which happen before the given
/// instruction are considered. This routine can be expensive, so consider
/// caching the results.  The boolean ReturnCaptures specifies whether
/// returning the value (or part of it) from the function counts as capturing
/// it or not.  The boolean StoreCaptures specified whether storing the value
/// (or part of it) into memory anywhere automatically counts as capturing it
/// or not.
bool llvm::PointerMayBeCapturedBefore(const Value *V, bool ReturnCaptures,
                                      bool StoreCaptures, const Instruction *I,
                                      const DominatorTree *DT, bool IncludeI,
                                      unsigned MaxUsesToExplore,
                                      const LoopInfo *LI) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  if (!DT)
    return PointerMayBeCaptured(V, ReturnCaptures, StoreCaptures,
                                MaxUsesToExplore);

  // TODO: See comment in PointerMayBeCaptured regarding what could be done
  // with StoreCaptures.

  CapturesBefore CB(ReturnCaptures, I, DT, IncludeI, LI);
  PointerMayBeCaptured(V, &CB, MaxUsesToExplore);
  if (CB.Captured)
    ++NumCapturedBefore;
  else
    ++NumNotCapturedBefore;
  return CB.Captured;
}

Instruction *llvm::FindEarliestCapture(const Value *V, Function &F,
                                       bool ReturnCaptures, bool StoreCaptures,
                                       const DominatorTree &DT,
                                       unsigned MaxUsesToExplore) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  EarliestCaptures CB(ReturnCaptures, F, DT);
  PointerMayBeCaptured(V, &CB, MaxUsesToExplore);
  if (CB.Captured)
    ++NumCapturedBefore;
  else
    ++NumNotCapturedBefore;
  return CB.EarliestCapture;
}

CaptureInfo llvm::DetermineUseCaptureKind(
    const Use &U,
    function_ref<bool(Value *, const DataLayout &)> IsDereferenceableOrNull) {
  Instruction *I = dyn_cast<Instruction>(U.getUser());

  // TODO: Investigate non-instruction uses.
  if (!I)
    return CaptureInfo::otherOnly();

  switch (I->getOpcode()) {
  case Instruction::Call:
  case Instruction::Invoke: {
    // TODO(captures): Make this more precise.
    auto *Call = cast<CallBase>(I);
    // Not captured if the callee is readonly, doesn't return a copy through
    // its return value and doesn't unwind (a readonly function can leak bits
    // by throwing an exception or not depending on the input value).
    if (Call->onlyReadsMemory() && Call->doesNotThrow() &&
        Call->getType()->isVoidTy())
      return CaptureInfo::none();

    // The pointer is not captured if returned pointer is not captured.
    // NOTE: CaptureTracking users should not assume that only functions
    // marked with nocapture do not capture. This means that places like
    // getUnderlyingObject in ValueTracking or DecomposeGEPExpression
    // in BasicAA also need to know about this property.
    if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(Call, true))
      return CaptureInfo::retOnly();

    // Volatile operations effectively capture the memory location that they
    // load and store to.
    if (auto *MI = dyn_cast<MemIntrinsic>(Call))
      if (MI->isVolatile())
        return CaptureInfo::otherOnly();

    // Calling a function pointer does not in itself cause the pointer to
    // be captured.  This is a subtle point considering that (for example)
    // the callee might return its own address.  It is analogous to saying
    // that loading a value from a pointer does not cause the pointer to be
    // captured, even though the loaded value might be the pointer itself
    // (think of self-referential objects).
    if (Call->isCallee(&U))
      return CaptureInfo::none();

    // Not captured if only passed via 'nocapture' arguments.
    assert(Call->isDataOperand(&U) && "Non-callee must be data operand");
    return Call->getCaptureInfo(Call->getDataOperandNo(&U));
  }
  case Instruction::Load:
    // Volatile loads make the address observable.
    if (cast<LoadInst>(I)->isVolatile())
      return CaptureInfo::otherOnly();
    return CaptureInfo::none();
  case Instruction::VAArg:
    // "va-arg" from a pointer does not cause it to be captured.
    return CaptureInfo::none();
  case Instruction::Store:
    // Stored the pointer - conservatively assume it may be captured.
    // Volatile stores make the address observable.
    if (U.getOperandNo() == 0 || cast<StoreInst>(I)->isVolatile())
      return CaptureInfo::otherOnly();
    return CaptureInfo::none();
  case Instruction::AtomicRMW: {
    // atomicrmw conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed is not captured,
    // but the value being stored is.
    // Volatile stores make the address observable.
    auto *ARMWI = cast<AtomicRMWInst>(I);
    if (U.getOperandNo() == 1 || ARMWI->isVolatile())
      return CaptureInfo::otherOnly();
    return CaptureInfo::none();
  }
  case Instruction::AtomicCmpXchg: {
    // cmpxchg conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed is not captured,
    // but the value being stored is.
    // Volatile stores make the address observable.
    auto *ACXI = cast<AtomicCmpXchgInst>(I);
    if (U.getOperandNo() == 1 || U.getOperandNo() == 2 || ACXI->isVolatile())
      return CaptureInfo::otherOnly();
    return CaptureInfo::none();
  }
  case Instruction::GetElementPtr:
    // AA does not support pointers of vectors, so GEP vector splats need to
    // be considered as captures.
    if (I->getType()->isVectorTy())
      return CaptureInfo::otherOnly();
    return CaptureInfo::retOnly();
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::AddrSpaceCast:
    // The original value is not captured via this if the new value isn't.
    return CaptureInfo::retOnly();
  case Instruction::ICmp: {
    unsigned Idx = U.getOperandNo();
    unsigned OtherIdx = 1 - Idx;
    if (auto *CPN = dyn_cast<ConstantPointerNull>(I->getOperand(OtherIdx))) {
      // TODO(captures): Remove these special cases once we make use of
      // captures(address_is_null).

      // Don't count comparisons of a no-alias return value against null as
      // captures. This allows us to ignore comparisons of malloc results
      // with null, for example.
      if (CPN->getType()->getAddressSpace() == 0)
        if (isNoAliasCall(U.get()->stripPointerCasts()))
          return CaptureInfo::none();
      if (!I->getFunction()->nullPointerIsDefined()) {
        auto *O = I->getOperand(Idx)->stripPointerCastsSameRepresentation();
        // Comparing a dereferenceable_or_null pointer against null cannot
        // lead to pointer escapes, because if it is not null it must be a
        // valid (in-bounds) pointer.
        const DataLayout &DL = I->getDataLayout();
        if (IsDereferenceableOrNull && IsDereferenceableOrNull(O, DL))
          return CaptureInfo::none();
      }
      return CaptureInfo::otherOnly(CaptureComponents::AddressIsNull);
    }

    // Otherwise, be conservative. There are crazy ways to capture pointers
    // using comparisons. However, only the address is captured, not the
    // provenance.
    return CaptureInfo::otherOnly(CaptureComponents::Address);
  }
  default:
    // Something else - be conservative and say it is captured.
    return CaptureInfo::otherOnly();
  }
}

void llvm::PointerMayBeCaptured(const Value *V, CaptureTracker *Tracker,
                                unsigned MaxUsesToExplore) {
  assert(V->getType()->isPointerTy() && "Capture is for pointers only!");
  if (MaxUsesToExplore == 0)
    MaxUsesToExplore = DefaultMaxUsesToExplore;

  SmallVector<const Use *, 20> Worklist;
  Worklist.reserve(getDefaultMaxUsesToExploreForCaptureTracking());
  SmallSet<const Use *, 20> Visited;

  auto AddUses = [&](const Value *V) {
    for (const Use &U : V->uses()) {
      // If there are lots of uses, conservatively say that the value
      // is captured to avoid taking too much compile time.
      if (Visited.size()  >= MaxUsesToExplore) {
        Tracker->tooManyUses();
        return false;
      }
      if (!Visited.insert(&U).second)
        continue;
      if (!Tracker->shouldExplore(&U))
        continue;
      Worklist.push_back(&U);
    }
    return true;
  };
  if (!AddUses(V))
    return;

  auto IsDereferenceableOrNull = [Tracker](Value *V, const DataLayout &DL) {
    return Tracker->isDereferenceableOrNull(V, DL);
  };
  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();
    CaptureInfo CI = DetermineUseCaptureKind(*U, IsDereferenceableOrNull);
    if (capturesNothing(CI))
      continue;
    CaptureComponents RetCC = CI.getRetComponents();
    if (!capturesNothing(CI.getOtherComponents())) {
      std::optional<CaptureComponents> Res = Tracker->captured(U, CI);
      if (!Res)
        return;
      assert(capturesNothing(*Res & ~RetCC) &&
             "captures() result must be subset of getRetComponents()");
      RetCC = *Res;
    }
    // TODO(captures): We could keep track of RetCC for the users.
    if (!capturesNothing(RetCC) && !AddUses(U->getUser()))
      return;
  }

  // All uses examined.
}

bool llvm::isNonEscapingLocalObject(
    const Value *V, SmallDenseMap<const Value *, bool, 8> *IsCapturedCache) {
  SmallDenseMap<const Value *, bool, 8>::iterator CacheIt;
  if (IsCapturedCache) {
    bool Inserted;
    std::tie(CacheIt, Inserted) = IsCapturedCache->insert({V, false});
    if (!Inserted)
      // Found cached result, return it!
      return CacheIt->second;
  }

  // If this is an identified function-local object, check to see if it escapes.
  if (isIdentifiedFunctionLocal(V)) {
    // Set StoreCaptures to True so that we can assume in our callers that the
    // pointer is not the result of a load instruction. Currently
    // PointerMayBeCaptured doesn't have any special analysis for the
    // StoreCaptures=false case; if it did, our callers could be refined to be
    // more precise.
    auto Ret = !PointerMayBeCaptured(V, false, /*StoreCaptures=*/true);
    if (IsCapturedCache)
      CacheIt->second = Ret;
    return Ret;
  }

  return false;
}
