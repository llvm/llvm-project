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

namespace {
struct SimpleCaptureTracker : public CaptureTracker {
  explicit SimpleCaptureTracker(bool ReturnCaptures, CaptureComponents Mask,
                                function_ref<bool(CaptureComponents)> StopFn)
      : ReturnCaptures(ReturnCaptures), Mask(Mask), StopFn(StopFn) {}

  void tooManyUses() override {
    LLVM_DEBUG(dbgs() << "Captured due to too many uses\n");
    CC = Mask;
  }

  Action captured(const Use *U, UseCaptureInfo CI) override {
    if (isa<ReturnInst>(U->getUser()) && !ReturnCaptures)
      return ContinueIgnoringReturn;

    if (capturesNothing(CI.UseCC & Mask))
      return Continue;

    LLVM_DEBUG(dbgs() << "Captured by: " << *U->getUser() << "\n");
    CC |= CI.UseCC & Mask;
    return StopFn(CC) ? Stop : Continue;
  }

  bool ReturnCaptures;
  CaptureComponents Mask;
  function_ref<bool(CaptureComponents)> StopFn;

  CaptureComponents CC = CaptureComponents::None;
};

/// Only find pointer captures which happen before the given instruction. Uses
/// the dominator tree to determine whether one instruction is before another.
/// Only support the case where the Value is defined in the same basic block
/// as the given instruction and the use.
struct CapturesBefore : public CaptureTracker {

  CapturesBefore(bool ReturnCaptures, const Instruction *I,
                 const DominatorTree *DT, bool IncludeI, const LoopInfo *LI,
                 CaptureComponents Mask,
                 function_ref<bool(CaptureComponents)> StopFn)
      : BeforeHere(I), DT(DT), ReturnCaptures(ReturnCaptures),
        IncludeI(IncludeI), LI(LI), Mask(Mask), StopFn(StopFn) {}

  void tooManyUses() override { CC = Mask; }

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

  Action captured(const Use *U, UseCaptureInfo CI) override {
    Instruction *I = cast<Instruction>(U->getUser());
    if (isa<ReturnInst>(I) && !ReturnCaptures)
      return ContinueIgnoringReturn;

    // Check isSafeToPrune() here rather than in shouldExplore() to avoid
    // an expensive reachability query for every instruction we look at.
    // Instead we only do one for actual capturing candidates.
    if (isSafeToPrune(I))
      // If the use is not reachable, the instruction result isn't either.
      return ContinueIgnoringReturn;

    if (capturesNothing(CI.UseCC & Mask))
      return Continue;

    CC |= CI.UseCC & Mask;
    return StopFn(CC) ? Stop : Continue;
  }

  const Instruction *BeforeHere;
  const DominatorTree *DT;

  bool ReturnCaptures;
  bool IncludeI;

  CaptureComponents CC = CaptureComponents::None;

  const LoopInfo *LI;
  CaptureComponents Mask;
  function_ref<bool(CaptureComponents)> StopFn;
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

  EarliestCaptures(bool ReturnCaptures, Function &F, const DominatorTree &DT,
                   CaptureComponents Mask)
      : DT(DT), ReturnCaptures(ReturnCaptures), F(F), Mask(Mask) {}

  void tooManyUses() override {
    CC = Mask;
    EarliestCapture = &*F.getEntryBlock().begin();
  }

  Action captured(const Use *U, UseCaptureInfo CI) override {
    Instruction *I = cast<Instruction>(U->getUser());
    if (isa<ReturnInst>(I) && !ReturnCaptures)
      return ContinueIgnoringReturn;

    if (capturesAnything(CI.UseCC & Mask)) {
      if (!EarliestCapture)
        EarliestCapture = I;
      else
        EarliestCapture = DT.findNearestCommonDominator(EarliestCapture, I);
      CC |= CI.UseCC & Mask;
    }

    // Continue analysis, as we need to see all potential captures.
    return Continue;
  }

  const DominatorTree &DT;
  bool ReturnCaptures;
  Function &F;
  CaptureComponents Mask;

  Instruction *EarliestCapture = nullptr;
  CaptureComponents CC = CaptureComponents::None;
};
} // namespace

CaptureComponents llvm::PointerMayBeCaptured(
    const Value *V, bool ReturnCaptures, CaptureComponents Mask,
    function_ref<bool(CaptureComponents)> StopFn, unsigned MaxUsesToExplore) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  LLVM_DEBUG(dbgs() << "Captured?: " << *V << " = ");

  SimpleCaptureTracker SCT(ReturnCaptures, Mask, StopFn);
  PointerMayBeCaptured(V, &SCT, MaxUsesToExplore);
  if (capturesAnything(SCT.CC))
    ++NumCaptured;
  else {
    ++NumNotCaptured;
    LLVM_DEBUG(dbgs() << "not captured\n");
  }
  return SCT.CC;
}

bool llvm::PointerMayBeCaptured(const Value *V, bool ReturnCaptures,
                                unsigned MaxUsesToExplore) {
  return capturesAnything(
      PointerMayBeCaptured(V, ReturnCaptures, CaptureComponents::All,
                           capturesAnything, MaxUsesToExplore));
}

CaptureComponents llvm::PointerMayBeCapturedBefore(
    const Value *V, bool ReturnCaptures, const Instruction *I,
    const DominatorTree *DT, bool IncludeI, CaptureComponents Mask,
    function_ref<bool(CaptureComponents)> StopFn, const LoopInfo *LI,
    unsigned MaxUsesToExplore) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  if (!DT)
    return PointerMayBeCaptured(V, ReturnCaptures, Mask, StopFn,
                                MaxUsesToExplore);

  CapturesBefore CB(ReturnCaptures, I, DT, IncludeI, LI, Mask, StopFn);
  PointerMayBeCaptured(V, &CB, MaxUsesToExplore);
  if (capturesAnything(CB.CC))
    ++NumCapturedBefore;
  else
    ++NumNotCapturedBefore;
  return CB.CC;
}

bool llvm::PointerMayBeCapturedBefore(const Value *V, bool ReturnCaptures,
                                      const Instruction *I,
                                      const DominatorTree *DT, bool IncludeI,
                                      unsigned MaxUsesToExplore,
                                      const LoopInfo *LI) {
  return capturesAnything(PointerMayBeCapturedBefore(
      V, ReturnCaptures, I, DT, IncludeI, CaptureComponents::All,
      capturesAnything, LI, MaxUsesToExplore));
}

std::pair<Instruction *, CaptureComponents>
llvm::FindEarliestCapture(const Value *V, Function &F, bool ReturnCaptures,
                          const DominatorTree &DT, CaptureComponents Mask,
                          unsigned MaxUsesToExplore) {
  assert(!isa<GlobalValue>(V) &&
         "It doesn't make sense to ask whether a global is captured.");

  EarliestCaptures CB(ReturnCaptures, F, DT, Mask);
  PointerMayBeCaptured(V, &CB, MaxUsesToExplore);
  if (capturesAnything(CB.CC))
    ++NumCapturedBefore;
  else
    ++NumNotCapturedBefore;
  return {CB.EarliestCapture, CB.CC};
}

UseCaptureInfo llvm::DetermineUseCaptureKind(const Use &U, const Value *Base) {
  Instruction *I = dyn_cast<Instruction>(U.getUser());

  // TODO: Investigate non-instruction uses.
  if (!I)
    return CaptureComponents::All;

  switch (I->getOpcode()) {
  case Instruction::Call:
  case Instruction::Invoke: {
    auto *Call = cast<CallBase>(I);
    // Not captured if the callee is readonly, doesn't return a copy through
    // its return value and doesn't unwind or diverge (a readonly function can
    // leak bits by throwing an exception or not depending on the input value).
    if (Call->onlyReadsMemory() && Call->doesNotThrow() && Call->willReturn() &&
        Call->getType()->isVoidTy())
      return CaptureComponents::None;

    // The pointer is not captured if returned pointer is not captured.
    // NOTE: CaptureTracking users should not assume that only functions
    // marked with nocapture do not capture. This means that places like
    // getUnderlyingObject in ValueTracking or DecomposeGEPExpression
    // in BasicAA also need to know about this property.
    if (isIntrinsicReturningPointerAliasingArgumentWithoutCapturing(Call, true))
      return UseCaptureInfo::passthrough();

    // Volatile operations effectively capture the memory location that they
    // load and store to.
    if (auto *MI = dyn_cast<MemIntrinsic>(Call))
      if (MI->isVolatile())
        return CaptureComponents::All;

    // Calling a function pointer does not in itself cause the pointer to
    // be captured.  This is a subtle point considering that (for example)
    // the callee might return its own address.  It is analogous to saying
    // that loading a value from a pointer does not cause the pointer to be
    // captured, even though the loaded value might be the pointer itself
    // (think of self-referential objects).
    if (Call->isCallee(&U))
      return CaptureComponents::None;

    // Not captured if only passed via 'nocapture' arguments.
    assert(Call->isDataOperand(&U) && "Non-callee must be data operand");
    CaptureInfo CI = Call->getCaptureInfo(Call->getDataOperandNo(&U));
    return UseCaptureInfo(CI.getOtherComponents(), CI.getRetComponents());
  }
  case Instruction::Load:
    // Volatile loads make the address observable.
    if (cast<LoadInst>(I)->isVolatile())
      return CaptureComponents::All;
    return CaptureComponents::None;
  case Instruction::VAArg:
    // "va-arg" from a pointer does not cause it to be captured.
    return CaptureComponents::None;
  case Instruction::Store:
    // Stored the pointer - conservatively assume it may be captured.
    // Volatile stores make the address observable.
    if (U.getOperandNo() == 0 || cast<StoreInst>(I)->isVolatile())
      return CaptureComponents::All;
    return CaptureComponents::None;
  case Instruction::AtomicRMW: {
    // atomicrmw conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed is not captured,
    // but the value being stored is.
    // Volatile stores make the address observable.
    auto *ARMWI = cast<AtomicRMWInst>(I);
    if (U.getOperandNo() == 1 || ARMWI->isVolatile())
      return CaptureComponents::All;
    return CaptureComponents::None;
  }
  case Instruction::AtomicCmpXchg: {
    // cmpxchg conceptually includes both a load and store from
    // the same location.
    // As with a store, the location being accessed is not captured,
    // but the value being stored is.
    // Volatile stores make the address observable.
    auto *ACXI = cast<AtomicCmpXchgInst>(I);
    if (U.getOperandNo() == 1 || U.getOperandNo() == 2 || ACXI->isVolatile())
      return CaptureComponents::All;
    return CaptureComponents::None;
  }
  case Instruction::GetElementPtr:
    // AA does not support pointers of vectors, so GEP vector splats need to
    // be considered as captures.
    if (I->getType()->isVectorTy())
      return CaptureComponents::All;
    return UseCaptureInfo::passthrough();
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::AddrSpaceCast:
    // The original value is not captured via this if the new value isn't.
    return UseCaptureInfo::passthrough();
  case Instruction::ICmp: {
    unsigned Idx = U.getOperandNo();
    unsigned OtherIdx = 1 - Idx;
    if (isa<ConstantPointerNull>(I->getOperand(OtherIdx)) &&
        cast<ICmpInst>(I)->isEquality()) {
      // TODO(captures): Remove these special cases once we make use of
      // captures(address_is_null).

      // Don't count comparisons of a no-alias return value against null as
      // captures. This allows us to ignore comparisons of malloc results
      // with null, for example.
      if (U->getType()->getPointerAddressSpace() == 0)
        if (isNoAliasCall(U.get()->stripPointerCasts()))
          return CaptureComponents::None;

      // Check whether this is a comparison of the base pointer against
      // null.
      if (U.get() == Base)
        return CaptureComponents::AddressIsNull;
    }

    // Otherwise, be conservative. There are crazy ways to capture pointers
    // using comparisons. However, only the address is captured, not the
    // provenance.
    return CaptureComponents::Address;
  }
  default:
    // Something else - be conservative and say it is captured.
    return CaptureComponents::All;
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

  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();
    UseCaptureInfo CI = DetermineUseCaptureKind(*U, V);
    if (capturesAnything(CI.UseCC)) {
      switch (Tracker->captured(U, CI)) {
      case CaptureTracker::Stop:
        return;
      case CaptureTracker::ContinueIgnoringReturn:
        continue;
      case CaptureTracker::Continue:
        // Fall through to passthrough handling, but only if ResultCC contains
        // additional components that UseCC does not. We assume that a
        // capture at this point will be strictly more constraining than a
        // later capture from following the return value.
        if (capturesNothing(CI.ResultCC & ~CI.UseCC))
          continue;
        break;
      }
    }
    // TODO(captures): We could keep track of ResultCC for the users.
    if (capturesAnything(CI.ResultCC) && !AddUses(U->getUser()))
      return;
  }

  // All uses examined.
}
