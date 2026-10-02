//===- AssumptionCache.cpp - Cache finding @llvm.assume calls -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that keeps track of @llvm.assume intrinsics in
// the functions of a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool>
    VerifyAssumptionCache("verify-assumption-cache", cl::Hidden,
                          cl::desc("Enable verification of assumption cache"),
                          cl::init(false));

static cl::opt<unsigned> MaxAssumesPerValue(
    "max-assumes-per-value", cl::Hidden, cl::init(1024),
    cl::desc("Maximum number of assumptions to cache for a single value"));

SmallVector<AssumptionCache::ResultElem, 1> &
AssumptionCache::getOrInsertAffectedValues(Value *V) {
  // Try using find_as first to avoid creating extra value handles just for the
  // purpose of doing the lookup.
  auto AVI = AffectedValues.find_as(V);
  if (AVI != AffectedValues.end())
    return AVI->second;

  return AffectedValues[AffectedValueCallbackVH(V, this)];
}

void AssumptionCache::findValuesAffectedByOperandBundle(
    OperandBundleUse Bundle, function_ref<void(Value *)> InsertAffected) {
  auto AddAffectedVal = [&](Value *V) {
    if (isa<Argument, GlobalValue, Instruction>(V))
      InsertAffected(V);
  };

  if (Bundle.getTagName() == "separate_storage") {
    assert(Bundle.Inputs.size() == 2 && "separate_storage must have two args");
    AddAffectedVal(getUnderlyingObject(Bundle.Inputs[0]));
    AddAffectedVal(getUnderlyingObject(Bundle.Inputs[1]));
  } else if (Bundle.Inputs.size() > ABA_WasOn &&
             Bundle.getTagName() != IgnoreBundleTag)
    AddAffectedVal(Bundle.Inputs[ABA_WasOn]);
}

static void
findAffectedValues(CallBase *CI, TargetTransformInfo *TTI,
                   SmallVectorImpl<AssumptionCache::ResultElem> &Affected) {
  // Note: This code must be kept in-sync with the code in
  // computeKnownBitsFromAssume in ValueTracking.

  auto InsertAffected = [&Affected](Value *V) {
    Affected.push_back({V, AssumptionCache::ExprResultIdx});
  };

  auto AddAffectedVal = [&Affected](Value *V, unsigned Idx) {
    if (isa<Argument>(V) || isa<GlobalValue>(V) || isa<Instruction>(V)) {
      Affected.push_back({V, Idx});
    }
  };

  for (unsigned Idx = 0; Idx != CI->getNumOperandBundles(); Idx++)
    AssumptionCache::findValuesAffectedByOperandBundle(
        CI->getOperandBundleAt(Idx),
        [&](Value *V) { Affected.push_back({V, Idx}); });

  Value *Cond = CI->getArgOperand(0);
  findValuesAffectedByCondition(Cond, /*IsAssume=*/true, InsertAffected);

  if (TTI) {
    const Value *Ptr;
    unsigned AS;
    std::tie(Ptr, AS) = TTI->getPredicatedAddrSpace(Cond);
    if (Ptr)
      AddAffectedVal(const_cast<Value *>(Ptr->stripInBoundsOffsets()),
                     AssumptionCache::ExprResultIdx);
  }
}

void AssumptionCache::updateAffectedValues(AssumeInst *CI) {
  SmallVector<AssumptionCache::ResultElem, 16> Affected;
  findAffectedValues(CI, TTI, Affected);

  for (auto &AV : Affected) {
    auto &AVV = getOrInsertAffectedValues(AV.Assume);

    // Callers walk every entry cached for a value, including the ones left
    // behind by erased assumptions, so cache no more of them than an analysis
    // should walk.
    if (AVV.size() >= MaxAssumesPerValue)
      continue;

    if (llvm::none_of(AVV, [&](ResultElem &Elem) {
          return Elem.Assume == CI && Elem.Index == AV.Index;
        }))
      AVV.push_back({CI, AV.Index});
  }
}

void AssumptionCache::removeAffectedValues(AssumeInst *CI) {
  SmallVector<AssumptionCache::ResultElem, 16> Affected;
  findAffectedValues(CI, TTI, Affected);

  for (auto &AV : Affected) {
    auto AVI = AffectedValues.find_as(AV.Assume);
    if (AVI == AffectedValues.end())
      continue;
    bool Found = false;
    bool HasNonnull = false;
    for (ResultElem &Elem : AVI->second) {
      if (Elem.Assume == CI) {
        Found = true;
        Elem.Assume = nullptr;
      }

      // We need to iterate through this loop to determine the value of
      // HasNonnull, to avoid prematurely calling AffectedValues.erase(AVI).
      HasNonnull |= !!Elem.Assume;
      if (HasNonnull && Found)
        break;
    }

    if (!Found) {
      // It may well be the case that we fail to find an affected value in the
      // cache. In particular, if an assume call is updated via `Use::set()`, we
      // won't be notified that the affected value has changed and the cache
      // will silently go stale.
    } else if (!HasNonnull)
      AffectedValues.erase(AVI);
  }
}

void AssumptionCache::unregisterAssumption(AssumeInst *CI) {
  removeAffectedValues(CI);
  llvm::erase(AssumeHandles, CI);
}

void AssumptionCache::replaceAssumption(WeakVH &Handle, AssumeInst *New) {
  removeAffectedValues(cast<AssumeInst>(Handle));
  Handle = New;
  updateAffectedValues(New);
}

void AssumptionCache::AffectedValueCallbackVH::deleted() {
  AC->AffectedValues.erase(getValPtr());
  // 'this' now dangles!
}

void AssumptionCache::transferAffectedValuesInCache(Value *OV, Value *NV) {
  auto &NAVV = getOrInsertAffectedValues(NV);
  auto AVI = AffectedValues.find(OV);
  if (AVI == AffectedValues.end())
    return;

  for (auto &A : AVI->second) {
    if (NAVV.size() >= MaxAssumesPerValue)
      break;
    if (!llvm::is_contained(NAVV, A))
      NAVV.push_back(A);
  }
  AffectedValues.erase(OV);
}

void AssumptionCache::AffectedValueCallbackVH::allUsesReplacedWith(Value *NV) {
  if (!isa<Instruction>(NV) && !isa<Argument>(NV))
    return;

  // Any assumptions that affected this value now affect the new value.

  AC->transferAffectedValuesInCache(getValPtr(), NV);
  // 'this' now might dangle! If the AffectedValues map was resized to add an
  // entry for NV then this object might have been destroyed in favor of some
  // copy in the grown map.
}

void AssumptionCache::scanFunction() {
  assert(!Scanned && "Tried to scan the function twice!");
  assert(AssumeHandles.empty() && "Already have assumes when scanning!");

  // Go through all instructions in all blocks, add all calls to @llvm.assume
  // to this cache.
  for (BasicBlock &B : F)
    for (Instruction &I : B)
      if (isa<AssumeInst>(&I))
        AssumeHandles.push_back(&I);

  // Mark the scan as complete.
  Scanned = true;

  // Update affected values.
  for (auto &A : AssumeHandles)
    updateAffectedValues(cast<AssumeInst>(A));
}

/// Check the assumptions cached for \p F, collecting them in \p Cached. Returns
/// a description of the first invariant violated, or nullptr if there is none.
static const char *
findCacheViolation(const Function &F, ArrayRef<WeakVH> Assumptions,
                   SmallPtrSetImpl<const CallInst *> &Cached) {
  for (const WeakVH &VH : Assumptions) {
    if (!VH)
      continue;

    const auto *CI = cast<CallInst>(VH);
    if (CI->getFunction() != &F)
      return "Cached assumption not inside this function";
    if (!match(CI, m_Intrinsic<Intrinsic::assume>()))
      return "Cached something other than a call to @llvm.assume";
    if (!Cached.insert(CI).second)
      return "Cache contains multiple copies of a call";
  }

  return nullptr;
}

void AssumptionCache::registerAssumption(AssumeInst *CI) {
  // If we haven't scanned the function yet, just drop this assumption. It will
  // be found when we scan later.
  if (!Scanned)
    return;

  AssumeHandles.push_back(CI);

#ifndef NDEBUG
  assert(CI->getParent() &&
         "Cannot register @llvm.assume call not in a basic block");
  assert(&F == CI->getParent()->getParent() &&
         "Cannot register @llvm.assume call not in this function");

  // We expect the number of assumptions to be small, so in an asserts build
  // check that we don't accumulate duplicates and that all assumptions point
  // to the same function. Scanning the whole cache on every registration is
  // quadratic, so stop once it outgrows that expectation unless expensive
  // checks are enabled. Larger caches are checked by
  // AssumptionCacheTracker::verifyAnalysis() instead.
#ifdef EXPENSIVE_CHECKS
  constexpr unsigned MaxAssumesToVerify = std::numeric_limits<unsigned>::max();
#else
  constexpr unsigned MaxAssumesToVerify = 64;
#endif
  if (AssumeHandles.size() <= MaxAssumesToVerify) {
    SmallPtrSet<const CallInst *, 16> Cached;
    if (const char *Violation = findCacheViolation(F, AssumeHandles, Cached))
      llvm_unreachable(Violation);
  }
#endif

  updateAffectedValues(CI);
}

AssumptionCache AssumptionAnalysis::run(Function &F,
                                        FunctionAnalysisManager &FAM) {
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  return AssumptionCache(F, &TTI);
}

AnalysisKey AssumptionAnalysis::Key;

PreservedAnalyses AssumptionPrinterPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);

  OS << "Cached assumptions for function: " << F.getName() << "\n";
  for (auto &VH : AC.assumptions()) {
    if (!VH)
      continue;

    auto *Assume = cast<CallInst>(VH);
    if (!Assume->hasOperandBundles()) {
      OS << "  " << *Assume->getArgOperand(0) << "\n";
      continue;
    }

    assert(match(Assume->getArgOperand(0), m_One()) &&
           "assume must have trivial cond");
    OS << "  [ ";
    ListSeparator LS;
    for (const OperandBundleUse &BU : Assume->operand_bundles()) {
      OS << LS << '"' << BU.getTagName() << "\"(";
      interleaveComma(BU.Inputs, OS,
                      [&](const Use &Input) { Input->printAsOperand(OS); });
      OS << ')';
    }
    OS << " ]\n";
  }

  return PreservedAnalyses::all();
}

void AssumptionCacheTracker::FunctionCallbackVH::deleted() {
  auto I = ACT->AssumptionCaches.find_as(cast<Function>(getValPtr()));
  if (I != ACT->AssumptionCaches.end())
    ACT->AssumptionCaches.erase(I);
  // 'this' now dangles!
}

AssumptionCache &AssumptionCacheTracker::getAssumptionCache(Function &F) {
  // We probe the function map twice to try and avoid creating a value handle
  // around the function in common cases. This makes insertion a bit slower,
  // but if we have to insert we're going to scan the whole function so that
  // shouldn't matter.
  auto I = AssumptionCaches.find_as(&F);
  if (I != AssumptionCaches.end())
    return *I->second;

  auto *TTIWP = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
  auto *TTI = TTIWP ? &TTIWP->getTTI(F) : nullptr;

  // Ok, build a new cache by scanning the function, insert it and the value
  // handle into our map, and return the newly populated cache.
  auto IP = AssumptionCaches.insert(std::make_pair(
      FunctionCallbackVH(&F, this), std::make_unique<AssumptionCache>(F, TTI)));
  assert(IP.second && "Scanning function already in the map?");
  return *IP.first->second;
}

AssumptionCache *AssumptionCacheTracker::lookupAssumptionCache(Function &F) {
  auto I = AssumptionCaches.find_as(&F);
  if (I != AssumptionCaches.end())
    return I->second.get();
  return nullptr;
}

void AssumptionCacheTracker::verifyAnalysis() const {
  // FIXME: In the long term the verifier should not be controllable with a
  // flag. We should either fix all passes to correctly update the assumption
  // cache and enable the verifier unconditionally or somehow arrange for the
  // assumption list to be updated automatically by passes.
  if (!VerifyAssumptionCache)
    return;

  for (const auto &I : AssumptionCaches) {
    const Function &F = cast<Function>(*I.first);

    SmallPtrSet<const CallInst *, 4> Cached;
    if (const char *Violation =
            findCacheViolation(F, I.second->assumptions(), Cached))
      report_fatal_error(Violation);

    for (const BasicBlock &B : F)
      for (const Instruction &II : B)
        if (match(&II, m_Intrinsic<Intrinsic::assume>()) &&
            !Cached.count(cast<CallInst>(&II)))
          report_fatal_error("Assumption in scanned function not in cache");
  }
}

AssumptionCacheTracker::AssumptionCacheTracker() : ImmutablePass(ID) {}

AssumptionCacheTracker::~AssumptionCacheTracker() = default;

char AssumptionCacheTracker::ID = 0;

INITIALIZE_PASS(AssumptionCacheTracker, "assumption-cache-tracker",
                "Assumption Cache Tracker", false, true)
