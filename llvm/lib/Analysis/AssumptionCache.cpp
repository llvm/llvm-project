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
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/TargetTransformInfo.h"
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
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool>
    VerifyAssumptionCache("verify-assumption-cache", cl::Hidden,
                          cl::desc("Enable verification of assumption cache"),
                          cl::init(false));

SmallVector<AssumptionCache::ResultElem, 1> &
AssumptionCache::getOrInsertAffectedValues(Value *V) {
  // Try using find_as first to avoid creating extra value handles just for the
  // purpose of doing the lookup.
  auto AVI = AffectedValues.find_as(V);
  if (AVI != AffectedValues.end())
    return AVI->second;

  auto AVIP = AffectedValues.insert(
      {AffectedValueCallbackVH(V, this), SmallVector<ResultElem, 1>()});
  return AVIP.first->second;
}

static void
findAffectedValues(CallBase *CI, TargetTransformInfo *TTI,
                   SmallVectorImpl<AssumptionCache::ResultElem> &Affected) {
  // Note: This code must be kept in-sync with the code in
  // computeKnownBitsFromAssume in ValueTracking.

  auto AddAffected = [&Affected](Value *V, unsigned Idx =
                                               AssumptionCache::ExprResultIdx) {
    if (isa<Argument>(V)) {
      Affected.push_back({V, Idx});
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      Affected.push_back({I, Idx});

      // Peek through unary operators to find the source of the condition.
      Value *Op;
      if (match(I, m_BitCast(m_Value(Op))) ||
          match(I, m_PtrToInt(m_Value(Op))) || match(I, m_Not(m_Value(Op)))) {
        if (isa<Instruction>(Op) || isa<Argument>(Op))
          Affected.push_back({Op, Idx});
      }
    }
  };

  for (unsigned Idx = 0; Idx != CI->getNumOperandBundles(); Idx++) {
    if (CI->getOperandBundleAt(Idx).Inputs.size() > ABA_WasOn &&
        CI->getOperandBundleAt(Idx).getTagName() != IgnoreBundleTag)
      AddAffected(CI->getOperandBundleAt(Idx).Inputs[ABA_WasOn], Idx);
  }

  Value *Cond = CI->getArgOperand(0), *A, *B;
  AddAffected(Cond);

  CmpInst::Predicate Pred;
  if (match(Cond, m_ICmp(Pred, m_Value(A), m_Value(B)))) {
    AddAffected(A);
    AddAffected(B);

    if (Pred == ICmpInst::ICMP_EQ) {
      // For equality comparisons, we handle the case of bit inversion.
      auto AddAffectedFromEq = [&AddAffected](Value *V) {
        Value *A;
        if (match(V, m_Not(m_Value(A)))) {
          AddAffected(A);
          V = A;
        }

        Value *B;
        // (A & B) or (A | B) or (A ^ B).
        if (match(V, m_BitwiseLogic(m_Value(A), m_Value(B)))) {
          AddAffected(A);
          AddAffected(B);
        // (A << C) or (A >>_s C) or (A >>_u C) where C is some constant.
        } else if (match(V, m_Shift(m_Value(A), m_ConstantInt()))) {
          AddAffected(A);
        }
      };

      AddAffectedFromEq(A);
      AddAffectedFromEq(B);
    }

    Value *X;
    // Handle (A + C1) u< C2, which is the canonical form of A > C3 && A < C4,
    // and recognized by LVI at least.
    if (Pred == ICmpInst::ICMP_ULT &&
        match(A, m_Add(m_Value(X), m_ConstantInt())) &&
        match(B, m_ConstantInt()))
      AddAffected(X);
  }

  if (TTI) {
    const Value *Ptr;
    unsigned AS;
    std::tie(Ptr, AS) = TTI->getPredicatedAddrSpace(Cond);
    if (Ptr)
      AddAffected(const_cast<Value *>(Ptr->stripInBoundsOffsets()));
  }
}

void AssumptionCache::updateAffectedValues(AssumeInst *CI) {
  SmallVector<AssumptionCache::ResultElem, 16> Affected;
  findAffectedValues(CI, TTI, Affected);

  for (auto &AV : Affected) {
    auto &AVV = getOrInsertAffectedValues(AV.Assume);
    if (llvm::none_of(AVV, [&](ResultElem &Elem) {
          return Elem.Assume == CI && Elem.Index == AV.Index;
        }))
      AVV.push_back({CI, AV.Index});
  }
}

void AssumptionCache::unregisterAssumption(AssumeInst *CI) {
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
      HasNonnull |= !!Elem.Assume;
      if (HasNonnull && Found)
        break;
    }
    assert(Found && "already unregistered or incorrect cache state");
    if (!HasNonnull)
      AffectedValues.erase(AVI);
  }

  erase_value(AssumeHandles, CI);
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

  for (auto &A : AVI->second)
    if (!llvm::is_contained(NAVV, A))
      NAVV.push_back(A);
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
        AssumeHandles.push_back({&I, ExprResultIdx});

  // Mark the scan as complete.
  Scanned = true;

  // Update affected values.
  for (auto &A : AssumeHandles)
    updateAffectedValues(cast<AssumeInst>(A));
}

void AssumptionCache::registerAssumption(AssumeInst *CI) {
  // If we haven't scanned the function yet, just drop this assumption. It will
  // be found when we scan later.
  if (!Scanned)
    return;

  AssumeHandles.push_back({CI, ExprResultIdx});

#ifndef NDEBUG
  assert(CI->getParent() &&
         "Cannot register @llvm.assume call not in a basic block");
  assert(&F == CI->getParent()->getParent() &&
         "Cannot register @llvm.assume call not in this function");

  // We expect the number of assumptions to be small, so in an asserts build
  // check that we don't accumulate duplicates and that all assumptions point
  // to the same function.
  SmallPtrSet<Value *, 16> AssumptionSet;
  for (auto &VH : AssumeHandles) {
    if (!VH)
      continue;

    assert(&F == cast<Instruction>(VH)->getParent()->getParent() &&
           "Cached assumption not inside this function!");
    assert(match(cast<CallInst>(VH), m_Intrinsic<Intrinsic::assume>()) &&
           "Cached something other than a call to @llvm.assume!");
    assert(AssumptionSet.insert(VH).second &&
           "Cache contains multiple copies of a call!");
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
  for (auto &VH : AC.assumptions())
    if (VH)
      OS << "  " << *cast<CallInst>(VH)->getArgOperand(0) << "\n";

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

  SmallPtrSet<const CallInst *, 4> AssumptionSet;
  for (const auto &I : AssumptionCaches) {
    for (auto &VH : I.second->assumptions())
      if (VH)
        AssumptionSet.insert(cast<CallInst>(VH));

    for (const BasicBlock &B : cast<Function>(*I.first))
      for (const Instruction &II : B)
        if (match(&II, m_Intrinsic<Intrinsic::assume>()) &&
            !AssumptionSet.count(cast<CallInst>(&II)))
          report_fatal_error("Assumption in scanned function not in cache");
  }
}

AssumptionCacheTracker::AssumptionCacheTracker() : ImmutablePass(ID) {
  initializeAssumptionCacheTrackerPass(*PassRegistry::getPassRegistry());
}

AssumptionCacheTracker::~AssumptionCacheTracker() = default;

char AssumptionCacheTracker::ID = 0;

INITIALIZE_PASS(AssumptionCacheTracker, "assumption-cache-tracker",
                "Assumption Cache Tracker", false, true)
