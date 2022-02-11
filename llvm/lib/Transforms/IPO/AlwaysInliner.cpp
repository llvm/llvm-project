//===- AlwaysInliner.cpp - Code to inline always_inline functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a custom inliner that handles only functions that
// are marked as "always inline".
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

PreservedAnalyses AlwaysInlinerPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  // Add inline assumptions during code generation.
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto &PSI = MAM.getResult<ProfileSummaryAnalysis>(M);

  SmallSetVector<CallBase *, 16> Calls;
  bool Changed = false;
  SmallVector<Function *, 16> InlinedFunctions;
  for (Function &F : M) {
    // When callee coroutine function is inlined into caller coroutine function
    // before coro-split pass,
    // coro-early pass can not handle this quiet well.
    // So we won't inline the coroutine function if it have not been unsplited
    if (F.isPresplitCoroutine())
      continue;

    if (!F.isDeclaration() && isInlineViable(F).isSuccess()) {
      Calls.clear();

      for (User *U : F.users())
        if (auto *CB = dyn_cast<CallBase>(U))
          if (CB->getCalledFunction() == &F) {
            if (F.hasFnAttribute(Attribute::AlwaysInline)) {
              // Avoid inlining if noinline call site attribute.
              if (!CB->isNoInline())
                Calls.insert(CB);
            } else if (CB->hasFnAttr(Attribute::AlwaysInline)) {
              // Ok, alwaysinline call site attribute.
              Calls.insert(CB);
            }
          }

      for (CallBase *CB : Calls) {
        Function *Caller = CB->getCaller();
        OptimizationRemarkEmitter ORE(Caller);
        auto OIC = shouldInline(
            *CB,
            [&](CallBase &CB) {
              return InlineCost::getAlways("always inline attribute");
            },
            ORE);
        assert(OIC);
        emitInlinedIntoBasedOnCost(ORE, CB->getDebugLoc(), CB->getParent(), F,
                                   *Caller, *OIC, false, DEBUG_TYPE);

        InlineFunctionInfo IFI(
            /*cg=*/nullptr, GetAssumptionCache, &PSI,
            &FAM.getResult<BlockFrequencyAnalysis>(*(CB->getCaller())),
            &FAM.getResult<BlockFrequencyAnalysis>(F));

        InlineResult Res = InlineFunction(
            *CB, IFI, &FAM.getResult<AAManager>(F), InsertLifetime);
        assert(Res.isSuccess() && "unexpected failure to inline");
        (void)Res;

        // Merge the attributes based on the inlining.
        AttributeFuncs::mergeAttributesForInlining(*Caller, F);

        Changed = true;
      }

      if (F.hasFnAttribute(Attribute::AlwaysInline)) {
        // Remember to try and delete this function afterward. This both avoids
        // re-walking the rest of the module and avoids dealing with any
        // iterator invalidation issues while deleting functions.
        InlinedFunctions.push_back(&F);
      }
    }
  }

  // Remove any live functions.
  erase_if(InlinedFunctions, [&](Function *F) {
    F->removeDeadConstantUsers();
    return !F->isDefTriviallyDead();
  });

  // Delete the non-comdat ones from the module and also from our vector.
  auto NonComdatBegin = partition(
      InlinedFunctions, [&](Function *F) { return F->hasComdat(); });
  for (Function *F : make_range(NonComdatBegin, InlinedFunctions.end())) {
    M.getFunctionList().erase(F);
    Changed = true;
  }
  InlinedFunctions.erase(NonComdatBegin, InlinedFunctions.end());

  if (!InlinedFunctions.empty()) {
    // Now we just have the comdat functions. Filter out the ones whose comdats
    // are not actually dead.
    filterDeadComdatFunctions(InlinedFunctions);
    // The remaining functions are actually dead.
    for (Function *F : InlinedFunctions) {
      M.getFunctionList().erase(F);
      Changed = true;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {

/// Inliner pass which only handles "always inline" functions.
///
/// Unlike the \c AlwaysInlinerPass, this uses the more heavyweight \c Inliner
/// base class to provide several facilities such as array alloca merging.
class AlwaysInlinerLegacyPass : public LegacyInlinerBase {

public:
  AlwaysInlinerLegacyPass() : LegacyInlinerBase(ID, /*InsertLifetime*/ true) {
    initializeAlwaysInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  AlwaysInlinerLegacyPass(bool InsertLifetime)
      : LegacyInlinerBase(ID, InsertLifetime) {
    initializeAlwaysInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  /// Main run interface method.  We override here to avoid calling skipSCC().
  bool runOnSCC(CallGraphSCC &SCC) override { return inlineCalls(SCC); }

  static char ID; // Pass identification, replacement for typeid

  InlineCost getInlineCost(CallBase &CB) override;

  using llvm::Pass::doFinalization;
  bool doFinalization(CallGraph &CG) override {
    return removeDeadFunctions(CG, /*AlwaysInlineOnly=*/true);
  }
};
}

char AlwaysInlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInlinerLegacyPass, "always-inline",
                      "Inliner for always_inline functions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AlwaysInlinerLegacyPass, "always-inline",
                    "Inliner for always_inline functions", false, false)

Pass *llvm::createAlwaysInlinerLegacyPass(bool InsertLifetime) {
  return new AlwaysInlinerLegacyPass(InsertLifetime);
}

/// Get the inline cost for the always-inliner.
///
/// The always inliner *only* handles functions which are marked with the
/// attribute to force inlining. As such, it is dramatically simpler and avoids
/// using the powerful (but expensive) inline cost analysis. Instead it uses
/// a very simple and boring direct walk of the instructions looking for
/// impossible-to-inline constructs.
///
/// Note, it would be possible to go to some lengths to cache the information
/// computed here, but as we only expect to do this for relatively few and
/// small functions which have the explicit attribute to force inlining, it is
/// likely not worth it in practice.
InlineCost AlwaysInlinerLegacyPass::getInlineCost(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();

  // Only inline direct calls to functions with always-inline attributes
  // that are viable for inlining.
  if (!Callee)
    return InlineCost::getNever("indirect call");

  // When callee coroutine function is inlined into caller coroutine function
  // before coro-split pass,
  // coro-early pass can not handle this quiet well.
  // So we won't inline the coroutine function if it have not been unsplited
  if (Callee->isPresplitCoroutine())
    return InlineCost::getNever("unsplited coroutine call");

  // FIXME: We shouldn't even get here for declarations.
  if (Callee->isDeclaration())
    return InlineCost::getNever("no definition");

  if (!CB.hasFnAttr(Attribute::AlwaysInline))
    return InlineCost::getNever("no alwaysinline attribute");

  if (Callee->hasFnAttribute(Attribute::AlwaysInline) && CB.isNoInline())
    return InlineCost::getNever("noinline call site attribute");

  auto IsViable = isInlineViable(*Callee);
  if (!IsViable.isSuccess())
    return InlineCost::getNever(IsViable.getFailureReason());

  return InlineCost::getAlways("always inliner");
}
