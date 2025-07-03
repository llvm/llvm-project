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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "inline"
static cl::opt<bool> EnableMem2RegInterleaving(
    "enable-always-inliner-mem2reg", cl::init(true), cl::Hidden,
    cl::desc("Enable interleaving always-inlining with alloca promotion"));

STATISTIC(NumAllocasPromoted,
          "Number of allocas promoted to registers after inlining");

namespace {

/// Sanity check for a call site's inlinability based on inline attributes.
static bool canInlineCallBase(CallBase *CB) {
  return CB->hasFnAttr(Attribute::AlwaysInline) &&
         !CB->getAttributes().hasFnAttr(Attribute::NoInline);
}

/// Attempt to inline a call site \p CB into its caller.
/// Returns true if the inlining was successful, false otherwise.
static bool attemptInlineFunction(
    Function &F, CallBase *CB, bool InsertLifetime,
    function_ref<AAResults &(Function &)> &GetAAR,
    function_ref<AssumptionCache &(Function &)> &GetAssumptionCache,
    ProfileSummaryInfo &PSI) {
  Function *Caller = CB->getCaller();
  OptimizationRemarkEmitter ORE(Caller);
  DebugLoc DLoc = CB->getDebugLoc();
  BasicBlock *Block = CB->getParent();

  InlineFunctionInfo IFI(GetAssumptionCache, &PSI, nullptr, nullptr);
  InlineResult Res = InlineFunction(*CB, IFI, /*MergeAttributes=*/true,
                                    &GetAAR(F), InsertLifetime);
  if (!Res.isSuccess()) {
    ORE.emit([&]() {
      return OptimizationRemarkMissed(DEBUG_TYPE, "NotInlined", DLoc, Block)
             << "'" << ore::NV("Callee", &F) << "' is not inlined into '"
             << ore::NV("Caller", Caller)
             << "': " << ore::NV("Reason", Res.getFailureReason());
    });
    return false;
  }

  emitInlinedIntoBasedOnCost(ORE, DLoc, Block, F, *Caller,
                             InlineCost::getAlways("always inline attribute"),
                             /*ForProfileContext=*/false, DEBUG_TYPE);

  return true;
}

/// This function inlines all functions that are marked with the always_inline
/// attribute. It also removes the inlined functions if they are dead after the
/// inlining process.
bool AlwaysInlineImpl(
    Module &M, bool InsertLifetime, ProfileSummaryInfo &PSI,
    FunctionAnalysisManager *FAM,
    function_ref<AssumptionCache &(Function &)> GetAssumptionCache,
    function_ref<AAResults &(Function &)> GetAAR) {
  SmallSetVector<CallBase *, 16> Calls;
  bool Changed = false;
  SmallVector<Function *, 16> InlinedComdatFunctions;

  for (Function &F : make_early_inc_range(M)) {
    if (F.isPresplitCoroutine())
      continue;

    if (F.isDeclaration() || !isInlineViable(F).isSuccess())
      continue;

    Calls.clear();

    for (User *U : F.users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (CB->getCalledFunction() == &F && canInlineCallBase(CB))
          Calls.insert(CB);

    for (CallBase *CB : Calls) {
      Function *Caller = CB->getCaller();
      Changed |= attemptInlineFunction(F, CB, InsertLifetime, GetAAR,
                                       GetAssumptionCache, PSI);
      if (FAM)
        FAM->invalidate(*Caller, PreservedAnalyses::none());
    }

    F.removeDeadConstantUsers();
    if (F.hasFnAttribute(Attribute::AlwaysInline) && F.isDefTriviallyDead()) {
      // Remember to try and delete this function afterward. This allows to call
      // filterDeadComdatFunctions() only once.
      if (F.hasComdat()) {
        InlinedComdatFunctions.push_back(&F);
      } else {
        if (FAM)
          FAM->clear(F, F.getName());
        M.getFunctionList().erase(F);
        Changed = true;
      }
    }
  }

  if (!InlinedComdatFunctions.empty()) {
    // Now we just have the comdat functions. Filter out the ones whose comdats
    // are not actually dead.
    filterDeadComdatFunctions(InlinedComdatFunctions);
    // The remaining functions are actually dead.
    for (Function *F : InlinedComdatFunctions) {
      if (FAM)
        FAM->clear(*F, F->getName());
      M.getFunctionList().erase(F);
      Changed = true;
    }
  }

  return Changed;
}

/// Promote allocas to registers if possible.
static void promoteAllocas(
    Function *Caller, SmallPtrSetImpl<AllocaInst *> &AllocasToPromote,
    function_ref<AssumptionCache &(Function &)> &GetAssumptionCache) {
  if (AllocasToPromote.empty())
    return;

  SmallVector<AllocaInst *, 4> PromotableAllocas;
  llvm::copy_if(AllocasToPromote, std::back_inserter(PromotableAllocas),
                isAllocaPromotable);
  if (PromotableAllocas.empty())
    return;

  DominatorTree DT(*Caller);
  AssumptionCache &AC = GetAssumptionCache(*Caller);
  PromoteMemToReg(PromotableAllocas, DT, &AC);
  NumAllocasPromoted += PromotableAllocas.size();
  // Emit a remark for the promotion.
  OptimizationRemarkEmitter ORE(Caller);
  DebugLoc DLoc = Caller->getEntryBlock().getTerminator()->getDebugLoc();
  ORE.emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "PromoteAllocas", DLoc,
                              &Caller->getEntryBlock())
           << "Promoting " << ore::NV("NumAlloca", PromotableAllocas.size())
           << " allocas to SSA registers in function '"
           << ore::NV("Function", Caller) << "'";
  });
  LLVM_DEBUG(dbgs() << "Promoted " << PromotableAllocas.size()
                    << " allocas to registers in function " << Caller->getName()
                    << "\n");
}

/// We use a different visitation order of functions here to solve a phase
/// ordering problem. After inlining, a caller function may have allocas that
/// were previously used for passing reference arguments to the callee that
/// are now promotable to registers, using SROA/mem2reg. However if we just let
/// the AlwaysInliner continue inlining everything at once, the later SROA pass
/// in the pipeline will end up placing phis for these allocas into blocks along
/// the dominance frontier which may extend further than desired (e.g. loop
/// headers). This can happen when the caller is then inlined into another
/// caller, and the allocas end up hoisted further before SROA is run.
///
/// Instead what we want is to try to do, as best as we can, is to inline leaf
/// functions into callers, and then run PromoteMemToReg() on the allocas that
/// were passed into the callee before it was inlined.
///
/// We want to do this *before* the caller is inlined into another caller
/// because we want the alloca promotion to happen before its scope extends too
/// far because of further inlining.
///
/// Here's a simple pseudo-example:
/// outermost_caller() {
///   for (...) {
///     middle_caller();
///   }
/// }
///
/// middle_caller() {
///   int stack_var;
///   inner_callee(&stack_var);
/// }
///
/// inner_callee(int *x) {
///   // Do something with x.
/// }
///
/// In this case, we want to inline inner_callee() into middle_caller() and
/// then promote stack_var to a register before we inline middle_caller() into
/// outermost_caller(). The regular always_inliner would inline everything at
/// once, and then SROA/mem2reg would promote stack_var to a register but in
/// the context of outermost_caller() which is not what we want.
bool AlwaysInlineInterleavedMem2RegImpl(
    Module &M, bool InsertLifetime, ProfileSummaryInfo &PSI,
    FunctionAnalysisManager &FAM,
    function_ref<AssumptionCache &(Function &)> GetAssumptionCache,
    function_ref<AAResults &(Function &)> GetAAR) {

  bool Changed = false;

  // Use SetVector as we may rely on the deterministic iteration order for
  // finding candidates later.
  SetVector<Function *> AlwaysInlineFunctions;

  MapVector<Function *, SmallVector<WeakVH>> CalleeToCallSites;
  // Incoming always-inline calls for a function.
  DenseMap<Function *, unsigned> IncomingAICount;
  // Outgoing always-inline calls for a function.
  DenseMap<Function *, unsigned> OutgoingAICount;
  // First collect all always_inline functions.
  for (Function &F : M) {
    if (F.isDeclaration() || !F.hasFnAttribute(Attribute::AlwaysInline) ||
        !isInlineViable(F).isSuccess())
      continue;
    if (F.isPresplitCoroutine())
      continue;
    AlwaysInlineFunctions.insert(&F);
  }

  DenseSet<Function *> ProcessedFunctions;
  SmallVector<Function *> InlinedComdatFns;
  // Build the call graph of always_inline functions.
  for (Function *F : AlwaysInlineFunctions) {
    for (User *U : F->users()) {
      if (auto *CB = dyn_cast<CallBase>(U)) {
        if (CB->getCalledFunction() != F || !canInlineCallBase(CB))
          continue;
        CalleeToCallSites[F].push_back(WeakVH(CB));
        // Keep track of the number of incoming calls to this function.
        // This is used to determine the order in which we inline functions.
        IncomingAICount[F]++;
        if (AlwaysInlineFunctions.count(CB->getCaller()))
          OutgoingAICount[CB->getCaller()]++;
      }
    }
  }

  SmallVector<Function *, 16> Worklist;
  for (Function *F : AlwaysInlineFunctions) {
    // If this is a always_inline leaf function, we select it for inlining.
    if (OutgoingAICount.lookup(F) == 0)
      Worklist.push_back(F);
  }

  while (!Worklist.empty()) {
    Function *Callee = Worklist.pop_back_val();
    auto &Calls = CalleeToCallSites[Callee];

    // Group the calls by their caller. This allows us to collect all allocas
    // which need to be promoted together.
    MapVector<Function *, SmallVector<WeakVH>> CallerToCalls;

    for (WeakVH &WH : Calls)
      if (auto *CB = dyn_cast_or_null<CallBase>(WH))
        CallerToCalls[CB->getCaller()].push_back(WH);

    // Now collect the allocas.
    for (auto &CallerAndCalls : CallerToCalls) {
      Function *Caller = CallerAndCalls.first;
      SmallVector<WeakVH> &CallerCalls = CallerAndCalls.second;
      SmallPtrSet<AllocaInst *, 4> AllocasToPromote;

      for (WeakVH &WH : CallerCalls) {
        if (auto *CB = dyn_cast_or_null<CallBase>(WH)) {
          for (Value *Arg : CB->args())
            if (auto *AI = dyn_cast<AllocaInst>(getUnderlyingObject(Arg)))
              AllocasToPromote.insert(AI);
        }
      }

      // Do the actual inlining.
      bool InlinedAny = false;
      SmallVector<WeakVH> SuccessfullyInlinedCalls;

      for (WeakVH &WH : CallerCalls) {
        if (auto *CB = dyn_cast_or_null<CallBase>(WH)) {
          if (attemptInlineFunction(*Callee, CB, InsertLifetime, GetAAR,
                                    GetAssumptionCache, PSI)) {
            Changed = true;
            InlinedAny = true;
            SuccessfullyInlinedCalls.push_back(WH);
          }
        }
      }

      if (!InlinedAny)
        continue;

      // Promote any allocas that were used by the just-inlined call site.
      promoteAllocas(Caller, AllocasToPromote, GetAssumptionCache);

      unsigned InlinedCountForCaller = SuccessfullyInlinedCalls.size();
      if (!AlwaysInlineFunctions.contains(Caller))
        continue; // Caller wasn't part of our always-inline call graph.
      unsigned OldOutgoing = OutgoingAICount[Caller];
      assert(OldOutgoing >= InlinedCountForCaller &&
             "Inlined more calls than we had outgoing calls!");
      OutgoingAICount[Caller] = OldOutgoing - InlinedCountForCaller;
      // If these were the last outgoing calls in the caller, we can now
      // consider it a leaf function and add it to the worklist.
      if (OutgoingAICount[Caller] == 0 && !ProcessedFunctions.count(Caller))
        Worklist.push_back(Caller);
    }

    ProcessedFunctions.insert(Callee);
    AlwaysInlineFunctions.remove(Callee);
    CalleeToCallSites.erase(Callee);

    Callee->removeDeadConstantUsers();
    if (Callee->hasFnAttribute(Attribute::AlwaysInline) &&
        Callee->isDefTriviallyDead()) {
      if (Callee->hasComdat()) {
        InlinedComdatFns.push_back(Callee);
      } else {
        M.getFunctionList().erase(Callee);
        Changed = true;
      }
    }

    if (AlwaysInlineFunctions.empty())
      break;

    // If we have no more leaf functions to inline, we use a greedy heuristic
    // that selects the function with the most incoming calls. The intuition is
    // inlining this function will eliminate the most call sites and give the
    // highest chance of creating new leaf functions.
    if (Worklist.empty()) {
      Function *BestFunc = nullptr;
      unsigned MaxIncoming = 0;
      for (Function *F : AlwaysInlineFunctions) {
        if (ProcessedFunctions.count(F))
          continue;

        unsigned CurrentIncoming = IncomingAICount.lookup(F);
        if (!BestFunc || CurrentIncoming > MaxIncoming) {
          BestFunc = F;
          MaxIncoming = CurrentIncoming;
        }
      }
      if (BestFunc)
        Worklist.push_back(BestFunc);
    }
  }

  if (!InlinedComdatFns.empty()) {
    filterDeadComdatFunctions(InlinedComdatFns);
    for (Function *F : InlinedComdatFns) {
      M.getFunctionList().erase(F);
      Changed = true;
    }
  }

  // We may have missed some call sites that were marked as always_inline but
  // for which the callee function itself wasn't always_inline. Call the
  // standard handler here to deal with those.
  Changed |= AlwaysInlineImpl(M, InsertLifetime, PSI, &FAM, GetAssumptionCache,
                              GetAAR);
  return Changed;
}

struct AlwaysInlinerLegacyPass : public ModulePass {
  bool InsertLifetime;

  AlwaysInlinerLegacyPass()
      : AlwaysInlinerLegacyPass(/*InsertLifetime*/ true) {}

  AlwaysInlinerLegacyPass(bool InsertLifetime)
      : ModulePass(ID), InsertLifetime(InsertLifetime) {
    initializeAlwaysInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  /// Main run interface method.
  bool runOnModule(Module &M) override {

    auto &PSI = getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
    auto GetAAR = [&](Function &F) -> AAResults & {
      return getAnalysis<AAResultsWrapperPass>(F).getAAResults();
    };
    auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
      return getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    };

    return AlwaysInlineImpl(M, InsertLifetime, PSI, /*FAM=*/nullptr,
                            GetAssumptionCache, GetAAR);
  }

  static char ID; // Pass identification, replacement for typeid

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
  }
};

} // namespace

char AlwaysInlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AlwaysInlinerLegacyPass, "always-inline",
                      "Inliner for always_inline functions", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(AlwaysInlinerLegacyPass, "always-inline",
                    "Inliner for always_inline functions", false, false)

Pass *llvm::createAlwaysInlinerLegacyPass(bool InsertLifetime) {
  return new AlwaysInlinerLegacyPass(InsertLifetime);
}

PreservedAnalyses AlwaysInlinerPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetAAR = [&](Function &F) -> AAResults & {
    return FAM.getResult<AAManager>(F);
  };
  auto &PSI = MAM.getResult<ProfileSummaryAnalysis>(M);

  bool Changed = false;
  if (EnableMem2RegInterleaving) {
    Changed = AlwaysInlineInterleavedMem2RegImpl(M, InsertLifetime, PSI, FAM,
                                                 GetAssumptionCache, GetAAR);
  } else {
    Changed = AlwaysInlineImpl(M, InsertLifetime, PSI, &FAM, GetAssumptionCache,
                               GetAAR);
  }
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  // We have already invalidated all analyses on modified functions.
  PA.preserveSet<AllAnalysesOn<Function>>();
  return PA;
}
