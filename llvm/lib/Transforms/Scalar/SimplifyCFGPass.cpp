//===- SimplifyCFGPass.cpp - CFG Simplification Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging, along
// with a collection of other peephole control flow optimizations.  For example:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//   * Changes invoke instructions to nounwind functions to be calls.
//   * Change things like "if (x) if (y)" into "if (x&y)".
//   * etc..
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "simplifycfg"

static cl::opt<unsigned> UserBonusInstThreshold(
    "bonus-inst-threshold", cl::Hidden, cl::init(1),
    cl::desc("Control the number of bonus instructions (default = 1)"));

static cl::opt<bool> UserKeepLoops(
    "keep-loops", cl::Hidden, cl::init(true),
    cl::desc("Preserve canonical loop structure (default = true)"));

static cl::opt<bool> UserSwitchToLookup(
    "switch-to-lookup", cl::Hidden, cl::init(false),
    cl::desc("Convert switches to lookup tables (default = false)"));

static cl::opt<bool> UserForwardSwitchCond(
    "forward-switch-cond", cl::Hidden, cl::init(false),
    cl::desc("Forward switch condition to phi ops (default = false)"));

static cl::opt<bool> UserHoistCommonInsts(
    "hoist-common-insts", cl::Hidden, cl::init(false),
    cl::desc("hoist common instructions (default = false)"));

static cl::opt<bool> UserSinkCommonInsts(
    "sink-common-insts", cl::Hidden, cl::init(false),
    cl::desc("Sink common instructions (default = false)"));


STATISTIC(NumSimpl, "Number of blocks simplified");

/// If we have more than one empty (other than phi node) return blocks,
/// merge them together to promote recursive block merging.
static bool mergeEmptyReturnBlocks(Function &F, DomTreeUpdater *DTU) {
  bool Changed = false;

  std::vector<DominatorTree::UpdateType> Updates;
  SmallVector<BasicBlock *, 8> DeadBlocks;

  BasicBlock *RetBlock = nullptr;

  // Scan all the blocks in the function, looking for empty return blocks.
  for (BasicBlock &BB : make_early_inc_range(F)) {
    if (DTU && DTU->isBBPendingDeletion(&BB))
      continue;

    // Only look at return blocks.
    ReturnInst *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!Ret) continue;

    // Only look at the block if it is empty or the only other thing in it is a
    // single PHI node that is the operand to the return.
    if (Ret != &BB.front()) {
      // Check for something else in the block.
      BasicBlock::iterator I(Ret);
      --I;
      // Skip over debug info.
      while (isa<DbgInfoIntrinsic>(I) && I != BB.begin())
        --I;
      if (!isa<DbgInfoIntrinsic>(I) &&
          (!isa<PHINode>(I) || I != BB.begin() || Ret->getNumOperands() == 0 ||
           Ret->getOperand(0) != &*I))
        continue;
    }

    // If this is the first returning block, remember it and keep going.
    if (!RetBlock) {
      RetBlock = &BB;
      continue;
    }

    // Skip merging if this would result in a CallBr instruction with a
    // duplicate destination. FIXME: See note in CodeGenPrepare.cpp.
    bool SkipCallBr = false;
    for (pred_iterator PI = pred_begin(&BB), E = pred_end(&BB);
         PI != E && !SkipCallBr; ++PI) {
      if (auto *CBI = dyn_cast<CallBrInst>((*PI)->getTerminator()))
        for (unsigned i = 0, e = CBI->getNumSuccessors(); i != e; ++i)
          if (RetBlock == CBI->getSuccessor(i)) {
            SkipCallBr = true;
            break;
          }
    }
    if (SkipCallBr)
      continue;

    // Otherwise, we found a duplicate return block.  Merge the two.
    Changed = true;

    // Case when there is no input to the return or when the returned values
    // agree is trivial.  Note that they can't agree if there are phis in the
    // blocks.
    if (Ret->getNumOperands() == 0 ||
        Ret->getOperand(0) ==
          cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0)) {
      // All predecessors of BB should now branch to RetBlock instead.
      if (DTU) {
        for (auto *Predecessor : predecessors(&BB)) {
          // But, iff Predecessor already branches to RetBlock,
          // don't (re-)add DomTree edge, because it already exists.
          if (!is_contained(successors(Predecessor), RetBlock))
            Updates.push_back({DominatorTree::Insert, Predecessor, RetBlock});
          Updates.push_back({DominatorTree::Delete, Predecessor, &BB});
        }
      }
      BB.replaceAllUsesWith(RetBlock);
      DeadBlocks.emplace_back(&BB);
      continue;
    }

    // If the canonical return block has no PHI node, create one now.
    PHINode *RetBlockPHI = dyn_cast<PHINode>(RetBlock->begin());
    if (!RetBlockPHI) {
      Value *InVal = cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0);
      pred_iterator PB = pred_begin(RetBlock), PE = pred_end(RetBlock);
      RetBlockPHI = PHINode::Create(Ret->getOperand(0)->getType(),
                                    std::distance(PB, PE), "merge",
                                    &RetBlock->front());

      for (pred_iterator PI = PB; PI != PE; ++PI)
        RetBlockPHI->addIncoming(InVal, *PI);
      RetBlock->getTerminator()->setOperand(0, RetBlockPHI);
    }

    // Turn BB into a block that just unconditionally branches to the return
    // block.  This handles the case when the two return blocks have a common
    // predecessor but that return different things.
    RetBlockPHI->addIncoming(Ret->getOperand(0), &BB);
    BB.getTerminator()->eraseFromParent();
    BranchInst::Create(RetBlock, &BB);
    if (DTU)
      Updates.push_back({DominatorTree::Insert, &BB, RetBlock});
  }

  if (DTU) {
    DTU->applyUpdates(Updates);
    for (auto *BB : DeadBlocks)
      DTU->deleteBB(BB);
  } else {
    for (auto *BB : DeadBlocks)
      BB->eraseFromParent();
  }

  return Changed;
}

/// Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool iterativelySimplifyCFG(Function &F, const TargetTransformInfo &TTI,
                                   DomTreeUpdater *DTU,
                                   const SimplifyCFGOptions &Options) {
  bool Changed = false;
  bool LocalChange = true;

  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 32> Edges;
  FindFunctionBackedges(F, Edges);
  SmallPtrSet<BasicBlock *, 16> UniqueLoopHeaders;
  for (unsigned i = 0, e = Edges.size(); i != e; ++i)
    UniqueLoopHeaders.insert(const_cast<BasicBlock *>(Edges[i].second));

  SmallVector<WeakVH, 16> LoopHeaders(UniqueLoopHeaders.begin(),
                                      UniqueLoopHeaders.end());

  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks and remove them if they are unneeded.
    for (Function::iterator BBIt = F.begin(); BBIt != F.end(); ) {
      BasicBlock &BB = *BBIt++;
      if (DTU) {
        assert(
            !DTU->isBBPendingDeletion(&BB) &&
            "Should not end up trying to simplify blocks marked for removal.");
        // Make sure that the advanced iterator does not point at the blocks
        // that are marked for removal, skip over all such blocks.
        while (BBIt != F.end() && DTU->isBBPendingDeletion(&*BBIt))
          ++BBIt;
      }
      if (simplifyCFG(&BB, TTI, DTU, Options, LoopHeaders)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

static bool simplifyFunctionCFGImpl(Function &F, const TargetTransformInfo &TTI,
                                    DominatorTree *DT,
                                    const SimplifyCFGOptions &Options) {
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

  bool EverChanged = removeUnreachableBlocks(F, DT ? &DTU : nullptr);
  EverChanged |= mergeEmptyReturnBlocks(F, DT ? &DTU : nullptr);
  EverChanged |= iterativelySimplifyCFG(F, TTI, DT ? &DTU : nullptr, Options);

  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // iterativelySimplifyCFG can (rarely) make some loops dead.  If this happens,
  // removeUnreachableBlocks is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid rerunning iterativelySimplifyCFG if the second pass of
  // removeUnreachableBlocks doesn't do anything.
  if (!removeUnreachableBlocks(F, DT ? &DTU : nullptr))
    return true;

  do {
    EverChanged = iterativelySimplifyCFG(F, TTI, DT ? &DTU : nullptr, Options);
    EverChanged |= removeUnreachableBlocks(F, DT ? &DTU : nullptr);
  } while (EverChanged);

  return true;
}

static bool simplifyFunctionCFG(Function &F, const TargetTransformInfo &TTI,
                                DominatorTree *DT,
                                const SimplifyCFGOptions &Options) {
  assert((!RequireAndPreserveDomTree ||
          (DT && DT->verify(DominatorTree::VerificationLevel::Full))) &&
         "Original domtree is invalid?");

  bool Changed = simplifyFunctionCFGImpl(F, TTI, DT, Options);

  assert((!RequireAndPreserveDomTree ||
          (DT && DT->verify(DominatorTree::VerificationLevel::Full))) &&
         "Failed to maintain validity of domtree!");

  return Changed;
}

// Command-line settings override compile-time settings.
static void applyCommandLineOverridesToOptions(SimplifyCFGOptions &Options) {
  if (UserBonusInstThreshold.getNumOccurrences())
    Options.BonusInstThreshold = UserBonusInstThreshold;
  if (UserForwardSwitchCond.getNumOccurrences())
    Options.ForwardSwitchCondToPhi = UserForwardSwitchCond;
  if (UserSwitchToLookup.getNumOccurrences())
    Options.ConvertSwitchToLookupTable = UserSwitchToLookup;
  if (UserKeepLoops.getNumOccurrences())
    Options.NeedCanonicalLoop = UserKeepLoops;
  if (UserHoistCommonInsts.getNumOccurrences())
    Options.HoistCommonInsts = UserHoistCommonInsts;
  if (UserSinkCommonInsts.getNumOccurrences())
    Options.SinkCommonInsts = UserSinkCommonInsts;
}

SimplifyCFGPass::SimplifyCFGPass() : Options() {
  applyCommandLineOverridesToOptions(Options);
}

SimplifyCFGPass::SimplifyCFGPass(const SimplifyCFGOptions &Opts)
    : Options(Opts) {
  applyCommandLineOverridesToOptions(Options);
}

PreservedAnalyses SimplifyCFGPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  Options.AC = &AM.getResult<AssumptionAnalysis>(F);
  DominatorTree *DT = nullptr;
  if (RequireAndPreserveDomTree)
    DT = &AM.getResult<DominatorTreeAnalysis>(F);
  if (F.hasFnAttribute(Attribute::OptForFuzzing)) {
    Options.setSimplifyCondBranch(false).setFoldTwoEntryPHINode(false);
  } else {
    Options.setSimplifyCondBranch(true).setFoldTwoEntryPHINode(true);
  }
  if (!simplifyFunctionCFG(F, TTI, DT, Options))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  if (RequireAndPreserveDomTree)
    PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}

namespace {
struct CFGSimplifyPass : public FunctionPass {
  static char ID;
  SimplifyCFGOptions Options;
  std::function<bool(const Function &)> PredicateFtor;

  CFGSimplifyPass(SimplifyCFGOptions Options_ = SimplifyCFGOptions(),
                  std::function<bool(const Function &)> Ftor = nullptr)
      : FunctionPass(ID), Options(Options_), PredicateFtor(std::move(Ftor)) {

    initializeCFGSimplifyPassPass(*PassRegistry::getPassRegistry());

    // Check for command-line overrides of options for debug/customization.
    applyCommandLineOverridesToOptions(Options);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F) || (PredicateFtor && !PredicateFtor(F)))
      return false;

    Options.AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    DominatorTree *DT = nullptr;
    if (RequireAndPreserveDomTree)
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    if (F.hasFnAttribute(Attribute::OptForFuzzing)) {
      Options.setSimplifyCondBranch(false)
             .setFoldTwoEntryPHINode(false);
    } else {
      Options.setSimplifyCondBranch(true)
             .setFoldTwoEntryPHINode(true);
    }

    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    return simplifyFunctionCFG(F, TTI, DT, Options);
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    if (RequireAndPreserveDomTree)
      AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    if (RequireAndPreserveDomTree)
      AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char CFGSimplifyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                    false)

// Public interface to the CFGSimplification pass
FunctionPass *
llvm::createCFGSimplificationPass(SimplifyCFGOptions Options,
                                  std::function<bool(const Function &)> Ftor) {
  return new CFGSimplifyPass(Options, std::move(Ftor));
}
