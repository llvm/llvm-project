//===- HotColdSplitting.cpp -- Outline Cold Regions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Outline cold regions to a separate function.
// TODO: Update BFI and BPI
// TODO: Add all the outlined functions to a separate section.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "hotcoldsplit"

STATISTIC(NumColdSESEFound,
          "Number of cold single entry single exit (SESE) regions found.");
STATISTIC(NumColdSESEOutlined,
          "Number of cold single entry single exit (SESE) regions outlined.");

using namespace llvm;

static cl::opt<bool> EnableStaticAnalyis("hot-cold-static-analysis",
                              cl::init(true), cl::Hidden);


namespace {

struct PostDomTree : PostDomTreeBase<BasicBlock> {
  PostDomTree(Function &F) { recalculate(F); }
};

typedef DenseSet<const BasicBlock *> DenseSetBB;

// From: https://reviews.llvm.org/D22558
// Exit is not part of the region.
static bool isSingleEntrySingleExit(BasicBlock *Entry, const BasicBlock *Exit,
                                    DominatorTree *DT, PostDomTree *PDT,
                                    SmallVectorImpl<BasicBlock *> &Region) {
  if (!DT->dominates(Entry, Exit))
    return false;

  if (!PDT->dominates(Exit, Entry))
    return false;

  Region.push_back(Entry);
  for (auto I = df_begin(Entry), E = df_end(Entry); I != E;) {
    if (*I == Exit) {
      I.skipChildren();
      continue;
    }
    if (!DT->dominates(Entry, *I))
      return false;
    Region.push_back(*I);
    ++I;
  }
  return true;
}

bool blockEndsInUnreachable(const BasicBlock &BB) {
  if (BB.empty())
    return true;
  const TerminatorInst *I = BB.getTerminator();
  if (isa<ReturnInst>(I) || isa<IndirectBrInst>(I))
    return true;
  // Unreachable blocks do not have any successor.
  return succ_empty(&BB);
}

static
bool unlikelyExecuted(const BasicBlock &BB) {
  if (blockEndsInUnreachable(BB))
    return true;
  // Exception handling blocks are unlikely executed.
  if (BB.isEHPad())
    return true;
  for (const Instruction &I : BB)
    if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
      // The block is cold if it calls functions tagged as cold or noreturn.
      if (CI->hasFnAttr(Attribute::Cold) ||
          CI->hasFnAttr(Attribute::NoReturn))
        return true;

      // Assume that inline assembly is hot code.
      if (isa<InlineAsm>(CI->getCalledValue()))
        return false;
    }
  return false;
}

static DenseSetBB getHotBlocks(Function &F) {
  // First mark all function basic blocks as hot or cold.
  DenseSet<const BasicBlock *> ColdBlocks;
  for (BasicBlock &BB : F)
    if (unlikelyExecuted(BB))
      ColdBlocks.insert(&BB);
  // Forward propagation.
  DenseSetBB AllColdBlocks;
  SmallVector<const BasicBlock *, 8> WL;
  DenseSetBB Visited; // Track hot blocks.

  const BasicBlock *It = &F.front();
  const TerminatorInst *TI = It->getTerminator();
  if (!ColdBlocks.count(It)) {
    Visited.insert(It);
    // Breadth First Search to mark edges not reachable from cold.
    WL.push_back(It);
    while (WL.size() > 0) {
      It = WL.pop_back_val();
      for (const BasicBlock *Succ : successors(TI)) {
        // Do not visit blocks that are cold.
        if (!ColdBlocks.count(Succ) && !Visited.count(Succ)) {
          Visited.insert(Succ);
          WL.push_back(Succ);
        }
      }
    }
  }

  return Visited;
}

class HotColdSplitting {
public:
  HotColdSplitting(ProfileSummaryInfo *ProfSI,
                   function_ref<BlockFrequencyInfo *(Function &)> GBFI,
                   std::function<OptimizationRemarkEmitter &(Function &)> *GORE)
      : PSI(ProfSI), GetBFI(GBFI), GetORE(GORE) {}
  bool run(Module &M);

private:
  bool shouldOutlineFrom(const Function &F) const;
  Function *outlineColdBlocks(Function &F,
                              const DenseSet<const BasicBlock *> &ColdBlock,
                              DominatorTree *DT, PostDomTree *PDT);
  Function *extractColdRegion(const SmallVectorImpl<BasicBlock *> &Region,
                              DominatorTree *DT, BlockFrequencyInfo *BFI,
                              OptimizationRemarkEmitter &ORE);
  bool isOutlineCandidate(const SmallVectorImpl<BasicBlock *> &Region,
                          const BasicBlock *Exit) const {
    if (!Exit)
      return false;
    // TODO: Find a better metric to compute the size of region being outlined.
    if (Region.size() == 1)
      return false;
    // Regions with landing pads etc.
    for (const BasicBlock *BB : Region) {
      if (BB->isEHPad() || BB->hasAddressTaken())
        return false;
    }
    return true;
  }
  SmallPtrSet<const Function *, 2> OutlinedFunctions;
  ProfileSummaryInfo *PSI;
  function_ref<BlockFrequencyInfo *(Function &)> GetBFI;
  std::function<OptimizationRemarkEmitter &(Function &)> *GetORE;
};

class HotColdSplittingLegacyPass : public ModulePass {
public:
  static char ID;
  HotColdSplittingLegacyPass() : ModulePass(ID) {
    initializeHotColdSplittingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

// Returns false if the function should not be considered for hot-cold split
// optimization. Already outlined functions have coldcc so no need to check
// for them here.
bool HotColdSplitting::shouldOutlineFrom(const Function &F) const {
  if (F.size() <= 2)
    return false;

  if (F.hasAddressTaken())
    return false;

  if (F.hasFnAttribute(Attribute::AlwaysInline))
    return false;

  if (F.hasFnAttribute(Attribute::NoInline))
    return false;

  if (F.getCallingConv() == CallingConv::Cold)
    return false;

  if (PSI->isFunctionEntryCold(&F))
    return false;
  return true;
}

Function *
HotColdSplitting::extractColdRegion(const SmallVectorImpl<BasicBlock *> &Region,
                                    DominatorTree *DT, BlockFrequencyInfo *BFI,
                                    OptimizationRemarkEmitter &ORE) {
  LLVM_DEBUG(for (auto *BB : Region)
          llvm::dbgs() << "\nExtracting: " << *BB;);
  // TODO: Pass BFI and BPI to update profile information.
  CodeExtractor CE(Region, DT, /*AggregateArgs*/ false, nullptr, nullptr,
                   /* AllowVarargs */ false);

  SetVector<Value *> Inputs, Outputs, Sinks;
  CE.findInputsOutputs(Inputs, Outputs, Sinks);

  // Do not extract regions that have live exit variables.
  if (Outputs.size() > 0)
    return nullptr;

  if (Function *OutF = CE.extractCodeRegion()) {
    User *U = *OutF->user_begin();
    CallInst *CI = cast<CallInst>(U);
    CallSite CS(CI);
    NumColdSESEOutlined++;
    OutF->setCallingConv(CallingConv::Cold);
    CS.setCallingConv(CallingConv::Cold);
    CI->setIsNoInline();
    LLVM_DEBUG(llvm::dbgs() << "Outlined Region at block: " << Region.front());
    return OutF;
  }

  ORE.emit([&]() {
    return OptimizationRemarkMissed(DEBUG_TYPE, "ExtractFailed",
                                    &*Region[0]->begin())
           << "Failed to extract region at block "
           << ore::NV("Block", Region.front());
  });
  return nullptr;
}

// Return the function created after outlining, nullptr otherwise.
Function *HotColdSplitting::outlineColdBlocks(Function &F,
                                              const  DenseSetBB &HotBlock,
                                              DominatorTree *DT,
                                              PostDomTree *PDT) {
  auto BFI = GetBFI(F);
  auto &ORE = (*GetORE)(F);
  // Walking the dominator tree allows us to find the largest
  // cold region.
  BasicBlock *Begin = DT->getRootNode()->getBlock();
  for (auto I = df_begin(Begin), E = df_end(Begin); I != E; ++I) {
    BasicBlock *BB = *I;
    if (PSI->isColdBB(BB, BFI) || !HotBlock.count(BB)) {
      SmallVector<BasicBlock *, 4> ValidColdRegion, Region;
      auto *BBNode = (*PDT)[BB];
      auto Exit = BBNode->getIDom()->getBlock();
      // We might need a virtual exit which post-dominates all basic blocks.
      if (!Exit)
        continue;
      BasicBlock *ExitColdRegion = nullptr;
      // Estimated cold region between a BB and its dom-frontier.
      while (isSingleEntrySingleExit(BB, Exit, DT, PDT, Region) &&
             isOutlineCandidate(Region, Exit)) {
        ExitColdRegion = Exit;
        ValidColdRegion = Region;
        Region.clear();
        // Update Exit recursively to its dom-frontier.
        Exit = (*PDT)[Exit]->getIDom()->getBlock();
      }
      if (ExitColdRegion) {
        ++NumColdSESEFound;
        // Candidate for outlining. FIXME: Continue outlining.
        // FIXME: Shouldn't need uniquing, debug isSingleEntrySingleExit
        //std::sort(ValidColdRegion.begin(), ValidColdRegion.end());
        auto last = std::unique(ValidColdRegion.begin(), ValidColdRegion.end());
        ValidColdRegion.erase(last, ValidColdRegion.end());
        return extractColdRegion(ValidColdRegion, DT, BFI, ORE);
      }
    }
  }
  return nullptr;
}

bool HotColdSplitting::run(Module &M) {
  for (auto &F : M) {
    if (!shouldOutlineFrom(F))
      continue;
    DominatorTree DT(F);
    PostDomTree PDT(F);
    PDT.recalculate(F);
    DenseSetBB HotBlocks;
    if (EnableStaticAnalyis) // Static analysis of cold blocks.
      HotBlocks = getHotBlocks(F);

    auto Outlined = outlineColdBlocks(F, HotBlocks, &DT, &PDT);
    if (Outlined)
      OutlinedFunctions.insert(Outlined);
  }
  return true;
}

bool HotColdSplittingLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;
  ProfileSummaryInfo *PSI =
      getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  auto GBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };
  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GetORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  return HotColdSplitting(PSI, GBFI, &GetORE).run(M);
}

char HotColdSplittingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(HotColdSplittingLegacyPass, "hotcoldsplit",
                      "Hot Cold Splitting", false, false)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(HotColdSplittingLegacyPass, "hotcoldsplit",
                    "Hot Cold Splitting", false, false)

ModulePass *llvm::createHotColdSplittingPass() {
  return new HotColdSplittingLegacyPass();
}
