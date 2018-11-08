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
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
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
#include "llvm/Transforms/IPO/HotColdSplitting.h"
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

STATISTIC(NumColdRegionsFound, "Number of cold regions found.");
STATISTIC(NumColdRegionsOutlined, "Number of cold regions outlined.");

using namespace llvm;

static cl::opt<bool> EnableStaticAnalyis("hot-cold-static-analysis",
                              cl::init(true), cl::Hidden);

static cl::opt<int>
    MinOutliningThreshold("min-outlining-thresh", cl::init(3), cl::Hidden,
                          cl::desc("Code size threshold for outlining within a "
                                   "single BB (as a multiple of TCC_Basic)"));

namespace {

struct PostDomTree : PostDomTreeBase<BasicBlock> {
  PostDomTree(Function &F) { recalculate(F); }
};

/// A sequence of basic blocks.
///
/// A 0-sized SmallVector is slightly cheaper to move than a std::vector.
using BlockSequence = SmallVector<BasicBlock *, 0>;

// Same as blockEndsInUnreachable in CodeGen/BranchFolding.cpp. Do not modify
// this function unless you modify the MBB version as well.
//
/// A no successor, non-return block probably ends in unreachable and is cold.
/// Also consider a block that ends in an indirect branch to be a return block,
/// since many targets use plain indirect branches to return.
bool blockEndsInUnreachable(const BasicBlock &BB) {
  if (!succ_empty(&BB))
    return false;
  if (BB.empty())
    return true;
  const Instruction *I = BB.getTerminator();
  return !(isa<ReturnInst>(I) || isa<IndirectBrInst>(I));
}

static bool exceptionHandlingFunctions(const CallInst *CI) {
  auto F = CI->getCalledFunction();
  if (!F)
    return false;
  auto FName = F->getName();
  return FName == "__cxa_begin_catch" ||
         FName == "__cxa_free_exception" ||
         FName == "__cxa_allocate_exception" ||
         FName == "__cxa_begin_catch" ||
         FName == "__cxa_end_catch";
}

static bool unlikelyExecuted(const BasicBlock &BB) {
  if (blockEndsInUnreachable(BB))
    return true;
  // Exception handling blocks are unlikely executed.
  if (BB.isEHPad())
    return true;
  for (const Instruction &I : BB)
    if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
      // The block is cold if it calls functions tagged as cold or noreturn.
      if (CI->hasFnAttr(Attribute::Cold) ||
          CI->hasFnAttr(Attribute::NoReturn) ||
          exceptionHandlingFunctions(CI))
        return true;

      // Assume that inline assembly is hot code.
      if (isa<InlineAsm>(CI->getCalledValue()))
        return false;
    }
  return false;
}

/// Check whether it's safe to outline \p BB.
static bool mayExtractBlock(const BasicBlock &BB) {
  return !BB.hasAddressTaken();
}

/// Check whether \p BB is profitable to outline (i.e. its code size cost meets
/// the threshold set in \p MinOutliningThreshold).
static bool isProfitableToOutline(const BasicBlock &BB,
                                  TargetTransformInfo &TTI) {
  int Cost = 0;
  for (const Instruction &I : BB) {
    if (isa<DbgInfoIntrinsic>(&I) || &I == BB.getTerminator())
      continue;

    Cost += TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);

    if (Cost >= (MinOutliningThreshold * TargetTransformInfo::TCC_Basic))
      return true;
  }
  return false;
}

/// Identify the maximal region of cold blocks which includes \p SinkBB.
///
/// Include all blocks post-dominated by \p SinkBB, \p SinkBB itself, and all
/// blocks dominated by \p SinkBB. Exclude all other blocks, and blocks which
/// cannot be outlined.
///
/// Return an empty sequence if the cold region is too small to outline, or if
/// the cold region has no warm predecessors.
static BlockSequence findMaximalColdRegion(BasicBlock &SinkBB,
                                           TargetTransformInfo &TTI,
                                           DominatorTree &DT,
                                           PostDomTree &PDT) {
  // The maximal cold region.
  BlockSequence ColdRegion = {};

  // The ancestor farthest-away from SinkBB, and also post-dominated by it.
  BasicBlock *MaxAncestor = &SinkBB;
  unsigned MaxAncestorHeight = 0;

  // Visit SinkBB's ancestors using inverse DFS.
  auto PredIt = ++idf_begin(&SinkBB);
  auto PredEnd = idf_end(&SinkBB);
  while (PredIt != PredEnd) {
    BasicBlock &PredBB = **PredIt;
    bool SinkPostDom = PDT.dominates(&SinkBB, &PredBB);

    // If SinkBB does not post-dominate a predecessor, do not mark the
    // predecessor (or any of its predecessors) cold.
    if (!SinkPostDom || !mayExtractBlock(PredBB)) {
      PredIt.skipChildren();
      continue;
    }

    // Keep track of the post-dominated ancestor farthest away from the sink.
    unsigned AncestorHeight = PredIt.getPathLength();
    if (AncestorHeight > MaxAncestorHeight) {
      MaxAncestor = &PredBB;
      MaxAncestorHeight = AncestorHeight;
    }

    ColdRegion.push_back(&PredBB);
    ++PredIt;
  }

  // CodeExtractor requires that all blocks to be extracted must be dominated
  // by the first block to be extracted.
  //
  // To avoid spurious or repeated outlining, require that the max ancestor
  // has a predecessor. By construction this predecessor is not in the cold
  // region, i.e. its existence implies we don't outline the whole function.
  //
  // TODO: If MaxAncestor has no predecessors, we may be able to outline the
  // second largest cold region that has a predecessor.
  if (pred_empty(MaxAncestor) ||
      MaxAncestor->getSinglePredecessor() == MaxAncestor)
    return {};

  // Filter out predecessors not dominated by the max ancestor.
  //
  // TODO: Blocks not dominated by the max ancestor could be extracted as
  // other cold regions. Marking outlined calls as noreturn when appropriate
  // and outlining more than once per function could achieve most of the win.
  auto EraseIt = remove_if(ColdRegion, [&](BasicBlock *PredBB) {
    return PredBB != MaxAncestor && !DT.dominates(MaxAncestor, PredBB);
  });
  ColdRegion.erase(EraseIt, ColdRegion.end());

  // Add SinkBB to the cold region.
  ColdRegion.push_back(&SinkBB);

  // Ensure that the first extracted block is the max ancestor.
  if (ColdRegion[0] != MaxAncestor) {
    auto AncestorIt = find(ColdRegion, MaxAncestor);
    *AncestorIt = ColdRegion[0];
    ColdRegion[0] = MaxAncestor;
  }

  // Find all successors of SinkBB dominated by SinkBB using DFS.
  auto SuccIt = ++df_begin(&SinkBB);
  auto SuccEnd = df_end(&SinkBB);
  while (SuccIt != SuccEnd) {
    BasicBlock &SuccBB = **SuccIt;
    bool SinkDom = DT.dominates(&SinkBB, &SuccBB);

    // If SinkBB does not dominate a successor, do not mark the successor (or
    // any of its successors) cold.
    if (!SinkDom || !mayExtractBlock(SuccBB)) {
      SuccIt.skipChildren();
      continue;
    }

    ColdRegion.push_back(&SuccBB);
    ++SuccIt;
  }

  if (ColdRegion.size() == 1 && !isProfitableToOutline(*ColdRegion[0], TTI))
    return {};

  return ColdRegion;
}

/// Get the largest cold region in \p F.
static BlockSequence getLargestColdRegion(Function &F, ProfileSummaryInfo &PSI,
                                          BlockFrequencyInfo *BFI,
                                          TargetTransformInfo &TTI,
                                          DominatorTree &DT, PostDomTree &PDT) {
  // Keep track of the largest cold region.
  BlockSequence LargestColdRegion = {};

  for (BasicBlock &BB : F) {
    // Identify cold blocks.
    if (!mayExtractBlock(BB))
      continue;
    bool Cold =
        PSI.isColdBB(&BB, BFI) || (EnableStaticAnalyis && unlikelyExecuted(BB));
    if (!Cold)
      continue;

    LLVM_DEBUG({
      dbgs() << "Found cold block:\n";
      BB.dump();
    });

    // Find a maximal cold region we can outline.
    BlockSequence ColdRegion = findMaximalColdRegion(BB, TTI, DT, PDT);
    if (ColdRegion.empty()) {
      LLVM_DEBUG(dbgs() << "  Skipping (block not profitable to extract)\n");
      continue;
    }

    ++NumColdRegionsFound;

    LLVM_DEBUG({
      llvm::dbgs() << "Identified cold region with " << ColdRegion.size()
                   << " blocks:\n";
      for (BasicBlock *BB : ColdRegion)
        BB->dump();
    });

    // TODO: Outline more than one region.
    if (ColdRegion.size() > LargestColdRegion.size())
      LargestColdRegion = std::move(ColdRegion);
  }

  return LargestColdRegion;
}

class HotColdSplitting {
public:
  HotColdSplitting(ProfileSummaryInfo *ProfSI,
                   function_ref<BlockFrequencyInfo *(Function &)> GBFI,
                   function_ref<TargetTransformInfo &(Function &)> GTTI,
                   std::function<OptimizationRemarkEmitter &(Function &)> *GORE)
      : PSI(ProfSI), GetBFI(GBFI), GetTTI(GTTI), GetORE(GORE) {}
  bool run(Module &M);

private:
  bool shouldOutlineFrom(const Function &F) const;
  Function *extractColdRegion(const BlockSequence &Region, DominatorTree &DT,
                              BlockFrequencyInfo *BFI, TargetTransformInfo &TTI,
                              OptimizationRemarkEmitter &ORE, unsigned Count);
  SmallPtrSet<const Function *, 2> OutlinedFunctions;
  ProfileSummaryInfo *PSI;
  function_ref<BlockFrequencyInfo *(Function &)> GetBFI;
  function_ref<TargetTransformInfo &(Function &)> GetTTI;
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
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

// Returns false if the function should not be considered for hot-cold split
// optimization.
bool HotColdSplitting::shouldOutlineFrom(const Function &F) const {
  // Do not try to outline again from an already outlined cold function.
  if (OutlinedFunctions.count(&F))
    return false;

  if (F.size() <= 2)
    return false;

  // TODO: Consider only skipping functions marked `optnone` or `cold`.

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

Function *HotColdSplitting::extractColdRegion(const BlockSequence &Region,
                                              DominatorTree &DT,
                                              BlockFrequencyInfo *BFI,
                                              TargetTransformInfo &TTI,
                                              OptimizationRemarkEmitter &ORE,
                                              unsigned Count) {
  assert(!Region.empty());
  LLVM_DEBUG(for (auto *BB : Region)
          llvm::dbgs() << "\nExtracting: " << *BB;);

  // TODO: Pass BFI and BPI to update profile information.
  CodeExtractor CE(Region, &DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                   /* BPI */ nullptr, /* AllowVarArgs */ false,
                   /* AllowAlloca */ false,
                   /* Suffix */ "cold." + std::to_string(Count));

  SetVector<Value *> Inputs, Outputs, Sinks;
  CE.findInputsOutputs(Inputs, Outputs, Sinks);

  // Do not extract regions that have live exit variables.
  if (Outputs.size() > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Not outlining; live outputs\n");
    return nullptr;
  }

  // TODO: Run MergeBasicBlockIntoOnlyPred on the outlined function.
  Function *OrigF = Region[0]->getParent();
  if (Function *OutF = CE.extractCodeRegion()) {
    User *U = *OutF->user_begin();
    CallInst *CI = cast<CallInst>(U);
    CallSite CS(CI);
    NumColdRegionsOutlined++;
    if (TTI.useColdCCForColdCall(*OutF)) {
      OutF->setCallingConv(CallingConv::Cold);
      CS.setCallingConv(CallingConv::Cold);
    }
    CI->setIsNoInline();

    // Try to make the outlined code as small as possible on the assumption
    // that it's cold.
    assert(!OutF->hasFnAttribute(Attribute::OptimizeNone) &&
           "An outlined function should never be marked optnone");
    OutF->addFnAttr(Attribute::MinSize);

    LLVM_DEBUG(llvm::dbgs() << "Outlined Region: " << *OutF);
    ORE.emit([&]() {
      return OptimizationRemark(DEBUG_TYPE, "HotColdSplit",
                                &*Region[0]->begin())
             << ore::NV("Original", OrigF) << " split cold code into "
             << ore::NV("Split", OutF);
    });
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

bool HotColdSplitting::run(Module &M) {
  bool Changed = false;
  for (auto &F : M) {
    if (!shouldOutlineFrom(F)) {
      LLVM_DEBUG(llvm::dbgs() << "Not outlining in " << F.getName() << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Outlining in " << F.getName() << "\n");
    DominatorTree DT(F);
    PostDomTree PDT(F);
    PDT.recalculate(F);
    BlockFrequencyInfo *BFI = GetBFI(F);
    TargetTransformInfo &TTI = GetTTI(F);

    BlockSequence ColdRegion = getLargestColdRegion(F, *PSI, BFI, TTI, DT, PDT);
    if (ColdRegion.empty())
      continue;

    OptimizationRemarkEmitter &ORE = (*GetORE)(F);
    Function *Outlined =
        extractColdRegion(ColdRegion, DT, BFI, TTI, ORE, /*Count=*/1);
    if (Outlined) {
      OutlinedFunctions.insert(Outlined);
      Changed = true;
    }
  }
  return Changed;
}

bool HotColdSplittingLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;
  ProfileSummaryInfo *PSI =
      getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  auto GTTI = [this](Function &F) -> TargetTransformInfo & {
    return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };
  auto GBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };
  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GetORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  return HotColdSplitting(PSI, GBFI, GTTI, &GetORE).run(M);
}

PreservedAnalyses
HotColdSplittingPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  std::function<AssumptionCache &(Function &)> GetAssumptionCache =
      [&FAM](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };

  auto GBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };

  std::function<TargetTransformInfo &(Function &)> GTTI =
      [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GetORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  ProfileSummaryInfo *PSI = &AM.getResult<ProfileSummaryAnalysis>(M);

  if (HotColdSplitting(PSI, GBFI, GTTI, &GetORE).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
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
