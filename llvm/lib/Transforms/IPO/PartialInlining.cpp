//===- PartialInlining.cpp - Inline parts of functions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs partial inlining, typically by inlining an if statement
// that surrounds the body of the function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PartialInlining.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
using namespace llvm;

#define DEBUG_TYPE "partial-inlining"

STATISTIC(NumPartialInlined,
          "Number of callsites functions partially inlined into.");

// Command line option to disable partial-inlining. The default is false:
static cl::opt<bool>
    DisablePartialInlining("disable-partial-inlining", cl::init(false),
                           cl::Hidden, cl::desc("Disable partial ininling"));

static cl::opt<unsigned> MaxNumInlineBlocks(
    "max-num-inline-blocks", cl::init(5), cl::Hidden,
    cl::desc("Max Number of Blocks  To be Partially Inlined"));

// Command line option to set the maximum number of partial inlining allowed
// for the module. The default value of -1 means no limit.
static cl::opt<int> MaxNumPartialInlining(
    "max-partial-inlining", cl::init(-1), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Max number of partial inlining. The default is unlimited"));

namespace {

struct FunctionOutliningInfo {
  FunctionOutliningInfo()
      : Entries(), ReturnBlock(nullptr), NonReturnBlock(nullptr),
        ReturnBlockPreds() {}
  // Returns the number of blocks to be inlined including all blocks
  // in Entries and one return block.
  unsigned GetNumInlinedBlocks() const { return Entries.size() + 1; }

  // A set of blocks including the function entry that guard
  // the region to be outlined.
  SmallVector<BasicBlock *, 4> Entries;
  // The return block that is not included in the outlined region.
  BasicBlock *ReturnBlock;
  // The dominating block of the region ot be outlined.
  BasicBlock *NonReturnBlock;
  // The set of blocks in Entries that that are predecessors to ReturnBlock
  SmallVector<BasicBlock *, 4> ReturnBlockPreds;
};

struct PartialInlinerImpl {
  PartialInlinerImpl(
      std::function<AssumptionCache &(Function &)> *GetAC,
      std::function<TargetTransformInfo &(Function &)> *GTTI,
      Optional<function_ref<BlockFrequencyInfo &(Function &)>> GBFI,
      ProfileSummaryInfo *ProfSI)
      : GetAssumptionCache(GetAC), GetTTI(GTTI), GetBFI(GBFI), PSI(ProfSI) {}
  bool run(Module &M);
  Function *unswitchFunction(Function *F);

  std::unique_ptr<FunctionOutliningInfo> computeOutliningInfo(Function *F);

private:
  int NumPartialInlining = 0;
  std::function<AssumptionCache &(Function &)> *GetAssumptionCache;
  std::function<TargetTransformInfo &(Function &)> *GetTTI;
  Optional<function_ref<BlockFrequencyInfo &(Function &)>> GetBFI;
  ProfileSummaryInfo *PSI;

  bool shouldPartialInline(CallSite CS, OptimizationRemarkEmitter &ORE);
  bool IsLimitReached() {
    return (MaxNumPartialInlining != -1 &&
            NumPartialInlining >= MaxNumPartialInlining);
  }
};

struct PartialInlinerLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  PartialInlinerLegacyPass() : ModulePass(ID) {
    initializePartialInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    AssumptionCacheTracker *ACT = &getAnalysis<AssumptionCacheTracker>();
    TargetTransformInfoWrapperPass *TTIWP =
        &getAnalysis<TargetTransformInfoWrapperPass>();
    ProfileSummaryInfo *PSI =
        getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

    std::function<AssumptionCache &(Function &)> GetAssumptionCache =
        [&ACT](Function &F) -> AssumptionCache & {
      return ACT->getAssumptionCache(F);
    };

    std::function<TargetTransformInfo &(Function &)> GetTTI =
        [&TTIWP](Function &F) -> TargetTransformInfo & {
      return TTIWP->getTTI(F);
    };

    return PartialInlinerImpl(&GetAssumptionCache, &GetTTI, None, PSI).run(M);
  }
};
}

std::unique_ptr<FunctionOutliningInfo>
PartialInlinerImpl::computeOutliningInfo(Function *F) {
  BasicBlock *EntryBlock = &F->front();
  BranchInst *BR = dyn_cast<BranchInst>(EntryBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return std::unique_ptr<FunctionOutliningInfo>();

  // Returns true if Succ is BB's successor
  auto IsSuccessor = [](BasicBlock *Succ, BasicBlock *BB) {
    return is_contained(successors(BB), Succ);
  };

  auto SuccSize = [](BasicBlock *BB) {
    return std::distance(succ_begin(BB), succ_end(BB));
  };

  auto IsReturnBlock = [](BasicBlock *BB) {
    TerminatorInst *TI = BB->getTerminator();
    return isa<ReturnInst>(TI);
  };

  auto GetReturnBlock = [=](BasicBlock *Succ1, BasicBlock *Succ2) {
    if (IsReturnBlock(Succ1))
      return std::make_tuple(Succ1, Succ2);
    if (IsReturnBlock(Succ2))
      return std::make_tuple(Succ2, Succ1);

    return std::make_tuple<BasicBlock *, BasicBlock *>(nullptr, nullptr);
  };

  // Detect a triangular shape:
  auto GetCommonSucc = [=](BasicBlock *Succ1, BasicBlock *Succ2) {
    if (IsSuccessor(Succ1, Succ2))
      return std::make_tuple(Succ1, Succ2);
    if (IsSuccessor(Succ2, Succ1))
      return std::make_tuple(Succ2, Succ1);

    return std::make_tuple<BasicBlock *, BasicBlock *>(nullptr, nullptr);
  };

  std::unique_ptr<FunctionOutliningInfo> OutliningInfo =
      llvm::make_unique<FunctionOutliningInfo>();

  BasicBlock *CurrEntry = EntryBlock;
  bool CandidateFound = false;
  do {
    // The number of blocks to be inlined has already reached
    // the limit. When MaxNumInlineBlocks is set to 0 or 1, this
    // disables partial inlining for the function.
    if (OutliningInfo->GetNumInlinedBlocks() >= MaxNumInlineBlocks)
      break;

    if (SuccSize(CurrEntry) != 2)
      break;

    BasicBlock *Succ1 = *succ_begin(CurrEntry);
    BasicBlock *Succ2 = *(succ_begin(CurrEntry) + 1);

    BasicBlock *ReturnBlock, *NonReturnBlock;
    std::tie(ReturnBlock, NonReturnBlock) = GetReturnBlock(Succ1, Succ2);

    if (ReturnBlock) {
      OutliningInfo->Entries.push_back(CurrEntry);
      OutliningInfo->ReturnBlock = ReturnBlock;
      OutliningInfo->NonReturnBlock = NonReturnBlock;
      CandidateFound = true;
      break;
    }

    BasicBlock *CommSucc;
    BasicBlock *OtherSucc;
    std::tie(CommSucc, OtherSucc) = GetCommonSucc(Succ1, Succ2);

    if (!CommSucc)
      break;

    OutliningInfo->Entries.push_back(CurrEntry);
    CurrEntry = OtherSucc;

  } while (true);

  if (!CandidateFound)
    return std::unique_ptr<FunctionOutliningInfo>();

  // Do sanity check of the entries: threre should not
  // be any successors (not in the entry set) other than
  // {ReturnBlock, NonReturnBlock}
  assert(OutliningInfo->Entries[0] == &F->front());
  DenseSet<BasicBlock *> Entries;
  for (BasicBlock *E : OutliningInfo->Entries)
    Entries.insert(E);

  // Returns true of BB has Predecessor which is not
  // in Entries set.
  auto HasNonEntryPred = [Entries](BasicBlock *BB) {
    for (auto Pred : predecessors(BB)) {
      if (!Entries.count(Pred))
        return true;
    }
    return false;
  };
  auto CheckAndNormalizeCandidate =
      [Entries, HasNonEntryPred](FunctionOutliningInfo *OutliningInfo) {
        for (BasicBlock *E : OutliningInfo->Entries) {
          for (auto Succ : successors(E)) {
            if (Entries.count(Succ))
              continue;
            if (Succ == OutliningInfo->ReturnBlock)
              OutliningInfo->ReturnBlockPreds.push_back(E);
            else if (Succ != OutliningInfo->NonReturnBlock)
              return false;
          }
          // There should not be any outside incoming edges either:
          if (HasNonEntryPred(E))
            return false;
        }
        return true;
      };

  if (!CheckAndNormalizeCandidate(OutliningInfo.get()))
    return std::unique_ptr<FunctionOutliningInfo>();

  // Now further growing the candidate's inlining region by
  // peeling off dominating blocks from the outlining region:
  while (OutliningInfo->GetNumInlinedBlocks() < MaxNumInlineBlocks) {
    BasicBlock *Cand = OutliningInfo->NonReturnBlock;
    if (SuccSize(Cand) != 2)
      break;

    if (HasNonEntryPred(Cand))
      break;

    BasicBlock *Succ1 = *succ_begin(Cand);
    BasicBlock *Succ2 = *(succ_begin(Cand) + 1);

    BasicBlock *ReturnBlock, *NonReturnBlock;
    std::tie(ReturnBlock, NonReturnBlock) = GetReturnBlock(Succ1, Succ2);
    if (!ReturnBlock || ReturnBlock != OutliningInfo->ReturnBlock)
      break;

    if (NonReturnBlock->getSinglePredecessor() != Cand)
      break;

    // Now grow and update OutlininigInfo:
    OutliningInfo->Entries.push_back(Cand);
    OutliningInfo->NonReturnBlock = NonReturnBlock;
    OutliningInfo->ReturnBlockPreds.push_back(Cand);
    Entries.insert(Cand);
  }

  return OutliningInfo;
}

bool PartialInlinerImpl::shouldPartialInline(CallSite CS,
                                             OptimizationRemarkEmitter &ORE) {
  // TODO : more sharing with shouldInline in Inliner.cpp
  using namespace ore;
  Instruction *Call = CS.getInstruction();
  Function *Callee = CS.getCalledFunction();
  Function *Caller = CS.getCaller();
  auto &CalleeTTI = (*GetTTI)(*Callee);
  InlineCost IC = getInlineCost(CS, getInlineParams(), CalleeTTI,
                                *GetAssumptionCache, GetBFI, PSI);

  if (IC.isAlways()) {
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "AlwaysInline", Call)
             << NV("Callee", Callee)
             << " should always be fully inlined, not partially");
    return false;
  }

  if (IC.isNever()) {
    ORE.emit(OptimizationRemarkMissed(DEBUG_TYPE, "NeverInline", Call)
             << NV("Callee", Callee) << " not partially inlined into "
             << NV("Caller", Caller)
             << " because it should never be inlined (cost=never)");
    return false;
  }

  if (!IC) {
    ORE.emit(OptimizationRemarkMissed(DEBUG_TYPE, "TooCostly", Call)
             << NV("Callee", Callee) << " not partially inlined into "
             << NV("Caller", Caller) << " because too costly to inline (cost="
             << NV("Cost", IC.getCost()) << ", threshold="
             << NV("Threshold", IC.getCostDelta() + IC.getCost()) << ")");
    return false;
  }

  ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "CanBePartiallyInlined", Call)
           << NV("Callee", Callee) << " can be partially inlined into "
           << NV("Caller", Caller) << " with cost=" << NV("Cost", IC.getCost())
           << " (threshold="
           << NV("Threshold", IC.getCostDelta() + IC.getCost()) << ")");
  return true;
}

Function *PartialInlinerImpl::unswitchFunction(Function *F) {

  if (F->hasAddressTaken())
    return nullptr;

  std::unique_ptr<FunctionOutliningInfo> OutliningInfo =
      computeOutliningInfo(F);

  if (!OutliningInfo)
    return nullptr;

  // Clone the function, so that we can hack away on it.
  ValueToValueMapTy VMap;
  Function *DuplicateFunction = CloneFunction(F, VMap);
  BasicBlock *NewReturnBlock =
      cast<BasicBlock>(VMap[OutliningInfo->ReturnBlock]);
  BasicBlock *NewNonReturnBlock =
      cast<BasicBlock>(VMap[OutliningInfo->NonReturnBlock]);
  DenseSet<BasicBlock *> NewEntries;
  for (BasicBlock *BB : OutliningInfo->Entries) {
    NewEntries.insert(cast<BasicBlock>(VMap[BB]));
  }

  // Go ahead and update all uses to the duplicate, so that we can just
  // use the inliner functionality when we're done hacking.
  F->replaceAllUsesWith(DuplicateFunction);

  auto getFirstPHI = [](BasicBlock *BB) {
    BasicBlock::iterator I = BB->begin();
    PHINode *FirstPhi = nullptr;
    while (I != BB->end()) {
      PHINode *Phi = dyn_cast<PHINode>(I);
      if (!Phi)
        break;
      if (!FirstPhi) {
        FirstPhi = Phi;
        break;
      }
    }
    return FirstPhi;
  };
  // Special hackery is needed with PHI nodes that have inputs from more than
  // one extracted block.  For simplicity, just split the PHIs into a two-level
  // sequence of PHIs, some of which will go in the extracted region, and some
  // of which will go outside.
  BasicBlock *PreReturn = NewReturnBlock;
  // only split block when necessary:
  PHINode *FirstPhi = getFirstPHI(PreReturn);
  unsigned NumPredsFromEntries = OutliningInfo->ReturnBlockPreds.size();
  if (FirstPhi && FirstPhi->getNumIncomingValues() > NumPredsFromEntries + 1) {

    NewReturnBlock = NewReturnBlock->splitBasicBlock(
        NewReturnBlock->getFirstNonPHI()->getIterator());
    BasicBlock::iterator I = PreReturn->begin();
    Instruction *Ins = &NewReturnBlock->front();
    while (I != PreReturn->end()) {
      PHINode *OldPhi = dyn_cast<PHINode>(I);
      if (!OldPhi)
        break;

      PHINode *RetPhi =
          PHINode::Create(OldPhi->getType(), NumPredsFromEntries + 1, "", Ins);
      OldPhi->replaceAllUsesWith(RetPhi);
      Ins = NewReturnBlock->getFirstNonPHI();

      RetPhi->addIncoming(&*I, PreReturn);
      for (BasicBlock *E : OutliningInfo->ReturnBlockPreds) {
        BasicBlock *NewE = cast<BasicBlock>(VMap[E]);
        RetPhi->addIncoming(OldPhi->getIncomingValueForBlock(NewE), NewE);
        OldPhi->removeIncomingValue(NewE);
      }
      ++I;
    }
    for (auto E : OutliningInfo->ReturnBlockPreds) {
      BasicBlock *NewE = cast<BasicBlock>(VMap[E]);
      NewE->getTerminator()->replaceUsesOfWith(PreReturn, NewReturnBlock);
    }
  }

  // Returns true if the block is to be partial inlined into the caller
  // (i.e. not to be extracted to the out of line function)
  auto ToBeInlined = [=](BasicBlock *BB) {
    return BB == NewReturnBlock || NewEntries.count(BB);
  };
  // Gather up the blocks that we're going to extract.
  std::vector<BasicBlock *> ToExtract;
  ToExtract.push_back(NewNonReturnBlock);
  for (BasicBlock &BB : *DuplicateFunction)
    if (!ToBeInlined(&BB) && &BB != NewNonReturnBlock)
      ToExtract.push_back(&BB);

  // The CodeExtractor needs a dominator tree.
  DominatorTree DT;
  DT.recalculate(*DuplicateFunction);

  // Manually calculate a BlockFrequencyInfo and BranchProbabilityInfo.
  LoopInfo LI(DT);
  BranchProbabilityInfo BPI(*DuplicateFunction, LI);
  BlockFrequencyInfo BFI(*DuplicateFunction, BPI, LI);

  // Extract the body of the if.
  Function *ExtractedFunction =
      CodeExtractor(ToExtract, &DT, /*AggregateArgs*/ false, &BFI, &BPI)
          .extractCodeRegion();

  // Inline the top-level if test into all callers.
  std::vector<User *> Users(DuplicateFunction->user_begin(),
                            DuplicateFunction->user_end());

  for (User *User : Users) {
    CallSite CS;
    if (CallInst *CI = dyn_cast<CallInst>(User))
      CS = CallSite(CI);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(User))
      CS = CallSite(II);
    else
      llvm_unreachable("All uses must be calls");

    if (IsLimitReached())
      continue;

    OptimizationRemarkEmitter ORE(CS.getCaller());
    if (!shouldPartialInline(CS, ORE))
      continue;

    DebugLoc DLoc = CS.getInstruction()->getDebugLoc();
    BasicBlock *Block = CS.getParent();
    ORE.emit(OptimizationRemark(DEBUG_TYPE, "PartiallyInlined", DLoc, Block)
             << ore::NV("Callee", F) << " partially inlined into "
             << ore::NV("Caller", CS.getCaller()));

    InlineFunctionInfo IFI(nullptr, GetAssumptionCache);
    InlineFunction(CS, IFI);
    NumPartialInlining++;
    // update stats
    NumPartialInlined++;
  }

  // Ditch the duplicate, since we're done with it, and rewrite all remaining
  // users (function pointers, etc.) back to the original function.
  DuplicateFunction->replaceAllUsesWith(F);
  DuplicateFunction->eraseFromParent();


  return ExtractedFunction;
}

bool PartialInlinerImpl::run(Module &M) {
  if (DisablePartialInlining)
    return false;

  std::vector<Function *> Worklist;
  Worklist.reserve(M.size());
  for (Function &F : M)
    if (!F.use_empty() && !F.isDeclaration())
      Worklist.push_back(&F);

  bool Changed = false;
  while (!Worklist.empty()) {
    Function *CurrFunc = Worklist.back();
    Worklist.pop_back();

    if (CurrFunc->use_empty())
      continue;

    bool Recursive = false;
    for (User *U : CurrFunc->users())
      if (Instruction *I = dyn_cast<Instruction>(U))
        if (I->getParent()->getParent() == CurrFunc) {
          Recursive = true;
          break;
        }
    if (Recursive)
      continue;

    if (Function *NewFunc = unswitchFunction(CurrFunc)) {
      Worklist.push_back(NewFunc);
      Changed = true;
    }
  }

  return Changed;
}

char PartialInlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PartialInlinerLegacyPass, "partial-inliner",
                      "Partial Inliner", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PartialInlinerLegacyPass, "partial-inliner",
                    "Partial Inliner", false, false)

ModulePass *llvm::createPartialInliningPass() {
  return new PartialInlinerLegacyPass();
}

PreservedAnalyses PartialInlinerPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  std::function<AssumptionCache &(Function &)> GetAssumptionCache =
      [&FAM](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };

  std::function<BlockFrequencyInfo &(Function &)> GetBFI =
      [&FAM](Function &F) -> BlockFrequencyInfo & {
    return FAM.getResult<BlockFrequencyAnalysis>(F);
  };

  std::function<TargetTransformInfo &(Function &)> GetTTI =
      [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  ProfileSummaryInfo *PSI = &AM.getResult<ProfileSummaryAnalysis>(M);

  if (PartialInlinerImpl(&GetAssumptionCache, &GetTTI, {GetBFI}, PSI).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
