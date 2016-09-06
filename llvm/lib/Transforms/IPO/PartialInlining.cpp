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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
using namespace llvm;

#define DEBUG_TYPE "partialinlining"

STATISTIC(NumPartialInlined, "Number of functions partially inlined");

namespace {
struct PartialInlinerImpl {
  PartialInlinerImpl(InlineFunctionInfo IFI) : IFI(IFI) {}
  bool run(Module &M);
  Function *unswitchFunction(Function *F);

private:
  InlineFunctionInfo IFI;
};
struct PartialInlinerLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  PartialInlinerLegacyPass() : ModulePass(ID) {
    initializePartialInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
  }
  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    AssumptionCacheTracker *ACT = &getAnalysis<AssumptionCacheTracker>();
    std::function<AssumptionCache &(Function &)> GetAssumptionCache =
        [&ACT](Function &F) -> AssumptionCache & {
      return ACT->getAssumptionCache(F);
    };
    InlineFunctionInfo IFI(nullptr, &GetAssumptionCache);
    return PartialInlinerImpl(IFI).run(M);
  }
};
}

Function *PartialInlinerImpl::unswitchFunction(Function *F) {
  // First, verify that this function is an unswitching candidate...
  BasicBlock *EntryBlock = &F->front();
  BranchInst *BR = dyn_cast<BranchInst>(EntryBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return nullptr;

  BasicBlock *ReturnBlock = nullptr;
  BasicBlock *NonReturnBlock = nullptr;
  unsigned ReturnCount = 0;
  for (BasicBlock *BB : successors(EntryBlock)) {
    if (isa<ReturnInst>(BB->getTerminator())) {
      ReturnBlock = BB;
      ReturnCount++;
    } else
      NonReturnBlock = BB;
  }

  if (ReturnCount != 1)
    return nullptr;

  // Clone the function, so that we can hack away on it.
  ValueToValueMapTy VMap;
  Function *DuplicateFunction = CloneFunction(F, VMap);
  DuplicateFunction->setLinkage(GlobalValue::InternalLinkage);
  BasicBlock *NewEntryBlock = cast<BasicBlock>(VMap[EntryBlock]);
  BasicBlock *NewReturnBlock = cast<BasicBlock>(VMap[ReturnBlock]);
  BasicBlock *NewNonReturnBlock = cast<BasicBlock>(VMap[NonReturnBlock]);

  // Go ahead and update all uses to the duplicate, so that we can just
  // use the inliner functionality when we're done hacking.
  F->replaceAllUsesWith(DuplicateFunction);

  // Special hackery is needed with PHI nodes that have inputs from more than
  // one extracted block.  For simplicity, just split the PHIs into a two-level
  // sequence of PHIs, some of which will go in the extracted region, and some
  // of which will go outside.
  BasicBlock *PreReturn = NewReturnBlock;
  NewReturnBlock = NewReturnBlock->splitBasicBlock(
      NewReturnBlock->getFirstNonPHI()->getIterator());
  BasicBlock::iterator I = PreReturn->begin();
  Instruction *Ins = &NewReturnBlock->front();
  while (I != PreReturn->end()) {
    PHINode *OldPhi = dyn_cast<PHINode>(I);
    if (!OldPhi)
      break;

    PHINode *RetPhi = PHINode::Create(OldPhi->getType(), 2, "", Ins);
    OldPhi->replaceAllUsesWith(RetPhi);
    Ins = NewReturnBlock->getFirstNonPHI();

    RetPhi->addIncoming(&*I, PreReturn);
    RetPhi->addIncoming(OldPhi->getIncomingValueForBlock(NewEntryBlock),
                        NewEntryBlock);
    OldPhi->removeIncomingValue(NewEntryBlock);

    ++I;
  }
  NewEntryBlock->getTerminator()->replaceUsesOfWith(PreReturn, NewReturnBlock);

  // Gather up the blocks that we're going to extract.
  std::vector<BasicBlock *> ToExtract;
  ToExtract.push_back(NewNonReturnBlock);
  for (BasicBlock &BB : *DuplicateFunction)
    if (&BB != NewEntryBlock && &BB != NewReturnBlock &&
        &BB != NewNonReturnBlock)
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
  for (User *User : Users)
    if (CallInst *CI = dyn_cast<CallInst>(User))
      InlineFunction(CI, IFI);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(User))
      InlineFunction(II, IFI);

  // Ditch the duplicate, since we're done with it, and rewrite all remaining
  // users (function pointers, etc.) back to the original function.
  DuplicateFunction->replaceAllUsesWith(F);
  DuplicateFunction->eraseFromParent();

  ++NumPartialInlined;

  return ExtractedFunction;
}

bool PartialInlinerImpl::run(Module &M) {
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
  InlineFunctionInfo IFI(nullptr, &GetAssumptionCache);
  if (PartialInlinerImpl(IFI).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
