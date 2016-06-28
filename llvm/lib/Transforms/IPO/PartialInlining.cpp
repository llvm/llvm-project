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
struct PartialInlinerLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  PartialInlinerLegacyPass() : ModulePass(ID) {
    initializePartialInlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    ModuleAnalysisManager DummyMAM;
    auto PA = Impl.run(M, DummyMAM);
    return !PA.areAllPreserved();
  }

private:
  PartialInlinerPass Impl;
  };
}

char PartialInlinerLegacyPass::ID = 0;
INITIALIZE_PASS(PartialInlinerLegacyPass, "partial-inliner", "Partial Inliner",
                false, false)

ModulePass *llvm::createPartialInliningPass() {
  return new PartialInlinerLegacyPass();
}

Function *PartialInlinerPass::unswitchFunction(Function *F) {
  // First, verify that this function is an unswitching candidate...
  BasicBlock *entryBlock = &F->front();
  BranchInst *BR = dyn_cast<BranchInst>(entryBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return nullptr;
  
  BasicBlock* returnBlock = nullptr;
  BasicBlock* nonReturnBlock = nullptr;
  unsigned returnCount = 0;
  for (BasicBlock *BB : successors(entryBlock)) {
    if (isa<ReturnInst>(BB->getTerminator())) {
      returnBlock = BB;
      returnCount++;
    } else
      nonReturnBlock = BB;
  }
  
  if (returnCount != 1)
    return nullptr;
  
  // Clone the function, so that we can hack away on it.
  ValueToValueMapTy VMap;
  Function* duplicateFunction = CloneFunction(F, VMap);
  duplicateFunction->setLinkage(GlobalValue::InternalLinkage);
  BasicBlock* newEntryBlock = cast<BasicBlock>(VMap[entryBlock]);
  BasicBlock* newReturnBlock = cast<BasicBlock>(VMap[returnBlock]);
  BasicBlock* newNonReturnBlock = cast<BasicBlock>(VMap[nonReturnBlock]);
  
  // Go ahead and update all uses to the duplicate, so that we can just
  // use the inliner functionality when we're done hacking.
  F->replaceAllUsesWith(duplicateFunction);
  
  // Special hackery is needed with PHI nodes that have inputs from more than
  // one extracted block.  For simplicity, just split the PHIs into a two-level
  // sequence of PHIs, some of which will go in the extracted region, and some
  // of which will go outside.
  BasicBlock* preReturn = newReturnBlock;
  newReturnBlock = newReturnBlock->splitBasicBlock(
      newReturnBlock->getFirstNonPHI()->getIterator());
  BasicBlock::iterator I = preReturn->begin();
  Instruction *Ins = &newReturnBlock->front();
  while (I != preReturn->end()) {
    PHINode* OldPhi = dyn_cast<PHINode>(I);
    if (!OldPhi) break;

    PHINode *retPhi = PHINode::Create(OldPhi->getType(), 2, "", Ins);
    OldPhi->replaceAllUsesWith(retPhi);
    Ins = newReturnBlock->getFirstNonPHI();

    retPhi->addIncoming(&*I, preReturn);
    retPhi->addIncoming(OldPhi->getIncomingValueForBlock(newEntryBlock),
                        newEntryBlock);
    OldPhi->removeIncomingValue(newEntryBlock);
    
    ++I;
  }
  newEntryBlock->getTerminator()->replaceUsesOfWith(preReturn, newReturnBlock);
  
  // Gather up the blocks that we're going to extract.
  std::vector<BasicBlock*> toExtract;
  toExtract.push_back(newNonReturnBlock);
  for (BasicBlock &BB : *duplicateFunction)
    if (&BB != newEntryBlock && &BB != newReturnBlock &&
        &BB != newNonReturnBlock)
      toExtract.push_back(&BB);

  // The CodeExtractor needs a dominator tree.
  DominatorTree DT;
  DT.recalculate(*duplicateFunction);

  // Extract the body of the if.
  Function* extractedFunction
    = CodeExtractor(toExtract, &DT).extractCodeRegion();
  
  InlineFunctionInfo IFI;
  
  // Inline the top-level if test into all callers.
  std::vector<User *> Users(duplicateFunction->user_begin(),
                            duplicateFunction->user_end());
  for (User *User : Users)
    if (CallInst *CI = dyn_cast<CallInst>(User))
      InlineFunction(CI, IFI);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(User))
      InlineFunction(II, IFI);
  
  // Ditch the duplicate, since we're done with it, and rewrite all remaining
  // users (function pointers, etc.) back to the original function.
  duplicateFunction->replaceAllUsesWith(F);
  duplicateFunction->eraseFromParent();
  
  ++NumPartialInlined;
  
  return extractedFunction;
}

PreservedAnalyses PartialInlinerPass::run(Module &M, ModuleAnalysisManager &) {
  std::vector<Function*> worklist;
  worklist.reserve(M.size());
  for (Function &F : M)
    if (!F.use_empty() && !F.isDeclaration())
      worklist.push_back(&F);

  bool changed = false;
  while (!worklist.empty()) {
    Function* currFunc = worklist.back();
    worklist.pop_back();
  
    if (currFunc->use_empty()) continue;
    
    bool recursive = false;
    for (User *U : currFunc->users())
      if (Instruction* I = dyn_cast<Instruction>(U))
        if (I->getParent()->getParent() == currFunc) {
          recursive = true;
          break;
        }
    if (recursive) continue;
          
    
    if (Function* newFunc = unswitchFunction(currFunc)) {
      worklist.push_back(newFunc);
      changed = true;
    }
    
  }

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
