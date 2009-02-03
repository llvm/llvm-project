//===- CloneLoop.cpp - Clone loop nest ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneLoop interface which makes a copy of a loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/BasicBlock.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/ADT/DenseMap.h"


using namespace llvm;

/// CloneDominatorInfo - Clone basicblock's dominator tree and, if available,
/// dominance info. It is expected that basic block is already cloned.
static void CloneDominatorInfo(BasicBlock *BB, 
                               DenseMap<const Value *, Value *> &ValueMap,
                               DominatorTree *DT,
                               DominanceFrontier *DF) {

  assert (DT && "DominatorTree is not available");
  DenseMap<const Value *, Value*>::iterator BI = ValueMap.find(BB);
  assert (BI != ValueMap.end() && "BasicBlock clone is missing");
  BasicBlock *NewBB = cast<BasicBlock>(BI->second);

  // NewBB already got dominator info.
  if (DT->getNode(NewBB))
    return;

  assert (DT->getNode(BB) && "BasicBlock does not have dominator info");
  // Entry block is not expected here. Infinite loops are not to cloned.
  assert (DT->getNode(BB)->getIDom() && "BasicBlock does not have immediate dominator");
  BasicBlock *BBDom = DT->getNode(BB)->getIDom()->getBlock();

  // NewBB's dominator is either BB's dominator or BB's dominator's clone.
  BasicBlock *NewBBDom = BBDom;
  DenseMap<const Value *, Value*>::iterator BBDomI = ValueMap.find(BBDom);
  if (BBDomI != ValueMap.end()) {
    NewBBDom = cast<BasicBlock>(BBDomI->second);
    if (!DT->getNode(NewBBDom))
      CloneDominatorInfo(BBDom, ValueMap, DT, DF);
  }
  DT->addNewBlock(NewBB, NewBBDom);

  // Copy cloned dominance frontiner set
  if (DF) {
    DominanceFrontier::DomSetType NewDFSet;
    DominanceFrontier::iterator DFI = DF->find(BB);
    if ( DFI != DF->end()) {
      DominanceFrontier::DomSetType S = DFI->second;
        for (DominanceFrontier::DomSetType::iterator I = S.begin(), E = S.end();
             I != E; ++I) {
          BasicBlock *DB = *I;
          DenseMap<const Value*, Value*>::iterator IDM = ValueMap.find(DB);
          if (IDM != ValueMap.end())
            NewDFSet.insert(cast<BasicBlock>(IDM->second));
          else
            NewDFSet.insert(DB);
        }
    }
    DF->addBasicBlock(NewBB, NewDFSet);
  }
}

/// CloneLoop - Clone Loop. Clone dominator info. Populate ValueMap
/// using old blocks to new blocks mapping.
Loop *llvm::CloneLoop(Loop *OrigL, LPPassManager  *LPM, LoopInfo *LI,
                      DenseMap<const Value *, Value *> &ValueMap, Pass *P) {
  
  DominatorTree *DT = NULL;
  DominanceFrontier *DF = NULL;
  if (P) {
    DT = P->getAnalysisIfAvailable<DominatorTree>();
    DF = P->getAnalysisIfAvailable<DominanceFrontier>();
  }

  SmallVector<BasicBlock *, 16> NewBlocks;

  // Populate loop nest.
  SmallVector<Loop *, 8> LoopNest;
  LoopNest.push_back(OrigL);


  Loop *NewParentLoop = NULL;
  while (!LoopNest.empty()) {
    Loop *L = LoopNest.back();
    LoopNest.pop_back();
    Loop *NewLoop = new Loop();

    if (!NewParentLoop)
      NewParentLoop = NewLoop;

    LPM->insertLoop(NewLoop, L->getParentLoop());

    // Clone Basic Blocks.
    for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
         I != E; ++I) {
      BasicBlock *BB = *I;
      BasicBlock *NewBB = CloneBasicBlock(BB, ValueMap, ".clone");
      ValueMap[BB] = NewBB;
      if (P)
        LPM->cloneBasicBlockSimpleAnalysis(BB, NewBB, L);
      NewLoop->addBasicBlockToLoop(NewBB, LI->getBase());
      NewBlocks.push_back(NewBB);
    }

    // Clone dominator info.
    if (DT)
      for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
           I != E; ++I) {
        BasicBlock *BB = *I;
        CloneDominatorInfo(BB, ValueMap, DT, DF);
      }

    // Process sub loops
    for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
      LoopNest.push_back(*I);
  }

  // Remap instructions to reference operands from ValueMap.
  for(SmallVector<BasicBlock *, 16>::iterator NBItr = NewBlocks.begin(), 
        NBE = NewBlocks.end();  NBItr != NBE; ++NBItr) {
    BasicBlock *NB = *NBItr;
    for(BasicBlock::iterator BI = NB->begin(), BE = NB->end(); 
        BI != BE; ++BI) {
      Instruction *Insn = BI;
      for (unsigned index = 0, num_ops = Insn->getNumOperands(); 
           index != num_ops; ++index) {
        Value *Op = Insn->getOperand(index);
        DenseMap<const Value *, Value *>::iterator OpItr = ValueMap.find(Op);
        if (OpItr != ValueMap.end())
          Insn->setOperand(index, OpItr->second);
      }
    }
  }

  BasicBlock *Latch = OrigL->getLoopLatch();
  Function *F = Latch->getParent();
  F->getBasicBlockList().insert(OrigL->getHeader(), 
                                NewBlocks.begin(), NewBlocks.end());


  return NewParentLoop;
}
