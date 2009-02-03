//===- LoopPass.cpp - Loop Pass and Loop Pass Manager ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LoopPass and LPPassManager. All loop optimization
// and transformation passes are derived from LoopPass. LPPassManager is
// responsible for managing LoopPasses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// LPPassManager
//

char LPPassManager::ID = 0;
/// LPPassManager manages FPPassManagers and CalLGraphSCCPasses.

LPPassManager::LPPassManager(int Depth) 
  : FunctionPass(&ID), PMDataManager(Depth) { 
  skipThisLoop = false;
  redoThisLoop = false;
  LI = NULL;
  CurrentLoop = NULL;
}

/// Delete loop from the loop queue and loop hierarchy (LoopInfo). 
void LPPassManager::deleteLoopFromQueue(Loop *L) {

  if (Loop *ParentLoop = L->getParentLoop()) { // Not a top-level loop.
    // Reparent all of the blocks in this loop.  Since BBLoop had a parent,
    // they are now all in it.
    for (Loop::block_iterator I = L->block_begin(), E = L->block_end(); 
         I != E; ++I)
      if (LI->getLoopFor(*I) == L)    // Don't change blocks in subloops.
        LI->changeLoopFor(*I, ParentLoop);
    
    // Remove the loop from its parent loop.
    for (Loop::iterator I = ParentLoop->begin(), E = ParentLoop->end();;
         ++I) {
      assert(I != E && "Couldn't find loop");
      if (*I == L) {
        ParentLoop->removeChildLoop(I);
        break;
      }
    }
    
    // Move all subloops into the parent loop.
    while (!L->empty())
      ParentLoop->addChildLoop(L->removeChildLoop(L->end()-1));
  } else {
    // Reparent all of the blocks in this loop.  Since BBLoop had no parent,
    // they no longer in a loop at all.
    
    for (unsigned i = 0; i != L->getBlocks().size(); ++i) {
      // Don't change blocks in subloops.
      if (LI->getLoopFor(L->getBlocks()[i]) == L) {
        LI->removeBlock(L->getBlocks()[i]);
        --i;
      }
    }

    // Remove the loop from the top-level LoopInfo object.
    for (LoopInfo::iterator I = LI->begin(), E = LI->end();; ++I) {
      assert(I != E && "Couldn't find loop");
      if (*I == L) {
        LI->removeLoop(I);
        break;
      }
    }

    // Move all of the subloops to the top-level.
    while (!L->empty())
      LI->addTopLevelLoop(L->removeChildLoop(L->end()-1));
  }

  delete L;

  // If L is current loop then skip rest of the passes and let
  // runOnFunction remove L from LQ. Otherwise, remove L from LQ now
  // and continue applying other passes on CurrentLoop.
  if (CurrentLoop == L) {
    skipThisLoop = true;
    return;
  }

  for (std::deque<Loop *>::iterator I = LQ.begin(),
         E = LQ.end(); I != E; ++I) {
    if (*I == L) {
      LQ.erase(I);
      break;
    }
  }
}

// Inset loop into loop nest (LoopInfo) and loop queue (LQ).
void LPPassManager::insertLoop(Loop *L, Loop *ParentLoop) {

  assert (CurrentLoop != L && "Cannot insert CurrentLoop");

  // Insert into loop nest
  if (ParentLoop)
    ParentLoop->addChildLoop(L);
  else
    LI->addTopLevelLoop(L);

  // Insert L into loop queue
  if (L == CurrentLoop) 
    redoLoop(L);
  else if (!ParentLoop)
    // This is top level loop. 
    LQ.push_front(L);
  else {
    // Insert L after ParentLoop
    for (std::deque<Loop *>::iterator I = LQ.begin(),
           E = LQ.end(); I != E; ++I) {
      if (*I == ParentLoop) {
        // deque does not support insert after.
        ++I;
        LQ.insert(I, 1, L);
        break;
      }
    }
  }
}

// Reoptimize this loop. LPPassManager will re-insert this loop into the
// queue. This allows LoopPass to change loop nest for the loop. This
// utility may send LPPassManager into infinite loops so use caution.
void LPPassManager::redoLoop(Loop *L) {
  assert (CurrentLoop == L && "Can redo only CurrentLoop");
  redoThisLoop = true;
}

/// cloneBasicBlockSimpleAnalysis - Invoke cloneBasicBlockAnalysis hook for
/// all loop passes.
void LPPassManager::cloneBasicBlockSimpleAnalysis(BasicBlock *From, 
                                                  BasicBlock *To, Loop *L) {
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    LoopPass *LP = dynamic_cast<LoopPass *>(P);
    LP->cloneBasicBlockAnalysis(From, To, L);
  }
}

/// deleteSimpleAnalysisValue - Invoke deleteAnalysisValue hook for all passes.
void LPPassManager::deleteSimpleAnalysisValue(Value *V, Loop *L) {
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    LoopPass *LP = dynamic_cast<LoopPass *>(P);
    LP->deleteAnalysisValue(V, L);
  }
}


// Recurse through all subloops and all loops  into LQ.
static void addLoopIntoQueue(Loop *L, std::deque<Loop *> &LQ) {
  LQ.push_back(L);
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);
}

/// Pass Manager itself does not invalidate any analysis info.
void LPPassManager::getAnalysisUsage(AnalysisUsage &Info) const {
  // LPPassManager needs LoopInfo. In the long term LoopInfo class will 
  // become part of LPPassManager.
  Info.addRequired<LoopInfo>();
  // Used by IndVar doInitialization.
  Info.addRequired<ScalarEvolution>();
  Info.setPreservesAll();
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the function, and if so, return true.
bool LPPassManager::runOnFunction(Function &F) {
  LI = &getAnalysis<LoopInfo>();
  bool Changed = false;

  // Collect inherited analysis from Module level pass manager.
  populateInheritedAnalysis(TPM->activeStack);

  // Populate Loop Queue
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);

  // Initialization
  for (std::deque<Loop *>::const_iterator I = LQ.begin(), E = LQ.end();
       I != E; ++I) {
    Loop *L = *I;
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
      Pass *P = getContainedPass(Index);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      if (LP)
        Changed |= LP->doInitialization(L, *this);
    }
  }

  // Walk Loops
  while (!LQ.empty()) {
      
    CurrentLoop  = LQ.back();
    skipThisLoop = false;
    redoThisLoop = false;

    // Run all passes on current SCC
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
        
      Pass *P = getContainedPass(Index);

      dumpPassInfo(P, EXECUTION_MSG, ON_LOOP_MSG, "");
      dumpRequiredSet(P);

      initializeAnalysisImpl(P);

      StartPassTimer(P);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      assert (LP && "Invalid LPPassManager member");
      Changed |= LP->runOnLoop(CurrentLoop, *this);
      StopPassTimer(P);

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_LOOP_MSG, "");
      dumpPreservedSet(P);

      verifyPreservedAnalysis(LP);
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, "", ON_LOOP_MSG);

      // If dominator information is available then verify the info if requested.
      verifyDomInfo(*LP, F);

      if (skipThisLoop)
        // Do not run other passes on this loop.
        break;
    }
    
    // Pop the loop from queue after running all passes.
    LQ.pop_back();
    
    if (redoThisLoop)
      LQ.push_back(CurrentLoop);
  }
  
  // Finalization
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    Pass *P = getContainedPass(Index);
    LoopPass *LP = dynamic_cast <LoopPass *>(P);
    if (LP)
      Changed |= LP->doFinalization();
  }

  return Changed;
}


//===----------------------------------------------------------------------===//
// LoopPass

// Check if this pass is suitable for the current LPPassManager, if
// available. This pass P is not suitable for a LPPassManager if P
// is not preserving higher level analysis info used by other
// LPPassManager passes. In such case, pop LPPassManager from the
// stack. This will force assignPassManager() to create new
// LPPassManger as expected.
void LoopPass::preparePassManager(PMStack &PMS) {

  // Find LPPassManager 
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_LoopPassManager)
    PMS.pop();

  LPPassManager *LPPM = dynamic_cast<LPPassManager *>(PMS.top());

  // If this pass is destroying high level information that is used
  // by other passes that are managed by LPM then do not insert
  // this pass in current LPM. Use new LPPassManager.
  if (LPPM && !LPPM->preserveHigherLevelAnalysis(this)) 
    PMS.pop();
}

/// Assign pass manager to manage this pass.
void LoopPass::assignPassManager(PMStack &PMS,
                                 PassManagerType PreferredType) {
  // Find LPPassManager 
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_LoopPassManager)
    PMS.pop();

  LPPassManager *LPPM = dynamic_cast<LPPassManager *>(PMS.top());

  // Create new Loop Pass Manager if it does not exist. 
  if (!LPPM) {

    assert (!PMS.empty() && "Unable to create Loop Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Call Graph Pass Manager
    LPPM = new LPPassManager(PMD->getDepth() + 1);
    LPPM->populateInheritedAnalysis(PMS);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(LPPM);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    Pass *P = dynamic_cast<Pass *>(LPPM);
    TPM->schedulePass(P);

    // [4] Push new manager into PMS
    PMS.push(LPPM);
  }

  LPPM->add(this);
}
