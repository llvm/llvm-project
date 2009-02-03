//===- CallGraphSCCPass.cpp - Pass that operates BU on call graph ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CallGraphSCCPass class, which is used for passes
// which are implemented as bottom-up traversals on the call graph.  Because
// there may be cycles in the call graph, passes of this type operate on the
// call-graph in SCC order: that is, they process function bottom-up, except for
// recursive functions, which they process all at once.
//
//===----------------------------------------------------------------------===//

#include "llvm/CallGraphSCCPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// CGPassManager
//
/// CGPassManager manages FPPassManagers and CalLGraphSCCPasses.

namespace {

class CGPassManager : public ModulePass, public PMDataManager {

public:
  static char ID;
  explicit CGPassManager(int Depth) 
    : ModulePass(&ID), PMDataManager(Depth) { }

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);

  bool doInitialization(CallGraph &CG);
  bool doFinalization(CallGraph &CG);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    // CGPassManager walks SCC and it needs CallGraph.
    Info.addRequired<CallGraph>();
    Info.setPreservesAll();
  }

  virtual const char *getPassName() const {
    return "CallGraph Pass Manager";
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "Call Graph SCC Pass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      Pass *P = getContainedPass(Index);
      P->dumpPassStructure(Offset + 1);
      dumpLastUses(P, Offset+1);
    }
  }

  Pass *getContainedPass(unsigned N) {
    assert ( N < PassVector.size() && "Pass number out of range!");
    Pass *FP = static_cast<Pass *>(PassVector[N]);
    return FP;
  }

  virtual PassManagerType getPassManagerType() const { 
    return PMT_CallGraphPassManager; 
  }
};

}

char CGPassManager::ID = 0;
/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool CGPassManager::runOnModule(Module &M) {
  CallGraph &CG = getAnalysis<CallGraph>();
  bool Changed = doInitialization(CG);

  // Walk SCC
  for (scc_iterator<CallGraph*> I = scc_begin(&CG), E = scc_end(&CG);
       I != E; ++I) {

    // Run all passes on current SCC
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      Pass *P = getContainedPass(Index);

      dumpPassInfo(P, EXECUTION_MSG, ON_CG_MSG, "");
      dumpRequiredSet(P);

      initializeAnalysisImpl(P);

      StartPassTimer(P);
      if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass *>(P))
        Changed |= CGSP->runOnSCC(*I);   // TODO : What if CG is changed ?
      else {
        FPPassManager *FPP = dynamic_cast<FPPassManager *>(P);
        assert (FPP && "Invalid CGPassManager member");

        // Run pass P on all functions current SCC
        std::vector<CallGraphNode*> &SCC = *I;
        for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
          Function *F = SCC[i]->getFunction();
          if (F) {
            dumpPassInfo(P, EXECUTION_MSG, ON_FUNCTION_MSG, F->getNameStart());
            Changed |= FPP->runOnFunction(*F);
          }
        }
      }
      StopPassTimer(P);

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_CG_MSG, "");
      dumpPreservedSet(P);

      verifyPreservedAnalysis(P);      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, "", ON_CG_MSG);
    }
  }
  Changed |= doFinalization(CG);
  return Changed;
}

/// Initialize CG
bool CGPassManager::doInitialization(CallGraph &CG) {
  bool Changed = false;
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass *>(P)) 
      Changed |= CGSP->doInitialization(CG);
  }
  return Changed;
}

/// Finalize CG
bool CGPassManager::doFinalization(CallGraph &CG) {
  bool Changed = false;
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
    Pass *P = getContainedPass(Index);
    if (CallGraphSCCPass *CGSP = dynamic_cast<CallGraphSCCPass *>(P)) 
      Changed |= CGSP->doFinalization(CG);
  }
  return Changed;
}

/// Assign pass manager to manage this pass.
void CallGraphSCCPass::assignPassManager(PMStack &PMS,
                                         PassManagerType PreferredType) {
  // Find CGPassManager 
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_CallGraphPassManager)
    PMS.pop();

  assert (!PMS.empty() && "Unable to handle Call Graph Pass");
  CGPassManager *CGP = dynamic_cast<CGPassManager *>(PMS.top());

  // Create new Call Graph SCC Pass Manager if it does not exist. 
  if (!CGP) {

    assert (!PMS.empty() && "Unable to create Call Graph Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Call Graph Pass Manager
    CGP = new CGPassManager(PMD->getDepth() + 1);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(CGP);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    Pass *P = dynamic_cast<Pass *>(CGP);
    TPM->schedulePass(P);

    // [4] Push new manager into PMS
    PMS.push(CGP);
  }

  CGP->add(this);
}

/// getAnalysisUsage - For this class, we declare that we require and preserve
/// the call graph.  If the derived class implements this method, it should
/// always explicitly call the implementation here.
void CallGraphSCCPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraph>();
  AU.addPreserved<CallGraph>();
}
