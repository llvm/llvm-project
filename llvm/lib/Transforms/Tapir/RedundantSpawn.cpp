
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/CFG.h"

using namespace llvm;

namespace {
struct RedundantSpawn : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  RedundantSpawn() : FunctionPass(ID) {
    //initializeRedundantSpawnPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    F.setName("RedundantSpawn_"+F.getName());

    bool effective = false;
    do {
      effective = false;
      TerminatorInst* prior = nullptr;
      BasicBlock* start = nullptr;
      bool lookForDetach = false;
      int rank = 0;
      for (BasicBlock &BB: F) {
        if (isa<ReattachInst>(BB.getTerminator()) && BB.size() == 1) {
          lookForDetach = true;
          start = &BB;
          effective = true;
          break;
        }
        if (prior != nullptr && isa<DetachInst>(prior))
          rank +=1;
        if (prior != nullptr && isa<ReattachInst>(prior))
          rank -=1;
        prior = BB.getTerminator();
      }
      if (lookForDetach) {
        BasicBlock* current = start;
        int currentRank = rank;
        while (true) {
          for (BasicBlock *Pred : predecessors(current)) {
            current = Pred;
            break;
          }
          if (isa<DetachInst>(current->getTerminator()) && currentRank == rank) {
            BranchInst* replaceReattach = BranchInst::Create(start->getSingleSuccessor());
            BranchInst* replaceDetach = BranchInst::Create(current->getTerminator()->getSuccessor(0));
            ReplaceInstWithInst(start->getTerminator(), replaceReattach);
            ReplaceInstWithInst(current->getTerminator(), replaceDetach);
            break;
          }
          if (isa<DetachInst>(current->getTerminator()))
            currentRank -= 1;
          if (isa<ReattachInst>(current->getTerminator()))
            currentRank += 1;
        }
      }
    } while (effective);

    return true;
  }
};
}

char RedundantSpawn::ID = 0;
static RegisterPass<RedundantSpawn> X("redundantspawn", "Do RedundantSpawn pass", false, false);

// Public interface to the RedundantSpawn pass
FunctionPass *llvm::createRedundantSpawnPass() {
  return new RedundantSpawn();
}
