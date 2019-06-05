
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
struct SpawnUnswitch : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SpawnUnswitch() : FunctionPass(ID) {
    //initializeSpawnUnswitchPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    F.setName("SpawnUnswitch_"+F.getName());


    bool effective;
    do {
      effective = false;
      BasicBlock* body = nullptr;
      BasicBlock* end = nullptr;

      for (BasicBlock &BB: F) {
        if (BB.size() == 1 && isa<ReattachInst>(BB.getTerminator())) {
          end = BB.getSingleSuccessor();
          int count = 0;
          for (BasicBlock *Pred : predecessors(&BB)) {
            for (BasicBlock *PredPred : predecessors(Pred)) {
              if (!isa<DetachInst>(PredPred->getTerminator())) {
                body = Pred;
              }
            }
            count++;
          }
          if (count == 2) { // only predecessors are det.achd and if.then
            for (BasicBlock *Pred : predecessors(&BB)) {
              if (Pred->size() == 2 && isa<BranchInst>(Pred->getTerminator())) { // if clause only compares register contents
                Instruction* cmp = nullptr;
                for (Instruction &I : *Pred) {
                  cmp = &I;
                  break;
                }
                for (BasicBlock *PredPred : predecessors(Pred)) {
                  if (DetachInst *DI = dyn_cast<DetachInst>(PredPred->getTerminator())) { // outer spawn
                    Value *SyncRegion = DI->getSyncRegion();
                    effective = true;
                    // move cmp instruction to outside spawn
                    Instruction *pi = PredPred->getTerminator();
                    cmp->moveBefore(pi);

                    // branch now to detach or end
                    Instruction* temp = Pred->getTerminator();
                    BranchInst* replaceDetach = BranchInst::Create(Pred, end, ((BranchInst*)temp)->getCondition());
                    ReplaceInstWithInst(PredPred->getTerminator(), replaceDetach);

                    // detach now goes straight to body
                    DetachInst* newDetach = DetachInst::Create(body, end, SyncRegion);
                    ReplaceInstWithInst(Pred->getTerminator(), newDetach);
                  }
                }
              }
            }
          }
        }
      }
    } while (effective);

    return true;
  }
};
}

char SpawnUnswitch::ID = 0;
static RegisterPass<SpawnUnswitch> X("spawnunswitch", "Do SpawnUnswitch pass", false, false);

// Public interface to the RedundantSpawn pass
FunctionPass *llvm::createSpawnUnswitchPass() {
  return new SpawnUnswitch();
}
