
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
struct SpawnRestructure : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SpawnRestructure() : FunctionPass(ID) {
    //initializeSpawnRestructurePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    F.setName("SpawnRestructure_"+F.getName());

    for (BasicBlock &BB: F) {

    }

    return true;
  }
};
}

char SpawnRestructure::ID = 0;
static RegisterPass<SpawnRestructure> X("spawnrestructure", "Do SpawnRestructure pass", false, false);

// Public interface to the RedundantSpawn pass
FunctionPass *llvm::createSpawnRestructurePass() {
  return new SpawnRestructure();
}
