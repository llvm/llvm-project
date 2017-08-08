
#include "llvm/Transforms/Tapir.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {
struct SmallBlock : public FunctionPass {
  static const int threshold = 10;
  static char ID; // Pass identification, replacement for typeid
  SmallBlock() : FunctionPass(ID) {
    //initializeSmallBlockPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.addRequired<TargetTransformInfoWrapperPass>();
    //AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    F.setName("SmallBlock_"+F.getName());

    BasicBlock* b = nullptr;
    BasicBlock* prior = nullptr;
    bool effective;
    int count = 0;
    do {
      effective = false;
      for (BasicBlock &BB: F) {
        count += BB.size();
        if (isa<DetachInst>(BB.getTerminator())) {
          b = &BB;
          count = 0;
        }
        if (isa<ReattachInst>(BB.getTerminator()) && count < threshold && prior != b) {
          // b ensured to be the corresponding reattach
          effective = true;
          prior = b;
          BranchInst* replaceReattach = BranchInst::Create(BB.getSingleSuccessor());
          BranchInst* replaceDetach = BranchInst::Create(b->getTerminator()->getSuccessor(0));
          ReplaceInstWithInst(BB.getTerminator(), replaceReattach);
          ReplaceInstWithInst(b->getTerminator(), replaceDetach);
        }
      }
    } while (effective);

    return true;
  }
};
}

char SmallBlock::ID = 0;
static RegisterPass<SmallBlock> X("smallblock", "Do SmallBlock pass", false, false);

// Public interface to the SmallBlock pass
FunctionPass *llvm::createSmallBlockPass() {
  return new SmallBlock();
}
