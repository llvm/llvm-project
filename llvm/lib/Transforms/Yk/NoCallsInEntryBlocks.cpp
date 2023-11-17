// NoCallsInEntryBlocks.cpp - Ensure there are no calls in the first block.
//
// This pass splits up the entry block so that there are no calls inside it
// (one exception being the malloc call inserted by the shadowstack pass which
// will be applied later). This allows us to detect call backs from external
// functions more easily, which in turn simplifies our JITModBuilder
// implementation and fixes the soundness issue that existed in the old
// StackAdjust approach.
//
// As a simplified example, imagine this function:
//
// void foo(int argc) {
//    if (argc == 0) {
//      extfunc(foo)
//    }
// }
//
// where `extfunc` may or may not call back to `foo`. Previously, this might
// result in a trace like
//
// foo.0
// extfunc
// foo.0
// ...
//
// Upon seeing the second `foo.0` we are unable to determine whether we are
// observing a callback to `foo` or are returning from `extfunc` which hasn't
// called foo. When running this pass we force `extfunc` not to be in the entry
// block, resulting in a trace like
//
// foo.0
// foo.1
// extfunc
// foo.1
// ...
//
// Here it is clear that `extfunc` has not called `foo` as we are seeing block
// `foo.1` next. If `extfunc` had called `foo`, the next block would be
// `foo.0`.

#include "llvm/Transforms/Yk/NoCallsInEntryBlocks.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/LivenessAnalysis.h"

#define DEBUG_TYPE "yk-no-more-entryblock-calls"

using namespace llvm;

namespace llvm {
void initializeYkNoCallsInEntryBlocksPass(PassRegistry &);
} // namespace llvm

namespace {

class YkNoCallsInEntryBlocks : public ModulePass {
public:
  static char ID;
  YkNoCallsInEntryBlocks() : ModulePass(ID) {
    initializeYkNoCallsInEntryBlocksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    // split block at first function call
    std::set<Instruction *> Calls;
    for (Function &F : M) {
      if (F.empty()) // skip declarations.
        continue;
      BasicBlock &BB = F.getEntryBlock();
      auto CurrFuncName = F.getName();
      for (Instruction &I : BB) {
        if (isa<CallInst>(I)) {
          Function *F = cast<CallInst>(I).getCalledFunction();
          BB.splitBasicBlock(&I);
          break;
        }
      }
    }

#ifndef NDEBUG
    // Our pass runs after LLVM normally does its verify pass. In debug builds
    // we run it again to check that our pass is generating valid IR.
    if (verifyModule(M, &errs())) {
      Context.emitError("Stackmap insertion pass generated invalid IR!");
      return false;
    }
#endif
    return true;
  }
};
} // namespace

char YkNoCallsInEntryBlocks::ID = 0;
INITIALIZE_PASS(YkNoCallsInEntryBlocks, DEBUG_TYPE, "no more entry block calls",
                false, false)

ModulePass *llvm::createYkNoCallsInEntryBlocksPass() {
  return new YkNoCallsInEntryBlocks();
}
