//===- SplitBlocksAfterCalls.cpp -===//
//
// Makes function calls effectively terminators by splitting blocks after each
// call. This ensures that there can only be at most one call per block. This
// is used in order to detect recursion and external function calls within a
// trace.

#include "llvm/Transforms/Yk/SplitBlocksAfterCalls.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/LivenessAnalysis.h"

#include <map>

#define DEBUG_TYPE "yk-splitblocksaftercalls"

using namespace llvm;

namespace llvm {
void initializeYkSplitBlocksAfterCallsPass(PassRegistry &);
} // namespace llvm

namespace {

class YkSplitBlocksAfterCalls : public ModulePass {
public:
  static char ID;
  YkSplitBlocksAfterCalls() : ModulePass(ID) {
    initializeYkSplitBlocksAfterCallsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();

    for (Function &F : M) {
      if (F.empty()) // skip declarations.
        continue;
      // As we will be modifying the blocks of this function inplace, we
      // require a work list to process all existing and newly inserted blocks
      // in order to not miss any.
      std::vector<BasicBlock *> Todo;
      std::set<BasicBlock *> Seen;
      BasicBlock &Entry = F.getEntryBlock();

      // This pass requires the `NoCallsInEntryBlocksPass` to have run first,
      // which in turn needs to run before the shadowstack pass. Otherwise,
      // this pass would split the block after the shadowstack malloc, which
      // results in allocas outside of the entry block which breaks stackmaps.
      Instruction *T = Entry.getTerminator();
      for (size_t I = 0; I < T->getNumSuccessors(); I++) {
        Todo.push_back(T->getSuccessor(I));
      }
      Seen.insert(&Entry);
      while (!Todo.empty()) {
        BasicBlock *Next = Todo.back();
        Todo.pop_back();
        if (Seen.count(Next) > 0) {
          continue;
        }
        Seen.insert(Next);

        for (Instruction &I : *Next) {
          if (I.isDebugOrPseudoInst()) {
            continue;
          }
          if (isa<CallInst>(I)) {
            // YKFIXME: Can we determine at compile time if inline asm contains
            // calls or jumps, e.g. via `getAsmString`, and then not split the
            // block after them?
            CallInst *CI = cast<CallInst>(&I);
            Function *F = CI->getCalledFunction();
            if (F && F->getName() == "llvm.frameaddress.p0") {
              // This call is always inlined so we don't need to split the
              // block here.
              continue;
            }
            // YKFIXME: If the next instruction is an unconditional branch, we
            // don't need to split the block here.

            // Since `splitBasicBlock` splits before the given instruction,
            // pass the instruction following this call instead.
            Next->splitBasicBlock(I.getNextNode());
            break;
          }
        }

        // Add successors to todo list.
        Instruction *T = Next->getTerminator();
        for (size_t I = 0; I < T->getNumSuccessors(); I++) {
          Todo.insert(Todo.begin(), T->getSuccessor(I));
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

char YkSplitBlocksAfterCalls::ID = 0;
INITIALIZE_PASS(YkSplitBlocksAfterCalls, DEBUG_TYPE, "yk stackmaps", false,
                false)

ModulePass *llvm::createYkSplitBlocksAfterCallsPass() {
  return new YkSplitBlocksAfterCalls();
}
