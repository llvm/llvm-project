#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Yk/BasicBlockTracer.h"
#include "llvm/Transforms/Yk/ModuleClone.h"
#include "llvm/Analysis/AliasAnalysis.h"

#define DEBUG_TYPE "yk-basicblock-tracer-pass"

using namespace llvm;

namespace llvm {
void initializeYkBasicBlockTracerNoopPass(PassRegistry &);
} // namespace llvm

namespace {
struct YkBasicBlockTracerNoop : public ModulePass {
  static char ID;

  YkBasicBlockTracerNoop() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool Changed = false;
    // Iterate over each function in the module
    for (Function &F : M) {
      // Skip non cloned functions
      if (F.getName().startswith(YK_UNOPT_PREFIX) == false)
        continue;
      for (BasicBlock &BB : F) {
        for (auto it = BB.begin(); it != BB.end(); ) {
          Instruction *I = &*it++;
          if (auto *Call = dyn_cast<CallBase>(I)) {
            Function *CalledFunc = Call->getCalledFunction();
            if (!CalledFunc)
              continue; // Skip indirect calls
            if (CalledFunc->getName() == YK_TRACE_FUNCTION_DUMMY) {
              Call->eraseFromParent();
              Changed = true;
            }
          }
        }
      }
    }

    return Changed;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // AU.addPreserved<AAResultsWrapperPass>();
  }
};
} // namespace

char YkBasicBlockTracerNoop::ID = 0;

INITIALIZE_PASS(YkBasicBlockTracerNoop, DEBUG_TYPE, "yk basicblock tracer noop", false,
                false)

namespace llvm {
ModulePass *createYkBasicBlockTracerNoopPass() {
    return new YkBasicBlockTracerNoop();
}
} // namespace llvm