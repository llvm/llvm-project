#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"

using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  errs() << "I'm here in my Pass" << "\n";
  return PreservedAnalyses::all();
/* The PreservedAnalyses return value says that all analyses (e.g. dominator tree) are still valid after this pass since we didn’t modify any functions */
}
