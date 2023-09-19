#include "llvm/Transforms/Scalar/LVN.h"

using namespace llvm;

PreservedAnalyses LVNPass::run(Function &F, FunctionAnalysisManager &FAM) {
  return PreservedAnalyses::all();
}
