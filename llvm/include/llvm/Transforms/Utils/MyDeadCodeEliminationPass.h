#ifndef LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H
#define LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"

namespace llvm {

class MyDeadCodeEliminationPass : public PassInfoMixin<MyDeadCodeEliminationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  void analyzeInstructions(Function &F);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

