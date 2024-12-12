#ifndef LLVM_TRANSFORMS_MYDEADCODEELIMINATIONPASS_H
#define LLVM_TRANSFORMS_MYDEADCODEELIMINATIONPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class MyDeadCodeEliminationPass : public PassInfoMixin<MyDeadCodeEliminationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_MYDEADCODEELIMINATIONPASS_H
