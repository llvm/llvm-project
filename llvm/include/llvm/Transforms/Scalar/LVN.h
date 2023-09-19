#ifndef LLVM_TRANSFORMS_SCALAR_LVN_H
#define LLVM_TRANSFORMS_SCALAR_LVN_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LVNPass : public PassInfoMixin<LVNPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  void runOnBasicBlock(BasicBlock &BB);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LVN_H
