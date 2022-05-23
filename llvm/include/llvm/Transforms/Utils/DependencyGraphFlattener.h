#ifndef LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H
#define LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DependencyGraphFlattenerPass : public PassInfoMixin<DependencyGraphFlattenerPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H