#ifndef LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H
#define LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// IRNormalizer aims to transform LLVM IR into canonical form.
struct IRNormalizerPass : public PassInfoMixin<IRNormalizerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H
