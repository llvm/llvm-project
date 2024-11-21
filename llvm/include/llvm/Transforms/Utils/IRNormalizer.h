#ifndef LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H
#define LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// IRNormalizer aims to transform LLVM IR into normal form.
struct IRNormalizerPass : public PassInfoMixin<IRNormalizerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) const;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_IRNORMALIZER_H
