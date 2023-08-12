#ifndef LLVM_TRANSFORMS_UTILS_IRCANONICALIZER_H
#define LLVM_TRANSFORMS_UTILS_IRCANONICALIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// IRCanonicalizer aims to transform LLVM IR into canonical form.
struct IRCanonicalizerPass : public PassInfoMixin<IRCanonicalizerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_IRCANONICALIZER_H
