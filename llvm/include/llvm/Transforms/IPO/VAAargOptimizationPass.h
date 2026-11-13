#ifndef LLVM_TRANSFORMS_IPO_VAARGOPTIMIZATIONPASS_H
#define LLVM_TRANSFORMS_IPO_VAARGOPTIMIZATIONPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class VAAargOptimizationPass
    : public OptionalPassInfoMixin<VAAargOptimizationPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_VAARGOPTIMIZATIONPASS_H
