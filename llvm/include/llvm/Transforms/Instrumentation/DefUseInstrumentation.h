#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

struct DefUseInstrumentationPass
    : PassInfoMixin<DefUseInstrumentationPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif