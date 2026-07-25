#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

class Module;

struct DefUseInstrumentationPass
    : PassInfoMixin<DefUseInstrumentationPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    Function* Main = M.getFunction("main");
    if (!Main || Main->isDeclaration()) {
      return PreservedAnalyses::all();
    }
    return PreservedAnalyses::all();
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H