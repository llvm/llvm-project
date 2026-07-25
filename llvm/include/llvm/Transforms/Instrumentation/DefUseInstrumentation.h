#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace llvm {

class Module;

struct DefUseInstrumentationPass
    : PassInfoMixin<DefUseInstrumentationPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    Function* Main = M.getFunction("main");
    if (!Main || Main->isDeclaration()) {
      return PreservedAnalyses::all();
    }
    LLVMContext& Ctx = M.getContext();

    FunctionType* HookType = FunctionType::get(Type::getVoidTy(Ctx), false);

    FunctionCallee funccall =  M.getOrInsertFunction("__def_use_trace_main_enter", HookType);

    return PreservedAnalyses::none();
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H