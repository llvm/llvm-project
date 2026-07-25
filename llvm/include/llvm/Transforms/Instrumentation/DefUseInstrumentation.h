#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"
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

    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);
    FunctionType* HookType = FunctionType::get(Type::getVoidTy(Ctx), false);
    FunctionCallee Hook =  M.getOrInsertFunction("__def_use_trace_enter", HookType);

    for (Function &F : M) {
      if (F.isDeclaration()) {
        continue;
      }
      Builder.SetInsertPointPastAllocas(&F);
      Builder.CreateCall(Hook);
    }
    
    return PreservedAnalyses::none();
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H