#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include <cstdint>

namespace llvm {

class Module;

struct DefUseInstrumentationPass
    : PassInfoMixin<DefUseInstrumentationPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {

    LLVMContext& Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    FunctionType* HookType = FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt64Ty(Ctx)}, false);
    FunctionCallee Hook =  M.getOrInsertFunction("__def_use_trace_enter", HookType);

    uint64_t CallID = 0;

    for (Function &F : M) {
      if (F.isDeclaration()) {
        continue;
      }
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          Builder.SetInsertPoint(&I);
          Builder.CreateCall(Hook, Builder.getInt64(CallID));
          CallID++;
        }
      }
    }
    
    return PreservedAnalyses::none();
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_DEFUSEINSTRUMENTATION_H