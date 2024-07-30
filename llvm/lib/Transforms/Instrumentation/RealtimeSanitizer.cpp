#include "llvm/IR/Analysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "llvm/Transforms/Instrumentation/RealtimeSanitizer.h"

using namespace llvm;

namespace {

void insertCallBeforeInstruction(Function &Fn, Instruction &Instruction,
                                 const char *FunctionName) {
  LLVMContext &Context = Fn.getContext();
  FunctionType *FuncType = FunctionType::get(Type::getVoidTy(Context), false);
  FunctionCallee Func =
      Fn.getParent()->getOrInsertFunction(FunctionName, FuncType);
  IRBuilder<> Builder{&Instruction};
  Builder.CreateCall(Func, {});
}

void insertCallAtFunctionEntryPoint(Function &Fn, const char *InsertFnName) {

  insertCallBeforeInstruction(Fn, Fn.front().front(), InsertFnName);
}

void insertCallAtAllFunctionExitPoints(Function &Fn, const char *InsertFnName) {
  for (auto &BB : Fn) {
    for (auto &I : BB) {
      if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        insertCallBeforeInstruction(Fn, I, InsertFnName);
      }
    }
  }
}
} // namespace

RealtimeSanitizerPass::RealtimeSanitizerPass(
    const RealtimeSanitizerOptions &Options)
    : Options{Options} {}

PreservedAnalyses RealtimeSanitizerPass::run(Function &F,
                                             AnalysisManager<Function> &AM) {
  if (F.hasFnAttribute(Attribute::NonBlocking)) {
    insertCallAtFunctionEntryPoint(F, "__rtsan_realtime_enter");
    insertCallAtAllFunctionExitPoints(F, "__rtsan_realtime_exit");
    return PreservedAnalyses::none();
  }

  return PreservedAnalyses::all();
}
