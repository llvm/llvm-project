#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

#define DEBUG_TYPE "callGraph"

namespace {
/// Class for implemeting CallGraph building
struct CallGraph : ModulePass {
  static char ID;
  CallGraph() : ModulePass(ID) {}

  StringRef LoggerFuncName = "_Z6Loggerv";

  FunctionType *getLoggerFuncType(Module &M) const;
  FunctionCallee getCallLogFunc(Module &M) const;
  void createDummyLogger(Module &M, IRBuilder<> &Builder) const;

  bool runOnModule(Module &M) override {
    // Prepare builder for IR modification
    LLVMContext &Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);

    createDummyLogger(M, Builder);
    FunctionCallee CallLogFunc = getCallLogFunc(M);

    // Traverse all Functions
    for (Function &F : M.functions()) {
      if (F.empty())
        continue;
      StringRef FuncName = F.getName();
      if (FuncName == LoggerFuncName || FuncName == "main" ||
          FuncName.count("_GLOBAL_") || FuncName.count("global_var"))
        continue;

      Builder.SetInsertPointPastAllocas(&F);
      Builder.CreateCall(CallLogFunc);
    }

    return true;
  }
};

FunctionType *CallGraph::getLoggerFuncType(Module &M) const {
  Type *RetType = Type::getVoidTy(M.getContext());
  return FunctionType::get(RetType, false);
}

FunctionCallee CallGraph::getCallLogFunc(Module &M) const {
  return (M.getOrInsertFunction(LoggerFuncName, getLoggerFuncType(M)));
}

void CallGraph::createDummyLogger(Module &M, IRBuilder<> &Builder) const {
  // Use LinkOnceAnyLinkage linkage typem because we need to merge dummyLogger
  // with the actual one, when profiling. LinkOnce is better than weak in case
  // of profiing .cxx file with only global variables, because optimizer
  // can just cut off our dummy definition.
  Function *DummyLogger = Function::Create(
      getLoggerFuncType(M), Function::LinkOnceAnyLinkage, LoggerFuncName, M);

  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", DummyLogger);
  Builder.SetInsertPoint(BB);
  Builder.CreateRetVoid();
}

} // namespace

char CallGraph::ID = 0;
static RegisterPass<CallGraph> X("callgraph", "CallGraph Pass");
