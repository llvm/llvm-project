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

  StringRef LoggerFuncName = "_Z6Loggerll";

  FunctionCallee getCallLogFunc(Module &M) const;

  bool runOnModule(Module &M) override {
    // Prepare builder for IR modification
    LLVMContext &Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);
    FunctionCallee CallLogFunc = getCallLogFunc(M);

    // Traverse all Functions
    for (Function &F : M.functions()) {
      if (F.empty())
        continue;

      Builder.SetInsertPointPastAllocas(&F);
      Builder.CreateCall(CallLogFunc);
    }

    // TODO:
    // Need to define void function Logger for properly working in release mode
    // (without linking with our real Logger)
    // Also need make "linkonce" linkage type

    return true;
  }
};

FunctionCallee CallGraph::getCallLogFunc(Module &M) const {
  Type *RetType = Type::getVoidTy(M.getContext());
  FunctionType *CallLogFuncType = FunctionType::get(RetType, false);
  FunctionCallee CallLogFunc =
      M.getOrInsertFunction(LoggerFuncName, CallLogFuncType);

  return CallLogFunc;
}

} // namespace

char CallGraph::ID = 0;
static RegisterPass<CallGraph> X("callgraph", "CallGraph Pass");
