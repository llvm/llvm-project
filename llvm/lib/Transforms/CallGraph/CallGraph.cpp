#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "callGraph"

namespace {
void makeDotPrologue(raw_fd_ostream &File) { File << "digraph G {\n"; }
void makeDotEpilogue(raw_fd_ostream &File) { File << "}\n"; }

/// Class for implemeting CallGraph building
struct CallGraph : ModulePass {
  static char ID;
  CallGraph() : ModulePass(ID) { std::error_code EC; }

  // Edges map contains key=caller and value=set_of_calees
  std::unordered_map<const Function *, SmallPtrSet<const Function *, 8>> Edges;

  FunctionCallee getCallLogFunc(Module &M, IRBuilder<> &Builder) const;
  void insertCallLogger(IRBuilder<> &Builder, const Function *CallerFunc,
                        const Function *CalleeFunc,
                        FunctionCallee &LogFunc) const;
  bool isLoggerFunc(StringRef Name) const { return Name == "_Z6LoggerPclS_l"; }
  bool isLLVMTrap(StringRef Name) const { return Name == "llvm.trap"; }

  bool runOnModule(Module &M) override {
    std::error_code EC;
    // We use ostream, because while we collect static information, we
    // don't need to know previous file condition
    raw_fd_ostream File{"OutFile.txt", EC};
    assert(EC.value() == 0);

    makeDotPrologue(File);

    // Prepare builder for IR modification
    LLVMContext &Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);
    FunctionCallee CallLogFunc = getCallLogFunc(M, Builder);

    // Traverse all Functions
    for (Function &F : M.functions()) {
      auto &Bucket = Edges[&F];
      StringRef FuncName = F.getName();

      // Traverse all basic blocks
      for (BasicBlock &B : F) {
        // Traverse all Intructions
        for (Instruction &I : B) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            const Function *CalledFunc = Call->getCalledFunction();
            if (!CalledFunc)
              continue;

            Builder.SetInsertPoint(Call);
            insertCallLogger(Builder, &F, CalledFunc, CallLogFunc);

            // Here we will add static information, if we meet edge for the
            // first time
            if (Bucket.contains(CalledFunc))
              continue;

            // If we meet that edge for the first time, we need to add it to map
            // and print it to file
            File << FuncName << &F << " -> " << CalledFunc->getName()
                 << CalledFunc << '\n';
            Bucket.insert(CalledFunc);
          }
        }
      }
    }

    makeDotEpilogue(File);
    return true;
  }
};

FunctionCallee CallGraph::getCallLogFunc(Module &M,
                                         IRBuilder<> &Builder) const {
  Type *RetType = Type::getVoidTy(M.getContext());

  // Prepare callLogger function
  // Logger func will get 4 params: caller name + addr, callee name + addr
  SmallVector<Type *> CallParamType = {
      Builder.getInt8PtrTy(), Builder.getInt64Ty(), Builder.getInt8PtrTy(),
      Builder.getInt64Ty()};

  FunctionType *CallLogFuncType =
      FunctionType::get(RetType, CallParamType, false);
  FunctionCallee CallLogFunc = M.getOrInsertFunction("_Z6LoggerPclS_l", CallLogFuncType);

  return CallLogFunc;
}

void CallGraph::insertCallLogger(IRBuilder<> &Builder,
                                 const Function *CallerFunc,
                                 const Function *CalleeFunc,
                                 FunctionCallee &LogFunc) const {
  if (!CalleeFunc)
    return;

  StringRef CalleeFuncName = CalleeFunc->getName();
  if (isLoggerFunc(CalleeFuncName) || isLLVMTrap(CalleeFuncName))
    return;

  Value *CalleeName = Builder.CreateGlobalStringPtr(CalleeFuncName);
  Value *CallerName = Builder.CreateGlobalStringPtr(CallerFunc->getName());
  Value *CallerAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CallerFunc));
  Value *CalleeAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CalleeFunc));
  Value *Args[] = {CallerName, CallerAddr, CalleeName, CalleeAddr};

  Builder.CreateCall(LogFunc, Args);
}

} // namespace

char CallGraph::ID = 0;
static RegisterPass<CallGraph> X("callgraph", "CallGraph Pass");
