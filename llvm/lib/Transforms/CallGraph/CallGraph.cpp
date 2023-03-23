#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <sys/stat.h>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

#define DEBUG_TYPE "callGraph"

namespace {
void makeDotPrologue(raw_fd_ostream &File) { File << "digraph G {\n"; }

/// Class for implemeting CallGraph building
struct CallGraph : ModulePass {
  static char ID;
  CallGraph() : ModulePass(ID) { std::error_code EC; }

  StringRef LoggerFuncName = "_Z6Loggerll";

  // Edges map contains key=caller and value=set_of_calees
  std::unordered_map<const Function *, SmallPtrSet<const Function *, 8>> Edges;

  // Set for storing already existed nodes decl for graphviz
  std::unordered_set<const Function *> Nodes;

  FunctionCallee getCallLogFunc(Module &M, IRBuilder<> &Builder) const;
  void insertCallLogger(IRBuilder<> &Builder, const Function *CallerFunc,
                        const Function *CalleeFunc,
                        FunctionCallee &LogFunc) const;
  bool isLoggerFunc(StringRef Name) const { return Name == LoggerFuncName; }
  bool isLLVMTrap(StringRef Name) const { return Name == "llvm.trap"; }

  bool runOnModule(Module &M) override {
    constexpr StringRef FileName = "OutFile.dot";
    std::error_code EC;
    // We use append mode, because we support processing of multiple-modules
    // projects
    raw_fd_ostream File{FileName, EC, sys::fs::OF_Append};
    assert(EC.value() == 0);

    // For correct dot-format, we need to know if that file was alredy opened by
    // other modules or not.
    struct stat StatBuf;
    if (!stat(FileName.data(), &StatBuf)) {
      if (StatBuf.st_size == 0) {
        makeDotPrologue(File);
      }
    } else {
      perror("Stat failed:");
      return false;
    }

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
        const Function *FuncAddr = &F;
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
            if (Nodes.find(FuncAddr) == Nodes.end()) {
              File << "{} " << reinterpret_cast<int64_t>(&F) << " [label = \" "
                   << FuncName << " \" ]\n";
              Nodes.insert(FuncAddr);
            }
            if (Nodes.find(CalledFunc) == Nodes.end()) {
              File << "{} " << reinterpret_cast<int64_t>(CalledFunc)
                   << " [label = \" " << CalledFunc->getName() << " \" ]\n";
              Nodes.insert(CalledFunc);
            }

            File << reinterpret_cast<int64_t>(FuncAddr) << " -> "
                 << reinterpret_cast<int64_t>(CalledFunc) << '\n';
            Bucket.insert(CalledFunc);
          }
        }
      }
    }

    return true;
  }
};

FunctionCallee CallGraph::getCallLogFunc(Module &M,
                                         IRBuilder<> &Builder) const {
  Type *RetType = Type::getVoidTy(M.getContext());

  // Prepare callLogger function
  // Logger func will get 2 params: caller addr, callee addr
  SmallVector<Type *> CallParamType = {Builder.getInt64Ty(),
                                       Builder.getInt64Ty()};

  FunctionType *CallLogFuncType =
      FunctionType::get(RetType, CallParamType, false);
  FunctionCallee CallLogFunc =
      M.getOrInsertFunction(LoggerFuncName, CallLogFuncType);

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

  Value *CallerAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CallerFunc));
  Value *CalleeAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CalleeFunc));
  Value *Args[] = {CallerAddr, CalleeAddr};

  Builder.CreateCall(LogFunc, Args);
}

} // namespace

char CallGraph::ID = 0;
static RegisterPass<CallGraph> X("callgraph", "CallGraph Pass");
