#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
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
  std::unordered_map<int64_t, llvm::SmallSet<int64_t, 8>> Edges;

  // Set for storing already existed nodes decl for graphviz
  std::unordered_set<int64_t> Nodes;

  FunctionCallee getCallLogFunc(Module &M, IRBuilder<> &Builder) const;
  void insertCallLogger(IRBuilder<> &Builder, int64_t CallerFunc,
                        int64_t CalleeFunc, FunctionCallee &LogFunc) const;
  bool isLoggerFunc(StringRef Name) const { return Name == LoggerFuncName; }
  bool isLLVMTrap(StringRef Name) const { return Name == "llvm.trap"; }

  int64_t getFuncHash(const Function *Func, StringRef FuncName,
                      Module &M) const;

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
      StringRef FuncName = F.getName();
      int64_t CallerHash = getFuncHash(&F, FuncName, M);
      auto &Bucket = Edges[CallerHash];

      // Traverse all instructions in F
      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
        if (auto *Call = dyn_cast<CallInst>(&*I)) {
          const Function *CalleeFunc = Call->getCalledFunction();
          if (!CalleeFunc)
            continue;

          StringRef CalleeName = CalleeFunc->getName();
          if (isLLVMTrap(CalleeName) || isLoggerFunc(CalleeName))
            return false;

          int64_t CalleeHash = getFuncHash(CalleeFunc, CalleeName, M);

          if (GlobalValue::isLocalLinkage(CalleeFunc->getLinkage())) {
            std::string NewName = M.getSourceFileName();
            NewName += CalleeName;
            CalleeHash = llvm::hash_value(NewName);
          } else {
            CalleeHash = llvm::hash_value(CalleeName);
          }

          Builder.SetInsertPoint(Call);
          insertCallLogger(Builder, CallerHash, CalleeHash, CallLogFunc);

          // Here we will add static information, if we meet edge for the
          // first time
          if (Bucket.contains(CalleeHash))
            continue;

          // If we meet that edge for the first time, we need to add it to map
          // and print it to file
          if (Nodes.find(CallerHash) == Nodes.end()) {
            File << "{} " << CallerHash << " [label = \" " << FuncName
                 << " \" ]\n";
            Nodes.insert(CallerHash);
          }
          if (Nodes.find(CalleeHash) == Nodes.end()) {
            File << "{} " << CalleeHash << " [label = \" " << CalleeName
                 << " \" ]\n";
            Nodes.insert(CalleeHash);
          }

          File << CallerHash << " -> " << CalleeHash << '\n';
          Bucket.insert(CalleeHash);
        }
      }
    }

    // It is nessacary for comfortable parsing
    File << '\n';
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

void CallGraph::insertCallLogger(IRBuilder<> &Builder, int64_t CallerFunc,
                                 int64_t CalleeFunc,
                                 FunctionCallee &LogFunc) const {
  if (!CalleeFunc)
    return;

  Value *CallerAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CallerFunc));
  Value *CalleeAddr =
      ConstantInt::get(Builder.getInt64Ty(), int64_t(CalleeFunc));
  Value *Args[] = {CallerAddr, CalleeAddr};

  Builder.CreateCall(LogFunc, Args);
}

/// Method for calculating a hash for Function
/// Check if function is static, because if we have static function, we
/// need to add fileName to the funcName to avoid collisions with other
/// static functions from different files.
int64_t CallGraph::getFuncHash(const Function *Func, StringRef FuncName,
                               Module &M) const {
  if (GlobalValue::isLocalLinkage(Func->getLinkage())) {
    std::string NewName = M.getSourceFileName();
    NewName += FuncName;
    return llvm::hash_value(NewName);
  }
  return llvm::hash_value(FuncName);
}

} // namespace

char CallGraph::ID = 0;
static RegisterPass<CallGraph> X("callgraph", "CallGraph Pass");
