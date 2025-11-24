#ifndef LLVM_TRANSFORMS_UTILS_GLOBALTYPEINFO_H
#define LLVM_TRANSFORMS_UTILS_GLOBALTYPEINFO_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/MyTy.h"

namespace llvm {
  class GlobalTypeInfoPass
    : public AnalysisInfoMixin<GlobalTypeInfoPass> {
  public:
    struct Result {
      DenseMap<StructType *, std::shared_ptr<MyTy>> structInfo;
    };
    static llvm::AnalysisKey Key;
    Result run(Module &M, ModuleAnalysisManager &AM);
  };
}

#endif