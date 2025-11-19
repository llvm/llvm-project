#include "llvm/Transforms/Utils/GlobalTypeInfo.h"

using namespace llvm;

GlobalTypeInfoPass::Result GlobalTypeInfoPass::run(Module& M,
    ModuleAnalysisManager& AM) {
  Result result;
  for (auto st : M.getIdentifiedStructTypes()) {
    result.structInfo[st] = MyTy::from(st);
  }
  return result;
}