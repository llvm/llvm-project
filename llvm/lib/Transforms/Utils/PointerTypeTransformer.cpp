#include "llvm/Transforms/Utils/PointerTypeTransformer.h"

using namespace llvm;

PreservedAnalyses PointerTypeTransformerPass::run(Module& M,
    ModuleAnalysisManager& MAM) {
  PointerTypePrinter *printer = new PointerTypePrinter(outs());

  auto &FAMProxy = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M);
  FunctionAnalysisManager &FAM = FAMProxy.getManager();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.getName() == "main") {
      auto &result = FAM.getResult<PointerTypeInFunctionPass>(F);
      printer->loadPointerTypeMap(result.pointerTypeMap);
    }
  }

  printer->printModule(M);
  
  return PreservedAnalyses::all();
}