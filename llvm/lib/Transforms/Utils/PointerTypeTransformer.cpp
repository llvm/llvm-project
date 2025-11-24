#include "llvm/Transforms/Utils/PointerTypeTransformer.h"

using namespace llvm;

PreservedAnalyses PointerTypeTransformerPass::run(Module& M,
    ModuleAnalysisManager& MAM) {
  PointerTypePrinter printer(outs());
  PointerTypeHelpers helper;

  helper.initializeGlobalInfo(M);

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    helper.processInFunction(F);
  }

  printer.loadPointerTypeMap(helper.getPtm());
  printer.printModule(M);
  
  return PreservedAnalyses::all();
}

// wllvm