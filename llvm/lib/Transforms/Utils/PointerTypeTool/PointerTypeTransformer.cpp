#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypeTransformer.h"

using namespace llvm;

PreservedAnalyses PointerTypeTransformerPass::run(Module& M,
    ModuleAnalysisManager& MAM) {
  auto helper = new PointerTypeHelpers(M);
  helper->analyzer->visitModule(M);
  helper->printer->printModule(M);
  
  return PreservedAnalyses::all();
}

// wllvm