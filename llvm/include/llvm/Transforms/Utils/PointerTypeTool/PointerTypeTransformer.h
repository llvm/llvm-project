#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPETRANSFORMER_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPETRANSFORMER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypePrinter.h"
#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypeHelpers.h"

namespace llvm {

class PointerTypeTransformerPass
    : public PassInfoMixin<PointerTypeTransformerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

}
#endif
