#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPEHELPERS_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_POINTERTYPEHELPERS_H

#include "llvm/Transforms/Utils/PointerTypeTool/FlowAnalyzer.h"
#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypePrinter.h"

namespace llvm {

class PointerTypeHelpers {
public:
  FlowAnalyzer *analyzer;
  PointerTypePrinter *printer;
  PointerTypeHelpers(Module &M);
};

}

#endif
