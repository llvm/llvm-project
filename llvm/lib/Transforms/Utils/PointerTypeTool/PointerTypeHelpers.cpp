#include "llvm/Transforms/Utils/PointerTypeTool/PointerTypeHelpers.h"

using namespace llvm;

PointerTypeHelpers::PointerTypeHelpers(Module &M) { 
  analyzer = new FlowAnalyzer(M);
  printer = new PointerTypePrinter(outs(), *analyzer);
}