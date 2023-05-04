#include "llvm/Transforms/Utils/MyHelloWorld.h"

using namespace llvm;

PreservedAnalyses MyHelloWorldPass::run(Function& F, FunctionAnalysisManager& AM)
{
    errs() << F.getName() << "\n";
    return PreservedAnalyses::all();
}
