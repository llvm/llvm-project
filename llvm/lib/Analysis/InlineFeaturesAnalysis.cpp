#include "llvm/Analysis/InlineFeaturesAnalysis.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

AnalysisKey InlineFeaturesAnalysis::Key;

InlineFeaturesAnalysis::Result
InlineFeaturesAnalysis::run(const Function &F, FunctionAnalysisManager &FAM) {
  Result Ret;
  Ret.Uses = ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();
  for (const auto &BB : F) {
    ++Ret.BasicBlockCount;
    if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (BI->isConditional())
        Ret.BlocksReachedFromConditionalInstruction += BI->getNumSuccessors();
    } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator()))
      Ret.BlocksReachedFromConditionalInstruction +=
          (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
    for (const auto &I : BB)
      if (auto *CS = dyn_cast<CallBase>(&I)) {
        const auto *Callee = CS->getCalledFunction();
        if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
          ++Ret.DirectCallsToDefinedFunctions;
      }
  }
  return Ret;
}