#include "llvm/Transforms/UncoalescedAnalysisPass.h"

using namespace llvm;

bool UncoalescedAnalysisPass::runOnFunction(Function &F) {
  // MultiplierValue::testMultiplierValue();
  // GPUState::testGPUState();

  auto &DomTree = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  UncoalescedAnalysis UA(&F, &DomTree);
  errs() << "Analysis Results: \n";
  GPUState st = UA.BuildInitialState();
  UA.BuildAnalysisInfo(st);
  UncoalescedAccesses_ = UA.getUncoalescedAccesses();
  return false;
}

char UncoalescedAnalysisPass::ID = 0;
/*
INITIALIZE_PASS_BEGIN(UncoalescedAnalysisPass, "uncoalesced-analysis", "Pass to generate uncoalesced access analysis for GPU programs", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(UncoalescedAnalysisPass, "uncoalesced-analysis", "Pass to generate uncoalesced access analysis for GPU programs", true, true)
*/
static RegisterPass<UncoalescedAnalysisPass>
Y("uncoalesced-analysis", "Pass to detect uncoalesced accesses in gpu programs.");

FunctionPass* createUncoalescedAnalysisPass() {
  return new UncoalescedAnalysisPass();
}
