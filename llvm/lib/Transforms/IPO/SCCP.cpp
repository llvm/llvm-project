#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar/SCCP.h"

using namespace llvm;

PreservedAnalyses IPSCCPPass::run(Module &M, ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(M);
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto getPredicateInfo =
      [&FAM](Function &F) -> std::unique_ptr<PredicateInfo> {
    return make_unique<PredicateInfo>(F,
                                      FAM.getResult<DominatorTreeAnalysis>(F),
                                      FAM.getResult<AssumptionAnalysis>(F));
  };

  if (!runIPSCCP(M, DL, &TLI, getPredicateInfo))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

namespace {

//===--------------------------------------------------------------------===//
//
/// IPSCCP Class - This class implements interprocedural Sparse Conditional
/// Constant Propagation.
///
class IPSCCPLegacyPass : public ModulePass {
public:
  static char ID;

  IPSCCPLegacyPass() : ModulePass(ID) {
    initializeIPSCCPLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    const DataLayout &DL = M.getDataLayout();
    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

    auto getPredicateInfo =
        [this](Function &F) -> std::unique_ptr<PredicateInfo> {
      return make_unique<PredicateInfo>(
          F, this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree(),
          this->getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F));
    };

    return runIPSCCP(M, DL, TLI, getPredicateInfo);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

} // end anonymous namespace

char IPSCCPLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(IPSCCPLegacyPass, "ipsccp",
                      "Interprocedural Sparse Conditional Constant Propagation",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(IPSCCPLegacyPass, "ipsccp",
                    "Interprocedural Sparse Conditional Constant Propagation",
                    false, false)

// createIPSCCPPass - This is the public interface to this file.
ModulePass *llvm::createIPSCCPPass() { return new IPSCCPLegacyPass(); }
