#include "llvm/IR/PassManager.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/DependenceInfo.h"
#include "polly/ScopPass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "gtest/gtest.h"

using namespace polly;
using namespace llvm;

namespace {
class ScopPassRegistry : public ::testing::Test {
protected:
  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ScopAnalysisManager SAM;
  AAManager AM;

public:
  ScopPassRegistry(ScopPassRegistry &&) = delete;
  ScopPassRegistry(const ScopPassRegistry &) = delete;
  ScopPassRegistry &operator=(ScopPassRegistry &&) = delete;
  ScopPassRegistry &operator=(const ScopPassRegistry &) = delete;
  ScopPassRegistry() {
    PassBuilder PB;

    AM = PB.buildDefaultAAPipeline();
    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);

    FAM.registerPass([] { return ScopAnalysis(); });
    FAM.registerPass([] { return ScopInfoAnalysis(); });
    FAM.registerPass([this] { return ScopAnalysisManagerFunctionProxy(SAM); });

    // SAM.registerPass([] { return IslAstAnalysis(); });
    // SAM.registerPass([] { return DependenceAnalysis(); });
    SAM.registerPass([this] { return FunctionAnalysisManagerScopProxy(FAM); });

    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

TEST_F(ScopPassRegistry, PrintScops) {
  FunctionPassManager FPM;
  FPM.addPass(ScopAnalysisPrinterPass(errs()));
}

TEST_F(ScopPassRegistry, PrintScopInfo) {
  FunctionPassManager FPM;
  FPM.addPass(ScopInfoPrinterPass(errs()));
}

TEST_F(ScopPassRegistry, PrinIslAstInfo) {
  FunctionPassManager FPM;
  ScopPassManager SPM;
  // SPM.addPass(IslAstPrinterPass(errs()));
  FPM.addPass(createFunctionToScopPassAdaptor(std::move(SPM)));
}
} // namespace
