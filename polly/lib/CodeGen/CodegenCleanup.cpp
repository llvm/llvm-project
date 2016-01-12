#include "polly/CodeGen/CodegenCleanup.h"

#include "llvm/Analysis/CFLAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/PassInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#define DEBUG_TYPE "polly-cleanup"

using namespace llvm;
using namespace polly;

namespace {

class CodegenCleanup : public FunctionPass {
private:
  CodegenCleanup(const CodegenCleanup &) = delete;
  const CodegenCleanup &operator=(const CodegenCleanup &) = delete;

  llvm::legacy::FunctionPassManager *FPM;

public:
  static char ID;
  explicit CodegenCleanup() : FunctionPass(ID), FPM(nullptr) {}

  /// @name FunctionPass interface
  //@{
  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}

  virtual bool doInitialization(Module &M) override {
    assert(!FPM);

    FPM = new llvm::legacy::FunctionPassManager(&M);

    // TODO: How to make parent passes discoverable?
    // TODO: Should be sensitive to compiler options in PassManagerBuilder, to
    // which wo do not have access here.
    FPM->add(createCFLAAWrapperPass());
    FPM->add(createScopedNoAliasAAWrapperPass());
    FPM->add(createTypeBasedAAWrapperPass());
    FPM->add(createAAResultsWrapperPass());

    // TODO: These are non-conditional passes that run between
    // EP_ModuleOptimizerEarly and EP_VectorizerStart just to ensure we do not
    // miss any optimization that would have run after Polly with
    // -polly-position=early. This can probably be reduced to a more compact set
    // of passes.
    FPM->add(createCFGSimplificationPass());
    FPM->add(createScalarReplAggregatesPass());
    FPM->add(createEarlyCSEPass());
    FPM->add(createInstructionCombiningPass());
    FPM->add(createJumpThreadingPass());
    FPM->add(createCorrelatedValuePropagationPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createInstructionCombiningPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createReassociatePass());
    FPM->add(createLoopRotatePass());
    FPM->add(createLICMPass());
    FPM->add(createLoopUnswitchPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createInstructionCombiningPass());
    FPM->add(createIndVarSimplifyPass());
    FPM->add(createLoopIdiomPass());
    FPM->add(createLoopDeletionPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createSimpleLoopUnrollPass());
    FPM->add(createMergedLoadStoreMotionPass());
    FPM->add(createMemCpyOptPass());
    FPM->add(createBitTrackingDCEPass());
    FPM->add(createInstructionCombiningPass());
    FPM->add(createJumpThreadingPass());
    FPM->add(createCorrelatedValuePropagationPass());
    FPM->add(createDeadStoreEliminationPass());
    FPM->add(createLICMPass());
    FPM->add(createLoopRerollPass());
    FPM->add(createLoadCombinePass());
    FPM->add(createAggressiveDCEPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createInstructionCombiningPass());

    return FPM->doInitialization();
  }

  virtual bool doFinalization(Module &M) override {
    bool Result = FPM->doFinalization();

    delete FPM;
    FPM = nullptr;

    return Result;
  }

  virtual bool runOnFunction(llvm::Function &F) override {
    if (!F.hasFnAttribute("polly-optimized")) {
      DEBUG(dbgs() << F.getName()
                   << ": Skipping cleanup because Polly did not optimize it.");
      return false;
    }

    DEBUG(dbgs() << F.getName() << ": Running codegen cleanup...");
    return FPM->run(F);
  }
  //@}
};

char CodegenCleanup::ID;
}

FunctionPass *polly::createCodegenCleanupPass() { return new CodegenCleanup(); }

INITIALIZE_PASS_BEGIN(CodegenCleanup, "polly-cleanup",
                      "Polly - Cleanup after code generation", false, false)
INITIALIZE_PASS_END(CodegenCleanup, "polly-cleanup",
                    "Polly - Cleanup after code generation", false, false)
