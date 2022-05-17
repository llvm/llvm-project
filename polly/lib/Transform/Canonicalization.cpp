//===---- Canonicalization.cpp - Run canonicalization passes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run the set of default canonicalization passes.
//
// This pass is mainly used for debugging.
//
//===----------------------------------------------------------------------===//

#include "polly/Canonicalization.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

using namespace llvm;
using namespace polly;

static cl::opt<bool>
    PollyInliner("polly-run-inliner",
                 cl::desc("Run an early inliner pass before Polly"), cl::Hidden,
                 cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

void polly::registerCanonicalicationPasses(llvm::legacy::PassManagerBase &PM) {
  bool UseMemSSA = true;
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createEarlyCSEPass(UseMemSSA));
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createTailCallEliminationPass());
  PM.add(llvm::createCFGSimplificationPass());
  PM.add(llvm::createReassociatePass());
  PM.add(llvm::createLoopRotatePass());
  if (PollyInliner) {
    PM.add(llvm::createFunctionInliningPass(200));
    PM.add(llvm::createPromoteMemoryToRegisterPass());
    PM.add(llvm::createCFGSimplificationPass());
    PM.add(llvm::createInstructionCombiningPass());
    PM.add(createBarrierNoopPass());
  }
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createIndVarSimplifyPass());
}

/// Adapted from llvm::PassBuilder::buildInlinerPipeline
static ModuleInlinerWrapperPass
buildInlinePasses(llvm::OptimizationLevel Level) {
  InlineParams IP = getInlineParams(200);
  ModuleInlinerWrapperPass MIWP(IP);

  // Require the GlobalsAA analysis for the module so we can query it within
  // the CGSCC pipeline.
  MIWP.addModulePass(RequireAnalysisPass<GlobalsAA, Module>());
  // Invalidate AAManager so it can be recreated and pick up the newly available
  // GlobalsAA.
  MIWP.addModulePass(
      createModuleToFunctionPassAdaptor(InvalidateAnalysisPass<AAManager>()));

  // Require the ProfileSummaryAnalysis for the module so we can query it within
  // the inliner pass.
  MIWP.addModulePass(RequireAnalysisPass<ProfileSummaryAnalysis, Module>());

  // Now begin the main postorder CGSCC pipeline.
  // FIXME: The current CGSCC pipeline has its origins in the legacy pass
  // manager and trying to emulate its precise behavior. Much of this doesn't
  // make a lot of sense and we should revisit the core CGSCC structure.
  CGSCCPassManager &MainCGPipeline = MIWP.getPM();

  // Now deduce any function attributes based in the current code.
  MainCGPipeline.addPass(PostOrderFunctionAttrsPass());

  return MIWP;
}

FunctionPassManager
polly::buildCanonicalicationPassesForNPM(llvm::ModulePassManager &MPM,
                                         llvm::OptimizationLevel Level) {
  FunctionPassManager FPM;

  bool UseMemSSA = true;
  FPM.addPass(PromotePass());
  FPM.addPass(EarlyCSEPass(UseMemSSA));
  FPM.addPass(InstCombinePass());
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(TailCallElimPass());
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(ReassociatePass());
  {
    LoopPassManager LPM;
    LPM.addPass(LoopRotatePass(Level != OptimizationLevel::Oz));
    FPM.addPass(createFunctionToLoopPassAdaptor<LoopPassManager>(
        std::move(LPM), /*UseMemorySSA=*/false,
        /*UseBlockFrequencyInfo=*/false));
  }
  if (PollyInliner) {
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(buildInlinePasses(Level));
    FPM = FunctionPassManager();

    FPM.addPass(PromotePass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(InstCombinePass());
  }
  FPM.addPass(InstCombinePass());
  {
    LoopPassManager LPM;
    LPM.addPass(IndVarSimplifyPass());
    FPM.addPass(createFunctionToLoopPassAdaptor<LoopPassManager>(
        std::move(LPM), /*UseMemorySSA=*/false,
        /*UseBlockFrequencyInfo=*/true));
  }

  return FPM;
}

namespace {
class PollyCanonicalize final : public ModulePass {
  PollyCanonicalize(const PollyCanonicalize &) = delete;
  const PollyCanonicalize &operator=(const PollyCanonicalize &) = delete;

public:
  static char ID;

  explicit PollyCanonicalize() : ModulePass(ID) {}
  ~PollyCanonicalize();

  /// @name FunctionPass interface.
  //@{
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnModule(Module &M) override;
  void print(raw_ostream &OS, const Module *) const override;
  //@}
};
} // namespace

PollyCanonicalize::~PollyCanonicalize() {}

void PollyCanonicalize::getAnalysisUsage(AnalysisUsage &AU) const {}

void PollyCanonicalize::releaseMemory() {}

bool PollyCanonicalize::runOnModule(Module &M) {
  legacy::PassManager PM;
  registerCanonicalicationPasses(PM);
  PM.run(M);

  return true;
}

void PollyCanonicalize::print(raw_ostream &OS, const Module *) const {}

char PollyCanonicalize::ID = 0;

Pass *polly::createPollyCanonicalizePass() { return new PollyCanonicalize(); }

INITIALIZE_PASS_BEGIN(PollyCanonicalize, "polly-canonicalize",
                      "Polly - Run canonicalization passes", false, false)
INITIALIZE_PASS_END(PollyCanonicalize, "polly-canonicalize",
                    "Polly - Run canonicalization passes", false, false)
