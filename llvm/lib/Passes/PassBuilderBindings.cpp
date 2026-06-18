//===-------------- PassBuilder bindings for LLVM-C -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the C bindings to the new pass manager
///
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace llvm;

namespace llvm {
/// Helper struct for holding a set of builder options for LLVMRunPasses. This
/// structure is used to keep LLVMRunPasses backwards compatible with future
/// versions in case we modify the options the new Pass Manager utilizes.
class LLVMPassBuilderOptions {
public:
  explicit LLVMPassBuilderOptions(
      bool DebugLogging = false, bool VerifyEach = false,
      const char *AAPipeline = nullptr,
      PipelineTuningOptions PTO = PipelineTuningOptions())
      : DebugLogging(DebugLogging), VerifyEach(VerifyEach),
        AAPipeline(AAPipeline), PTO(PTO) {}

  bool DebugLogging;
  bool VerifyEach;
  const char *AAPipeline;
  PipelineTuningOptions PTO;
};
} // namespace llvm

static TargetMachine *unwrap(LLVMTargetMachineRef P) {
  return reinterpret_cast<TargetMachine *>(P);
}

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(LLVMPassBuilderOptions,
                                   LLVMPassBuilderOptionsRef)

static LLVMErrorRef runPasses(Module *Mod, Function *Fun, const char *Passes,
                              TargetMachine *Machine,
                              LLVMPassBuilderOptions *PassOpts) {
  bool Debug = PassOpts->DebugLogging;
  bool VerifyEach = PassOpts->VerifyEach;

  PassInstrumentationCallbacks PIC;
  PassBuilder PB(Machine, PassOpts->PTO, std::nullopt, &PIC);

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  if (PassOpts->AAPipeline) {
    // If we have a custom AA pipeline, we need to register it _before_ calling
    // registerFunctionAnalyses, or the default alias analysis pipeline is used.
    AAManager AA;
    if (auto Err = PB.parseAAPipeline(AA, PassOpts->AAPipeline))
      return wrap(std::move(Err));
    FAM.registerPass([&] { return std::move(AA); });
  }
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  StandardInstrumentations SI(Mod->getContext(), Debug, VerifyEach);
  SI.registerCallbacks(PIC, &MAM);

  // Run the pipeline.
  if (Fun) {
    FunctionPassManager FPM;
    if (VerifyEach)
      FPM.addPass(VerifierPass());
    if (auto Err = PB.parsePassPipeline(FPM, Passes))
      return wrap(std::move(Err));
    FPM.run(*Fun, FAM);
  } else {
    ModulePassManager MPM;
    if (VerifyEach)
      MPM.addPass(VerifierPass());
    if (auto Err = PB.parsePassPipeline(MPM, Passes))
      return wrap(std::move(Err));
    MPM.run(*Mod, MAM);
  }

  return LLVMErrorSuccess;
}

LLVMErrorRef LLVMRunPasses(LLVMModuleRef M, const char *Passes,
                           LLVMTargetMachineRef TM,
                           LLVMPassBuilderOptionsRef Options) {
  TargetMachine *Machine = unwrap(TM);
  LLVMPassBuilderOptions *PassOpts = unwrap(Options);
  Module *Mod = unwrap(M);
  return runPasses(Mod, nullptr, Passes, Machine, PassOpts);
}

LLVMErrorRef LLVMRunPassesOnFunction(LLVMValueRef F, const char *Passes,
                                     LLVMTargetMachineRef TM,
                                     LLVMPassBuilderOptionsRef Options) {
  TargetMachine *Machine = unwrap(TM);
  LLVMPassBuilderOptions *PassOpts = unwrap(Options);
  Function *Fun = unwrap<Function>(F);
  return runPasses(Fun->getParent(), Fun, Passes, Machine, PassOpts);
}

LLVMPassBuilderOptionsRef LLVMCreatePassBuilderOptions() {
  return wrap(new LLVMPassBuilderOptions());
}

void LLVMPassBuilderOptionsSetVerifyEach(LLVMPassBuilderOptionsRef Options,
                                         LLVMBool VerifyEach) {
  unwrap(Options)->VerifyEach = VerifyEach;
}

void LLVMPassBuilderOptionsSetDebugLogging(LLVMPassBuilderOptionsRef Options,
                                           LLVMBool DebugLogging) {
  unwrap(Options)->DebugLogging = DebugLogging;
}

void LLVMPassBuilderOptionsSetAAPipeline(LLVMPassBuilderOptionsRef Options,
                                         const char *AAPipeline) {
  unwrap(Options)->AAPipeline = AAPipeline;
}

void LLVMPassBuilderOptionsSetLoopInterleaving(
    LLVMPassBuilderOptionsRef Options, LLVMBool LoopInterleaving) {
  unwrap(Options)->PTO.LoopInterleaving = LoopInterleaving;
}

void LLVMPassBuilderOptionsSetLoopVectorization(
    LLVMPassBuilderOptionsRef Options, LLVMBool LoopVectorization) {
  unwrap(Options)->PTO.LoopVectorization = LoopVectorization;
}

void LLVMPassBuilderOptionsSetSLPVectorization(
    LLVMPassBuilderOptionsRef Options, LLVMBool SLPVectorization) {
  unwrap(Options)->PTO.SLPVectorization = SLPVectorization;
}

void LLVMPassBuilderOptionsSetLoopUnrolling(LLVMPassBuilderOptionsRef Options,
                                            LLVMBool LoopUnrolling) {
  unwrap(Options)->PTO.LoopUnrolling = LoopUnrolling;
}

void LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(
    LLVMPassBuilderOptionsRef Options, LLVMBool ForgetAllSCEVInLoopUnroll) {
  unwrap(Options)->PTO.ForgetAllSCEVInLoopUnroll = ForgetAllSCEVInLoopUnroll;
}

void LLVMPassBuilderOptionsSetLicmMssaOptCap(LLVMPassBuilderOptionsRef Options,
                                             unsigned LicmMssaOptCap) {
  unwrap(Options)->PTO.LicmMssaOptCap = LicmMssaOptCap;
}

void LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(
    LLVMPassBuilderOptionsRef Options, unsigned LicmMssaNoAccForPromotionCap) {
  unwrap(Options)->PTO.LicmMssaNoAccForPromotionCap =
      LicmMssaNoAccForPromotionCap;
}

void LLVMPassBuilderOptionsSetCallGraphProfile(
    LLVMPassBuilderOptionsRef Options, LLVMBool CallGraphProfile) {
  unwrap(Options)->PTO.CallGraphProfile = CallGraphProfile;
}

void LLVMPassBuilderOptionsSetMergeFunctions(LLVMPassBuilderOptionsRef Options,
                                             LLVMBool MergeFunctions) {
  unwrap(Options)->PTO.MergeFunctions = MergeFunctions;
}

void LLVMPassBuilderOptionsSetInlinerThreshold(
    LLVMPassBuilderOptionsRef Options, int Threshold) {
  unwrap(Options)->PTO.InlinerThreshold = Threshold;
}

void LLVMDisposePassBuilderOptions(LLVMPassBuilderOptionsRef Options) {
  delete unwrap(Options);
}
