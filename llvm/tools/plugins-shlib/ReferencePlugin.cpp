//===- lib/plugins-shlib/ReferencePlugin.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

using namespace llvm;

static cl::opt<bool> Wave("wave-goodbye", cl::init(false),
                          cl::desc("wave good bye"));

static std::optional<std::string> getEnv(const std::string &Var) {
  const char *Val = std::getenv(Var.c_str());
  if (!Val)
    return std::nullopt;
  return std::string(Val);
}

static bool getEnvBool(const std::string &VarName, bool Default = false) {
  if (auto ValOpt = getEnv(VarName)) {
    std::string V = *ValOpt;
    std::transform(V.begin(), V.end(), V.begin(), ::tolower);

    if (V == "1" || V == "true" || V == "yes" || V == "on")
      return true;
    if (V == "0" || V == "false" || V == "no" || V == "off")
      return false;
  }
  return Default;
}

static std::string getLTOPhaseStr(ThinOrFullLTOPhase P) {
  switch (P) {
    case ThinOrFullLTOPhase::None: return "None";
    case ThinOrFullLTOPhase::ThinLTOPreLink: return "ThinLTOPreLink";
    case ThinOrFullLTOPhase::ThinLTOPostLink: return "ThinLTOPostLink";
    case ThinOrFullLTOPhase::FullLTOPreLink: return "FullLTOPreLink";
    case ThinOrFullLTOPhase::FullLTOPostLink: return "FullLTOPostLink";
  }
}

struct TestModulePass : public PassInfoMixin<TestModulePass> {
  std::string EP;
  std::string LTOPhase;
  TestModulePass(StringRef EntryPoint, ThinOrFullLTOPhase Phase = ThinOrFullLTOPhase::None) : EP(EntryPoint.str()), LTOPhase(getLTOPhaseStr(Phase)) {}
  PreservedAnalyses run(Module &, ModuleAnalysisManager &) {
    fprintf(stderr, "Entry-point: %s\n", EP.c_str());
    fprintf(stderr, "LTO-phase: %s\n", LTOPhase.c_str());
    return PreservedAnalyses::all();
  }
};

struct TestFunctionPass : public PassInfoMixin<TestFunctionPass> {
  std::string EP;
  TestFunctionPass(StringRef EntryPoint) : EP(EntryPoint.str()) {}
  PreservedAnalyses run(Function &, FunctionAnalysisManager &) {
    fprintf(stderr, "Entry-point: %s\n", EP.c_str());
    return PreservedAnalyses::all();
  }
};

static void registerCallbacks(PassBuilder &PB) {
  printf("Plugin parameter value -wave-goodbye=%s\n", Wave ? "true" : "false");

  // Entry-points for module passes
  if (getEnvBool("registerPipelineStartEPCallback"))
    PB.registerPipelineStartEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(TestModulePass("registerPipelineStartEPCallback"));
          return true;
        });
  if (getEnvBool("registerPipelineEarlySimplificationEPCallback", true))
    PB.registerPipelineEarlySimplificationEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(TestModulePass("registerPipelineEarlySimplificationEPCallback", Phase));
          return true;
        });
  if (getEnvBool("registerOptimizerEarlyEPCallback"))
    PB.registerOptimizerEarlyEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(TestModulePass("registerOptimizerEarlyEPCallback", Phase));
          return true;
        });
  if (getEnvBool("registerOptimizerLastEPCallback"))
    PB.registerOptimizerLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(TestModulePass("registerOptimizerLastEPCallback", Phase));
          return true;
        });
  if (getEnvBool("registerFullLinkTimeOptimizationEarlyEPCallback"))
    PB.registerFullLinkTimeOptimizationEarlyEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(TestModulePass("registerFullLinkTimeOptimizationEarlyEPCallback"));
          return true;
        });
  if (getEnvBool("registerFullLinkTimeOptimizationLastEPCallback"))
    PB.registerFullLinkTimeOptimizationLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(TestModulePass("registerFullLinkTimeOptimizationLastEPCallback"));
          return true;
        });

  // Entry-points for function passes
  if (getEnvBool("registerPeepholeEPCallback"))
    PB.registerPeepholeEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(TestFunctionPass("registerPeepholeEPCallback"));
          return true;
        });
  if (getEnvBool("registerScalarOptimizerLateEPCallback"))
    PB.registerScalarOptimizerLateEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(TestFunctionPass("registerScalarOptimizerLateEPCallback"));
          return true;
        });
  if (getEnvBool("registerVectorizerStartEPCallback"))
    PB.registerVectorizerStartEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(TestFunctionPass("registerVectorizerStartEPCallback"));
          return true;
        });

#if LLVM_VERSION_MAJOR > 20
  if (getEnvBool("registerVectorizerEndEPCallback"))
    PB.registerVectorizerEndEPCallback(
        [](FunctionPassManager &FPM, OptimizationLevel Opt) {
          FPM.addPass(TestFunctionPass("registerVectorizerEndEPCallback"));
          return true;
        });
#endif

  // TODO: registerLateLoopOptimizationsEPCallback, registerCGSCCOptimizerLateEPCallback
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ReferencePlugin", LLVM_VERSION_STRING,
          registerCallbacks};
}
