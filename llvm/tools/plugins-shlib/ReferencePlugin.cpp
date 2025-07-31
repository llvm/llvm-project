//===- tools/plugins-shlib/ReferencePlugin.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Instrumentor.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

using namespace llvm;

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

static void registerCallbacks(PassBuilder &PB) {
  // Entry-points for module passes
  if (getEnvBool("registerPipelineStartEPCallback"))
    PB.registerPipelineStartEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt) {
          MPM.addPass(InstrumentorPass());
          return true;
        });
  if (getEnvBool("registerPipelineEarlySimplificationEPCallback", true))
    PB.registerPipelineEarlySimplificationEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(InstrumentorPass());
          return true;
        });
  if (getEnvBool("registerOptimizerEarlyEPCallback"))
    PB.registerOptimizerEarlyEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(InstrumentorPass());
          return true;
        });
  if (getEnvBool("registerOptimizerLastEPCallback"))
    PB.registerOptimizerLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Opt, ThinOrFullLTOPhase Phase) {
          MPM.addPass(InstrumentorPass());
          return true;
        });
  if (getEnvBool("registerPipelineParsingCallback"), true)
    PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &PM,
           ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "instrumentor") {
            PM.addPass(InstrumentorPass());
            return true;
          }
          return false;
        });
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ReferencePlugin", LLVM_VERSION_STRING,
          registerCallbacks};
}
