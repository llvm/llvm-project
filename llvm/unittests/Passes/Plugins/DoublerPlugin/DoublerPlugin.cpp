//===- unittests/Passes/DoublerPlugin.cpp
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

struct DoublerModulePass : public PassInfoMixin<DoublerModulePass> {

  // Double the value of the initializer
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    auto *GV = cast<GlobalVariable>(M.getNamedValue("doubleme"));
    auto *Init = GV->getInitializer();
    auto *Init2 = ConstantExpr::getAdd(Init, Init);
    GV->setInitializer(Init2);

    return PreservedAnalyses::none();
  }

  static void registerCallbacks(PassBuilder &PB) {
    PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &PM,
           ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
          if (Name == "doubler-pass") {
            PM.addPass(DoublerModulePass());
            return true;
          }
          return false;
        });
  }
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DoublerPlugin", "2.2-unit",
          DoublerModulePass::registerCallbacks};
}
