//===- unittests/Passes/TestPlugin.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "../TestPlugin.h"

using namespace llvm;

struct TestModulePass : public PassInfoMixin<TestModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    return PreservedAnalyses::all();
  }

  static void registerCallbacks(PassBuilder &PB) {
    PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &PM,
           ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
          if (Name == "plugin-pass") {
            PM.addPass(TestModulePass());
            return true;
          }
          return false;
        });
  }
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, TEST_PLUGIN_NAME, TEST_PLUGIN_VERSION,
          TestModulePass::registerCallbacks};
}
