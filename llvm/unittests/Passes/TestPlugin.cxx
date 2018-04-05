//===- unittests/Passes/Plugins/Plugin.cxx --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "TestPlugin.h"

using namespace llvm;

struct TestModulePass : public PassInfoMixin<TestModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    return PreservedAnalyses::all();
  }
};

void registerCallbacks(PassBuilder &PB) {
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

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK LLVM_PLUGIN_EXPORT
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, TEST_PLUGIN_NAME, TEST_PLUGIN_VERSION,
          registerCallbacks};
}
