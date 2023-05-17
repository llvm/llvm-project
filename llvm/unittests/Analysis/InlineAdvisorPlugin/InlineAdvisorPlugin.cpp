#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Analysis/InlineAdvisor.h"

using namespace llvm;

namespace {

InlineAdvisor *DefaultAdvisorFactory(Module &M, FunctionAnalysisManager &FAM,
                                     InlineParams Params, InlineContext IC) {
  return new DefaultInlineAdvisor(M, FAM, Params, IC);
}

struct DefaultDynamicAdvisor : PassInfoMixin<DefaultDynamicAdvisor> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    PluginInlineAdvisorAnalysis DA(DefaultAdvisorFactory);
    MAM.registerPass([&] { return DA; });
    return PreservedAnalyses::all();
  }
};

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getDefaultDynamicAdvisorPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DynamicDefaultAdvisor", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level) {
                  MPM.addPass(DefaultDynamicAdvisor());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
                  if (Name == "dynamic-inline-advisor") {
                    PM.addPass(DefaultDynamicAdvisor());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getDefaultDynamicAdvisorPluginInfo();
}
