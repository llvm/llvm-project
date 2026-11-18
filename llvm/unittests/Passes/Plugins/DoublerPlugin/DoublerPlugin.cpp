//===- unittests/Passes/DoublerPlugin.cpp
//--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/LibraryOptionsContext.h"

#define OPTIONS_STRUCT_DECL
#include "DoublerOptions.inc"
#define OPTIONS_STRUCT_DEFS
#include "DoublerOptions.inc"

using namespace llvm;

struct DoublerModulePass : public OptionalPassInfoMixin<DoublerModulePass> {

  // Multiply the value of the initializer, by default with 2
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    // The options of the compilation that this module belongs to
    const DoublerOptions &Opts =
        M.getContext().getLibraryOptions().get<DoublerOptions>();
    auto *GV = cast<GlobalVariable>(M.getNamedValue("doubleme"));
    auto *Init = cast<ConstantInt>(GV->getInitializer());
    GV->setInitializer(
        ConstantInt::get(Init->getType(), Init->getSExtValue() * Opts.factor));

    return PreservedAnalyses::none();
  }

  static void registerCallbacks(PassBuilder &PB) {
    // The PassBuilder knows its context, so options are available here too
    StringRef PassName =
        PB.getContext().getLibraryOptions().get<DoublerOptions>().pass_name;
    PB.registerPipelineParsingCallback(
        [PassName](StringRef Name, ModulePassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
          if (Name == PassName) {
            PM.addPass(DoublerModulePass());
            return true;
          }
          return false;
        });
  }
};

// Parse into the host's options, which it copies into each context it
// compiles in
static Error parseArguments(cl::LibraryOptionsContext &LibraryOptions,
                            ArrayRef<const char *> Args) {
  static constexpr opt::LibraryOptionsParser Parser{
      std::in_place_type<DoublerOptions>};
  for (unsigned I = 0, Consumed = 0; I < Args.size(); I += Consumed)
    if (Error E = LibraryOptions.parse(Parser, Args.drop_front(I), Consumed))
      return E;
  return Error::success();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION,
          "DoublerPlugin",
          "2.2-unit",
          DoublerModulePass::registerCallbacks,
          nullptr,
          parseArguments};
}
