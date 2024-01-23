//===- LLVMPrintFunctionNames.cpp
//---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example opt plugin which simply prints the names of all the functions
// within the generated LLVM code.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class PrintPass final : public llvm::AnalysisInfoMixin<PrintPass> {
  friend struct llvm::AnalysisInfoMixin<PrintPass>;

private:
  static llvm::AnalysisKey key;

public:
  using Result = llvm::PreservedAnalyses;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    for (auto &F : M)
      llvm::outs() << "[PrintPass] Found function: " << F.getName() << "\n";
    return llvm::PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

} // namespace

llvm::PassPluginLibraryInfo getPrintPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PrintPlugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::ModulePassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "printpass") {
                    PM.addPass(PrintPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfoPrintPlugin() {
  return getPrintPluginInfo();
}
