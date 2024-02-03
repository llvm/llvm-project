//===- opt-printplugin.cpp - The LLVM Modular Optimizer
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example static opt plugin which simply prints the names of all the functions
// within the generated LLVM code.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

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

void registerPlugin(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, llvm::ModulePassManager &PM,
         ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "printpass") {
          PM.addPass(PrintPass());
          return true;
        }
        return false;
      });
}

} // namespace

extern "C" int optMain(int argc, char **argv,
                       llvm::ArrayRef<std::function<void(llvm::PassBuilder &)>>
                           PassBuilderCallbacks);

int main(int argc, char **argv) {
  std::function<void(llvm::PassBuilder &)> plugins[] = {registerPlugin};
  return optMain(argc, argv, plugins);
}
