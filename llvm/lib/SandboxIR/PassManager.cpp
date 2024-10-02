//===- PassManager.cpp - Runs a pipeline of Sandbox IR passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/PassManager.h"

using namespace llvm::sandboxir;

bool FunctionPassManager::runOnFunction(Function &F) {
  bool Change = false;
  for (FunctionPass *Pass : Passes) {
    Change |= Pass->runOnFunction(F);
    // TODO: run the verifier.
  }
  // TODO: Check ChangeAll against hashes before/after.
  return Change;
}

FunctionPassManager &
PassRegistry::parseAndCreatePassPipeline(StringRef Pipeline) {
  static constexpr const char EndToken = '\0';
  // Add EndToken to the end to ease parsing.
  std::string PipelineStr = std::string(Pipeline) + EndToken;
  int FlagBeginIdx = 0;
  // Start with a FunctionPassManager.
  auto &InitialPM = static_cast<FunctionPassManager &>(
      registerPass(std::make_unique<FunctionPassManager>("init-fpm")));

  for (auto [Idx, C] : enumerate(PipelineStr)) {
    // Keep moving Idx until we find the end of the pass name.
    bool FoundDelim = C == EndToken || C == PassDelimToken;
    if (!FoundDelim)
      continue;
    unsigned Sz = Idx - FlagBeginIdx;
    std::string PassName(&PipelineStr[FlagBeginIdx], Sz);
    FlagBeginIdx = Idx + 1;

    // Get the pass that corresponds to PassName and add it to the pass manager.
    auto *Pass = getPassByName(PassName);
    if (Pass == nullptr) {
      errs() << "Pass '" << PassName << "' not registered!\n";
      exit(1);
    }
    // TODO: This is safe for now, but would require proper upcasting once we
    // add more Pass sub-classes.
    InitialPM.addPass(static_cast<FunctionPass *>(Pass));
  }
  return InitialPM;
}
#ifndef NDEBUG
void PassRegistry::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
