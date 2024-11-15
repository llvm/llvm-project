//===- PassManager.cpp - Runs a pipeline of Sandbox IR passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/PassManager.h"

namespace llvm::sandboxir {

bool FunctionPassManager::runOnFunction(Function &F) {
  bool Change = false;
  for (auto &Pass : Passes) {
    Change |= Pass->runOnFunction(F);
    // TODO: run the verifier.
  }
  // TODO: Check ChangeAll against hashes before/after.
  return Change;
}

bool RegionPassManager::runOnRegion(Region &R) {
  bool Change = false;
  for (auto &Pass : Passes) {
    Change |= Pass->runOnRegion(R);
    // TODO: run the verifier.
  }
  // TODO: Check ChangeAll against hashes before/after.
  return Change;
}

} // namespace llvm::sandboxir
