//===- PassManager.cpp - Runs a pipeline of Sandbox IR passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/PassManager.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

bool FunctionPassManager::runOnFunction(Function &F, const Analyses &A) {
  bool Change = false;
  for (auto &Pass : Passes) {
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
    auto HashBefore = F.hash();
#endif
    bool PassChange = Pass->runOnFunction(F, A);
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
    auto HashAfter = F.hash();
    assert((PassChange || HashAfter == HashBefore) &&
           "The pass returned false (i.e., no change) but the IR changed!");
#endif
    Change |= PassChange;
    // TODO: run the verifier.
  }
  return Change;
}

bool RegionPassManager::runOnRegion(Region &R, const Analyses &A) {
  bool Change = false;
  for (auto &Pass : Passes) {
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
    auto &F = R.getParentFunction();
    auto HashBefore = F.hash();
#endif
    bool PassChange = Pass->runOnRegion(R, A);
#if !defined(NDEBUG) && defined(EXPENSIVE_CHECKS)
    auto HashAfter = F.hash();
    assert((PassChange || HashAfter == HashBefore) &&
           "The pass returned false (i.e., no change) but the IR changed!");
#endif
    Change |= PassChange;
    // TODO: run the verifier.
  }
  return Change;
}

} // namespace llvm::sandboxir
