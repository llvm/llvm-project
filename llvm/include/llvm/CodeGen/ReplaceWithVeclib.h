//===- ReplaceWithVeclib.h - Replace vector instrinsics with veclib calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces calls to LLVM vector intrinsics (i.e., calls to LLVM intrinsics
// with vector operands) with matching calls to functions from a vector
// library (e.g., libmvec, SVML) according to TargetLibraryInfo.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_REPLACEWITHVECLIB_H
#define LLVM_TRANSFORMS_UTILS_REPLACEWITHVECLIB_H

#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"

namespace llvm {
class ReplaceWithVeclib : public PassInfoMixin<ReplaceWithVeclib> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// Legacy pass
class ReplaceWithVeclibLegacy : public FunctionPass {
public:
  static char ID;
  ReplaceWithVeclibLegacy() : FunctionPass(ID) {
    initializeReplaceWithVeclibLegacyPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

} // End namespace llvm
#endif // LLVM_TRANSFORMS_UTILS_REPLACEWITHVECLIB_H
