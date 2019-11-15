//===- InjectTLIMAppings.h - TLI to VFABI attribute injection  ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Populates the VFABI attribute with the scalar-to-vector mappings
// from the TargetLibraryInfo.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_INJECTTLIMAPPINGS_H
#define LLVM_TRANSFORMS_UTILS_INJECTTLIMAPPINGS_H

#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"

namespace llvm {
class InjectTLIMappings : public PassInfoMixin<InjectTLIMappings> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// Legacy pass
class InjectTLIMappingsLegacy : public FunctionPass {
public:
  static char ID;
  InjectTLIMappingsLegacy() : FunctionPass(ID) {
    initializeInjectTLIMappingsLegacyPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

} // End namespace llvm
#endif // LLVM_TRANSFORMS_UTILS_INJECTTLIMAPPINGS_H
