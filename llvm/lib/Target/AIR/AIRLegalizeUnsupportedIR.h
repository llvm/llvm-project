//===- AIRLegalizeUnsupportedIR.h - Strip unsupported IR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRLEGALIZEUNSUPPORTEDIR_H
#define LLVM_LIB_TARGET_AIR_AIRLEGALIZEUNSUPPORTEDIR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRLegalizeUnsupportedIRPass
    : public PassInfoMixin<AIRLegalizeUnsupportedIRPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRLegalizeUnsupportedIRLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRLegalizeUnsupportedIRLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRLEGALIZEUNSUPPORTEDIR_H
