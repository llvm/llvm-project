//===- AIRDemoteF64.h - Demote double to float ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRDEMOTEF64_H
#define LLVM_LIB_TARGET_AIR_AIRDEMOTEF64_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRDemoteF64Pass : public PassInfoMixin<AIRDemoteF64Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRDemoteF64Legacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRDemoteF64Legacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRDEMOTEF64_H
