//===- AIRSystemValues.h - AIR system-value lowering -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Convert AIR system-value intrinsic calls (e.g.
/// `air.thread_position_in_grid`) into kernel function parameters and emit the
/// `!air.kernel` metadata that the AIR runtime needs to create a pipeline
/// state object.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSYSTEMVALUES_H
#define LLVM_LIB_TARGET_AIR_AIRSYSTEMVALUES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRSystemValuesPass
    : public PassInfoMixin<AIRSystemValuesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRSystemValuesLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRSystemValuesLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSYSTEMVALUES_H
