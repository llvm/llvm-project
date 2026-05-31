//===- AIRSplitI64Shuffle.h - Split i64 simd shuffle ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The AIR GPU JIT crashes on i64 `air.simd_shuffle`. Split each i64
/// shuffle into two i32 shuffles (lo/hi halves) and recombine.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSPLITI64SHUFFLE_H
#define LLVM_LIB_TARGET_AIR_AIRSPLITI64SHUFFLE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRSplitI64ShufflePass
    : public PassInfoMixin<AIRSplitI64ShufflePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRSplitI64ShuffleLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRSplitI64ShuffleLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSPLITI64SHUFFLE_H
