//===- AIRAsyncCopyToCooperative.h - lower async copy ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lower air.simdgroup_async_copy_2d into an inline cooperative threadgroup
/// copy (and air.wait_simdgroup_events into a threadgroup barrier) when the
/// module also uses air.simdgroup_matrix_8x8_* ops.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRASYNCCOPYTOCOOPERATIVE_H
#define LLVM_LIB_TARGET_AIR_AIRASYNCCOPYTOCOOPERATIVE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRAsyncCopyToCooperativePass
    : public PassInfoMixin<AIRAsyncCopyToCooperativePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRAsyncCopyToCooperativeLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRAsyncCopyToCooperativeLegacy() : ModulePass(ID) {}
  static char ID;
};

ModulePass *createAIRAsyncCopyToCooperativeLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRASYNCCOPYTOCOOPERATIVE_H
