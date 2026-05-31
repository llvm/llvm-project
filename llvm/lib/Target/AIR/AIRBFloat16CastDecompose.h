//===- AIRBFloat16CastDecompose.h - Decompose bf16 casts ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Decompose `sitofp`/`uitofp` to bfloat (and sub-32-bit int to float) into
/// integer/bit operations: AIR v1 bitcode treats sitofp iN->bfloat as if
/// targeting half, and cannot directly cast sub-32-bit integers to float.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRBFLOAT16CASTDECOMPOSE_H
#define LLVM_LIB_TARGET_AIR_AIRBFLOAT16CASTDECOMPOSE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRBFloat16CastDecomposePass
    : public PassInfoMixin<AIRBFloat16CastDecomposePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRBFloat16CastDecomposeLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRBFloat16CastDecomposeLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRBFLOAT16CASTDECOMPOSE_H
