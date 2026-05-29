//===- MetalBFloat16CastDecompose.h - Decompose bf16 casts ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Decompose `sitofp`/`uitofp` to bfloat (and sub-32-bit int to float) into
/// integer/bit operations: Metal v1 bitcode treats sitofp iN->bfloat as if
/// targeting half, and cannot directly cast sub-32-bit integers to float.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALBFLOAT16CASTDECOMPOSE_H
#define LLVM_LIB_TARGET_METAL_METALBFLOAT16CASTDECOMPOSE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalBFloat16CastDecomposePass
    : public PassInfoMixin<MetalBFloat16CastDecomposePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalBFloat16CastDecomposeLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalBFloat16CastDecomposeLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALBFLOAT16CASTDECOMPOSE_H
