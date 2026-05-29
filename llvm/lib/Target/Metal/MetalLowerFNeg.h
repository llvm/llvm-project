//===- MetalLowerFNeg.h - Lower fneg to fsub --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The Metal GPU JIT does not support `fneg`; rewrite it to `fsub -0.0, x`.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALLOWERFNEG_H
#define LLVM_LIB_TARGET_METAL_METALLOWERFNEG_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalLowerFNegPass : public PassInfoMixin<MetalLowerFNegPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalLowerFNegLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalLowerFNegLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALLOWERFNEG_H
