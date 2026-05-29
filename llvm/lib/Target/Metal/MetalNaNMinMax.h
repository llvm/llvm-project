//===- MetalNaNMinMax.h - NaN-propagating min/max lowering ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// llvm.minimum/maximum propagate NaN; AIR's air.fmin/fmax follow minnum
/// semantics (NaN is dropped). Rename the intrinsics and wrap each call in
/// an explicit NaN-check select so semantics are preserved.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALNANMINMAX_H
#define LLVM_LIB_TARGET_METAL_METALNANMINMAX_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MetalNaNMinMaxPass : public PassInfoMixin<MetalNaNMinMaxPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MetalNaNMinMaxLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalNaNMinMaxLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALNANMINMAX_H
