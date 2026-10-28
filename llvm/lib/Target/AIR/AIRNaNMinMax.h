//===- AIRNaNMinMax.h - NaN-propagating min/max lowering ------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_AIR_AIRNANMINMAX_H
#define LLVM_LIB_TARGET_AIR_AIRNANMINMAX_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRNaNMinMaxPass : public RequiredPassInfoMixin<AIRNaNMinMaxPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIRNaNMinMaxLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  /// With \p SignedZeros, llvm.minimum/maximum also order -0 below +0,
  /// which air.fmin/fmax do not on Metal (the Metal pipeline only).
  AIRNaNMinMaxLegacy(bool SignedZeros = false)
      : ModulePass(ID), SignedZeros(SignedZeros) {}
  static char ID;

private:
  bool SignedZeros;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRNANMINMAX_H
