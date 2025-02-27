//===- LegalizeI8Pass.h - A pass that reverts i8 conversions-*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_LEGALIZEI8_H
#define LLVM_TARGET_DIRECTX_LEGALIZEI8_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that transforms multidimensional arrays into one-dimensional arrays.
class LegalizeI8Pass : public PassInfoMixin<LegalizeI8Pass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_LEGALIZEI8_H
