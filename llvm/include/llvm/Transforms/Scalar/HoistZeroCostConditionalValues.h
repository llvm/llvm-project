//===- HoistZeroCostConditionalValues.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_HOISTZEROCOSTCONDITIONALVALUES_H
#define LLVM_TRANSFORMS_SCALAR_HOISTZEROCOSTCONDITIONALVALUES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
/// Hoist zero-cost conditional values and simplify merge PHIs.
class HoistZeroCostConditionalValuesPass
    : public OptionalPassInfoMixin<HoistZeroCostConditionalValuesPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_HOISTZEROCOSTCONDITIONALVALUES_H
