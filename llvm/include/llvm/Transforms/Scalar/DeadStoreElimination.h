//===- DeadStoreElimination.h - Dead Store Elimination ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides the interface for the Dead Store Elimination pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_DEADSTOREELIMINATION_H
#define LLVM_TRANSFORMS_SCALAR_DEADSTOREELIMINATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

/// Dead Store Elimination pass. Removes stores that have no observable effect.
class DSEPass : public OptionalPassInfoMixin<DSEPass> {
public:
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DEADSTOREELIMINATION_H
