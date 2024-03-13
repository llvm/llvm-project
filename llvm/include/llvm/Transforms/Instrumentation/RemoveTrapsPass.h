//===- RemoveTrapsPass.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for the pass responsible for removing
/// expensive ubsan checks.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_UBSANOPTIMIZATIONPASS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_UBSANOPTIMIZATIONPASS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

// This pass is responsible for removing optional traps, like llvm.ubsantrap
// from the hot code.
class RemoveTrapsPass : public PassInfoMixin<RemoveTrapsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif
