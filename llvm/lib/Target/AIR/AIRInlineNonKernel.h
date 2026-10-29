//===- AIRInlineNonKernel.h - Inline all non-kernel functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AIR has no call stack (no recursion): the functions kernels call are
/// inlined, except large ones called from many places, which stay internal
/// functions called directly (AIRInlineNonKernel.cpp).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRINLINENONKERNEL_H
#define LLVM_LIB_TARGET_AIR_AIRINLINENONKERNEL_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// New-PM pass: inline the calls to defined functions, keeping the large
/// ones called from many places as functions.
class AIRInlineNonKernelPass
    : public RequiredPassInfoMixin<AIRInlineNonKernelPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Legacy-PM wrapper so the pass runs from TargetPassConfig under `llc`.
class AIRInlineNonKernelLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRInlineNonKernelLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRINLINENONKERNEL_H
