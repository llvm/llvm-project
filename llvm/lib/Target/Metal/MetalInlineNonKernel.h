//===- MetalInlineNonKernel.h - Inline all non-kernel functions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Metal has no call stack: every non-kernel function must be inlined into its
/// callers before lowering to AIR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALINLINENONKERNEL_H
#define LLVM_LIB_TARGET_METAL_METALINLINENONKERNEL_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// New-PM pass: inline every call to a defined function (AIR has no call
/// stack).
class MetalInlineNonKernelPass
    : public PassInfoMixin<MetalInlineNonKernelPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

/// Legacy-PM wrapper so the pass runs from TargetPassConfig under `llc`.
class MetalInlineNonKernelLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  MetalInlineNonKernelLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALINLINENONKERNEL_H
